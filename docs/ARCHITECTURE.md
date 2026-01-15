# Architecture Overview

## CRSM: Continuous Reasoning State Model

CRSM is a language model architecture that combines Mamba-based efficient sequence modeling with continuous internal reasoning and asynchronous Monte Carlo Tree Search for advanced planning and autonomous operation.

## Core Components

### 1. Token Embedding Layer

- Converts discrete token IDs to dense embeddings
- Learnable embedding matrix of shape `[vocab_size, hidden_size]`
- Position embeddings for positional information (rotary or absolute)

### 2. S4/Mamba Layers

Efficient sequence modeling layers that operate in linear time:

- **State Space Model (S4)**: Compresses history into a state vector
- **Mamba**: Recent efficient SSM variant with selective state updates
- Typically stacked in `num_hidden_layers` blocks
- **Hierarchical Policy Fusion**: Instead of using only the final layer for the output policy, CRSM uses a **Learned Weighted Sum** of all layer states. This ensures the model integrates both high-level semantic rules and raw spatial/syntactic context.

### 3. Continuous Reasoning State

The continuous latent state $h(t)$ that:
- Maintains context across time with linear complexity
- Is updated via: $h_t = \bar{A}h_{t-1} + \bar{B}x_t$
- Supports asynchronous MCTS deliberation
- Enables proactive behavior through internal thresholds

The state matrix $\bar{A}$ is selected dynamically based on planning signals from the deliberation loop.

### 4. Deliberation Loop

Asynchronous MCTS/Tree-of-Thoughts module that:
- Runs parallel to token generation without blocking latency
- **Forward-Projected Planning**: Uses the dynamics model to "fast-forward" the latent state to the target position ($S_t \to S_{t+lag}$) before search begins.
- Explores multiple reasoning paths and future states
- Performs introspective node expansion for self-correction
- Generates planning signals that modulate the continuous state
- Produces soft guidance for output token selection

### 5. Output Projection

Linear layer projecting from hidden dimension to vocabulary:
- Input: `[batch, seq_len, hidden_size]`
- Output: `[batch, seq_len, vocab_size]`
- Logits are then passed to softmax for next-token prediction

### 6. Sparse-Gated Hierarchical Injection (Stability Mechanism)

**Note:** The primary innovation of CRSM is the **asynchronous interaction loop** between the System 1 backbone and the System 2 planner. The Gated Injection formula below is simply the stabilizing mechanism that makes this loop mathematically viable on a continuous manifold.

To safely integrate the asynchronous thought vectors into the sensitive Mamba manifold, CRSM employs **Sparse-Gated Hierarchical Injection**. Each layer is treated as a sovereign entity with its own independent gate.

The update rule for each layer $i$ is:
$h_{i,new} = (1 - \alpha_{i}) \cdot h_{i,old} + \alpha_{i} \cdot h_{i,target}$

Where:
- $h_{i,target}$ is the layer-specific state proposed by the **Multi-Layer Delta Broadcaster**.
- $\alpha_{i}$ is the effective injection rate for layer $i$, calculated using a Sigmoid gate over the **Multi-Headed Value Critic** output.
- Confidence is no longer a scalar; it is a vector representing how sure each level of abstraction is about the proposed plan.

This ensures that:
1.  **Sovereignty:** The planner can refine high-level logic (Strategy) without corrupting low-level syntax (Pixels).
2.  **Consensus:** The MCTS favors paths where all layers agree on the state's utility.

### 7. Targeted Delta Alignment

Planning occurs in parallel with generation. To ensure mathematical validity, a plan optimized for a future token position must be applied **exactly** at that position.

1.  **Buffering**: Planning results are stored in a **Targeted Delta Buffer**, keyed by the target generation step.
2.  **Application**: During the generation loop, the model checks the buffer. If a delta exists for the current step, it is injected immediately before the next token prediction.
3.  **Lag Correction**: If a plan arrives too late (generation has already passed the target step), it is decayed exponentially or pruned to prevent "Stale Thought" corruption.

## Configuration

Key hyperparameters in `CRSMConfig`:

```python
@dataclass
class CRSMConfig:
    vocab_size: int = 50257           # Vocabulary size
    hidden_size: int = 2048           # Model hidden dimension (continuous state size)
    intermediate_size: int = 8192     # FFN intermediate dimension
    num_hidden_layers: int = 24       # Number of Mamba SSM layers
    max_position_embeddings: int = 2048  # Max context length
    d_state: int = 256                # SSM state dimension
    dropout: float = 0.1              # Dropout rate
    mcts_depth: int = 8               # Max MCTS tree depth
    mcts_simulations: int = 32        # Simulations per deliberation step
    injection_rate: float = 0.05      # Gated injection rate (alpha) for state updates
    autonomy_threshold: float = 0.7   # Internal signal strength for autonomous action
```

## Training Strategy

### Two-Phase Training

1. **Base Model Pretraining**
   - Objective: Causal language modeling (predict next token)
   - Data: Large unsupervised corpus (e.g., SlimPajama)
   - Loss: Cross-entropy on all tokens

2. **Instruction Fine-tuning (SFT)**
   - Objective: Instruction following
   - Data: (instruction, response) pairs
   - Loss: Cross-entropy on response tokens only

### Loss Functions

**Causal Language Modeling**:
```
loss = CrossEntropyLoss(logits[:, :-1, :], labels[:, 1:])
```

Where:
- Logits: `[batch, seq_len - 1, vocab_size]` (predictions)
- Labels: `[batch, seq_len - 1]` (next tokens, with -100 for padding)

## Forward Pass

```
tokens (IDs)
    ↓
token_embedding
    ↓
position_encoding
    ↓
stack of [SSM_layer → state_machine → LN]
    ↓
output_projection
    ↓
logits [batch, seq_len, vocab_size]
    ↓
softmax (for inference)
    ↓
next_token (or full distribution)
```

## Memory and Compute

### Complexity Analysis

- **Sequence length**: L
- **Hidden dimension**: D
- **Vocab size**: V
- **Batch size**: B

Per-layer complexity:
- SSM layer: O(B × L × D) (linear in L, vs quadratic for attention)
- State machine: O(B × L × D_state) (compressed state tracking)
- Output projection: O(B × L × D × V)

### Memory Requirements

For ~2B parameter model:
- Model weights: ~8 GB (float32) / 4 GB (float16) / 2 GB (int8)
- Activations (batch_size=8, seq_len=2048): ~4-8 GB (float32/float16)
- Optimizer state (AdamW): ~12 GB (float32)

**Total for training**: ~16-24 GB GPU memory (fp32) or 8-12 GB (fp16)

## Inference Optimizations

1. **Batching**: Process multiple sequences in parallel
2. **KV caching**: Store hidden states for efficient generation (if applicable)
3. **Quantization**: INT8 or lower-bit quantization for inference
4. **Compilation**: Use torch.compile() for faster forward passes
5. **Pruning**: Remove less important heads/layers (optional)

## Extending CRSM

### Adding Custom Layers

```python
class CRSMModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            S4Layer(config) for _ in range(config.num_hidden_layers)
        ])
        self.output_proj = nn.Linear(config.hidden_size, config.vocab_size)
    
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        logits = self.output_proj(x)
        return logits
```

### Modifying State Machine

Replace the simple state accumulation with:
- Attention-based state updates
- Learned state mixing
- External memory integration

## Performance Benchmarks

Expected throughput (on A100):

| Configuration | Seq Len | Batch Size | Throughput (tokens/sec) | Memory (GB) |
|--------------|---------|-----------|------------------------|------------|
| Base 2B | 512 | 64 | ~50k | 20 |
| Base 2B | 2048 | 8 | ~15k | 24 |
| Base 2B | 2048 | 1 | ~2k | 6 |

## References

- **S4**: [Efficiently Modeling Long Sequences with Structured State Spaces](https://arxiv.org/abs/2111.00396)
- **Mamba**: [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.08956)
- **MuZero**: [Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model](https://arxiv.org/abs/1911.08265)
- **RepE**: [Representation Engineering: A Top-Down Approach to AI Transparency](https://arxiv.org/abs/2310.01405)
- **AlphaLLM**: [AlphaLLM: Monte Carlo Tree Search with Large Language Models](https://arxiv.org/abs/2404.05584)
- **Scaling Laws**: [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
