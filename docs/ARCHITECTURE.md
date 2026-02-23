# Architecture Overview

## CRSM: Continuous Reasoning State Model

CRSM is a modular architecture that combines Mamba-based sequence modeling with asynchronous MCTS planning. The project is decoupled into four primary domains to support rapid experimentation on tasks like ARC-AGI.

---

## 🏗️ Modular Structure

### 1. `core/` (Backbone & Planner)
Pure implementation of the CRSM machinery, free from data-loading or task-specific logic.
- **`mamba.py`**: The Mamba backbone, including **Multi-Headed Value Critics (MV-Critic)** and **Hierarchical Policy Fusion**.
- **`reasoning.py`**: The MCTS implementation, featuring **Forward-Projected Planning**.
- **`dynamics.py`**: The Recurrent World Model (GRU-based Broadcaster).
- **`crsm.py`**: The top-level wrapper managing the asynchronous interaction between generation and deliberation.

### 2. `tasks/` (Domain Logic)
Abstracts the "What" from the "How." Each task implements its own loss function and data-loading requirements.
- **`lm_task.py`**: Causal language modeling with Value Critic supervision and Hierarchical Entropy loss.
- **`distillation.py`**: Stage 2 Dynamics distillation, training the Broadcaster to predict state residuals.
- **`arc_task.py`**: (Upcoming) Grid-based reasoning logic for ARC-AGI benchmarking.

### 3. `training/` (The Engine)
A generic, task-agnostic training engine (`Trainer`) that handles gradient accumulation, mixed precision, and checkpointing.

### 4. `data/` (Providers)
Handles raw data access, sharding, and tokenization.

---

## Core Components

### 1. Token Embedding Layer
Converts discrete token IDs to dense embeddings. For Nano-scale models, this often accounts for a significant portion of the parameter budget.

### 2. S4/Mamba Layers
Efficient sequence modeling layers that operate in linear time.
- **Hierarchical Policy Fusion**: Instead of using only the final layer for the output policy, CRSM uses a **Learned Weighted Sum** of all layer states. This ensures the model integrates both high-level semantic rules and raw spatial/syntactic context.

### 3. Continuous Reasoning State
The continuous latent state $h(t)$ that:
- Maintains context across time with linear complexity.
- Supports asynchronous MCTS deliberation.
- Is modified via **Sparse-Gated Hierarchical Injection**.

### 4. Deliberation Loop
Asynchronous MCTS module that runs parallel to token generation.
- **Forward-Projected Planning**: Uses the dynamics model to "fast-forward" the latent state to the target position ($S_t \to S_{t+lag}$) before search begins.
- **Targeted Delta Buffer**: Stores planning results and injects them at the **exact** step they were optimized for, resolving asynchronous drift.

---

## 📊 Sparse-Gated Hierarchical Injection

To safely integrate the asynchronous thought vectors into the sensitive Mamba manifold, CRSM employs **Sparse-Gated Hierarchical Injection**. Each layer is treated as a sovereign entity with its own independent gate.

The update rule for each layer $i$ is:
$h_{i,new} = (1 - \alpha_{i}) \cdot h_{i,old} + \alpha_{i} \cdot h_{i,target}$

Where:
- $h_{i,target}$ is the layer-specific state proposed by the **Multi-Layer Delta Broadcaster**.
- $\alpha_{i}$ is the effective injection rate for layer $i$, calculated using a Sigmoid gate over the **Multi-Headed Value Critic** output.

This ensures that the planner can refine high-level logic (Strategy) without corrupting low-level syntax (Pixels).

---

## 🎯 Targeted Delta Alignment

Planning occurs in parallel with generation. To ensure mathematical validity, a plan optimized for a future token position must be applied **exactly** at that position.

1.  **Buffering**: Planning results are stored in a **Targeted Delta Buffer**, keyed by the target generation step.
2.  **Application**: During the generation loop, the model checks the buffer. If a delta exists for the current step, it is injected immediately before the next token prediction.
3.  **Lag Correction**: If a plan arrives too late (generation has already passed the target step), it is decayed exponentially or pruned to prevent "Stale Thought" corruption.

---

## Configuration

Key hyperparameters in `CRSMConfig`:

```python
@dataclass
class CRSMConfig:
    vocab_size: int = 1024            # Optimized for Nano-scale
    hidden_size: int = 256            # Model hidden dimension
    intermediate_size: int = 1024     # FFN intermediate dimension
    num_hidden_layers: int = 4        # Hierarchy depth
    d_state: int = 64                 # SSM state dimension
    injection_rate: float = 0.05      # Max injection alpha
```

---

## Training Strategy

CRSM uses a multi-stage pipeline orchestrated by the unified `run.py` entry point.

1. **Backbone Pretraining**: Standard CLM training using `LanguageModelingTask`.
2. **Dynamics Distillation**: Training the Broadcaster using `DistillationTask`.
3. **Value Head Fine-tuning**: Offline reinforcement learning to train the MV-Critics.
4. **Task-Specific Alignment**: Fine-tuning on ARC-AGI or other reasoning benchmarks.

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
