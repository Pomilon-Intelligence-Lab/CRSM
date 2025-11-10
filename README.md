# CRSM: Continuous Reasoning State Model

> ⚠️ **EXPERIMENTAL PROJECT**: This is an active research implementation. The project is still in the process of validation and proof-of-concept. We are working to produce a trained model that can verify the validity of this architecture and approach. Contributions, feedback, and validation attempts are welcome.

A next-generation autonomous language model architecture designed to overcome the fundamental limitations of Transformer-based autoregressive models. CRSM combines Mamba (State Space Models) for efficient continuous latent reasoning with Monte Carlo Tree Search (MCTS) for asynchronous deliberation, enabling low-latency reasoning and autonomous operation.

## Vision

The CRSM architecture represents a paradigm shift from sequential token prediction to continuous internal reasoning. Instead of relying on next-token prediction, the CRSM features:

- **Always-On Internal Reasoning**: Continuous latent state $h(t)$ that constantly models context and performs planning
- **Asynchronous Deliberation**: MCTS/Tree-of-Thoughts module runs in parallel with token generation for deep planning without sacrificing latency
- **Autonomous Operation**: Proactive behavior generation through internal state thresholds—the model can initiate actions, self-correct, and intervene without explicit user prompts
- **Linear Scaling**: Mamba backbone achieves O(N) complexity vs Transformer's O(N²), enabling efficient long-context processing

## Key Features

- **Mamba-Based Foundation**: Linear-time sequence modeling with continuous state representation
- **Integrated MCTS Deliberation**: Structured search for lookahead planning and self-correction
- **Strategic Knowledge Distillation**: Latent-SFT training with Prompt Erasure to cultivate genuine reasoning
- **Proof-of-Concept**: ~2B parameter model optimized for training on consumer GPUs (A100/V100 or Colab)
- **Autonomous Modes**: Toggleable autonomy for proactive intervention and self-correction
- **Production Ready**: Full training pipeline with mixed precision, gradient accumulation, checkpointing, and evaluation metrics

## Project Structure

```
.
├── crsm/                          # Core CRSM package
│   ├── __init__.py
│   ├── model.py                   # CRSMConfig, CRSMModel, and core architecture
│   ├── tokenizer.py               # Tokenizer wrapper (HF + fallback)
│   ├── dataset.py                 # StreamingTextDataset and in-memory datasets
│   ├── latent.py                  # Latent reasoning state management
│   ├── distill.py                 # Knowledge distillation utilities
│   ├── train.py                   # Training utilities and loops
│   ├── utils.py                   # Helper functions
│   └── s4_adapter.py              # S4/Mamba layer adapter
├── notebooks/
│   └── colab_train_crsm_2b.ipynb  # Full Colab training pipeline
├── tests/
│   ├── __init__.py
│   ├── test_crsm.py               # Model and config tests
│   ├── test_dataset_stream.py     # Dataset tests
│   └── test_tokenizer.py          # Tokenizer tests
├── docs/                          # Documentation
├── examples/                      # Usage examples
├── setup.py                       # Package setup
├── requirements.txt               # Dependencies
├── README.md                      # This file
├── LICENSE                        # MIT License
└── .gitignore                     # Git ignore rules
```

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/pomilon/CRSM.git
cd CRSM
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package:
```bash
pip install -e .
```

Or install dependencies directly:
```bash
pip install -r requirements.txt
```

## Quick Start

### Training on Colab

The easiest way to get started is using the provided Colab notebook:

1. Open `notebooks/colab_train_crsm_2b.ipynb` in Google Colab
2. Follow the cells to:
   - Set up GPU environment
   - Load datasets (SlimPajama for base training, OpenAssistant for SFT)
   - Train the base model
   - Fine-tune on instruction data
   - Evaluate and export

### Training Locally

```python
from crsm.model import CRSMConfig, CRSMModel
from crsm.tokenizer import Tokenizer
from crsm.dataset import StreamingTextDataset
from torch.utils.data import DataLoader
import torch

# Configure model
config = CRSMConfig(
    vocab_size=32000,
    hidden_size=2048,
    num_hidden_layers=24,
    num_attention_heads=16,
    max_position_embeddings=2048,
)

# Initialize model and tokenizer
model = CRSMModel(config)
tokenizer = Tokenizer()

# Load data
dataset = StreamingTextDataset(
    dataset_name="cerebras/SlimPajama-627B",
    seq_len=2048,
    tokenizer=tokenizer
)
dataloader = DataLoader(dataset, batch_size=8)

# Move to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training loop
model.train()
for batch in dataloader:
    input_ids = batch['input_ids'].to(device)
    labels = batch['labels'].to(device)
    
    logits, _ = model(input_ids)
    # Compute loss and backprop
    loss = loss_fn(logits.reshape(-1, config.vocab_size), labels.reshape(-1))
    loss.backward()
```

## Model Architecture

The CRSM architecture combines:

1. **Token Embedding Layer**: Converts tokens to dense vectors
2. **S4/Mamba Layers**: Efficient sequence modeling with linear complexity
3. **State Machine**: Continuous reasoning state tracking
4. **Output Projection**: Projects hidden states to vocabulary

### Configuration

Key hyperparameters in `CRSMConfig`:

- `vocab_size`: Vocabulary size (default: 32,000)
- `hidden_size`: Hidden dimension (default: 2048 for ~2B params)
- `num_hidden_layers`: Number of S4/Mamba blocks (default: 24)
- `intermediate_size`: Feed-forward hidden dimension (default: 8192)
- `max_position_embeddings`: Context length (default: 2048)
- `d_state`: S4 state dimension (default: 256)

## Training Configuration

### Recommended Settings by GPU

| GPU | Batch Size | Gradient Accumulation | Mixed Precision |
|-----|------------|----------------------|-----------------|
| A100 (40GB) | 32 | 4 | BF16 |
| V100 (32GB) | 16 | 8 | FP16 |
| T4 (16GB) | 8 | 16 | FP16 |
| Colab A100 | 16 | 8 | FP16 |

### Key Flags

- `--precision 16-mixed`: Use mixed precision training
- `--accumulate_grad_batches N`: Gradient accumulation steps
- `--val_check_interval 0.25`: Validate 4 times per epoch

## Data Formats

### Base Model Training

The model expects sequences with:
- `input_ids`: Token IDs (shape: [batch, seq_len])
- `labels`: Next-token labels (shape: [batch, seq_len]), with -100 for padding/ignore

Supported datasets:
- `cerebras/SlimPajama-627B` (default, streaming)
- `wikitext` variant
- Custom `.jsonl` files with `"text"` field

### Instruction Fine-tuning

Format: `instruction -> input -> output`

Supported datasets:
- `OpenAssistant/oasst1`
- Custom format: `[{"instruction": "...", "input": "...", "output": "..."}, ...]`

## Testing

Run tests to validate the setup:

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_crsm.py

# Run with coverage
pytest tests/ --cov=crsm
```

## Evaluation

The training pipeline includes built-in evaluation:

- **Perplexity**: Computed on validation set during training
- **BLEU**: For instruction model evaluation
- **ROUGE**: For summarization/instruction quality

```python
from evaluate import load

rouge = load('rouge')
bleu = load('bleu')

# Example
predictions = ["generated text"]
references = ["reference text"]
rouge_scores = rouge.compute(predictions=predictions, references=references)
bleu_score = bleu.compute(predictions=[pred.split() for pred in predictions],
                          references=[[ref.split()] for ref in references])
```

## Troubleshooting

### CUDA Out of Memory

- Reduce `batch_size`
- Increase `accumulate_grad_batches`
- Enable gradient checkpointing: `model.gradient_checkpointing_enable()`
- Use mixed precision: `precision='16-mixed'`

### Slow Training

- Ensure `pin_memory=False` in DataLoader (PyTorch Lightning default)
- Check GPU utilization: `nvidia-smi`
- Increase batch size if memory allows
- Use `num_workers > 0` in DataLoader

### Data Loading Issues

- For streaming datasets, ensure internet connectivity
- Cache first batch locally with `dataset.take(N)`
- Verify dataset format matches expected schema

## Contributing

Contributions are welcome! Areas for improvement:

1. **Performance**: Optimize S4/Mamba layers, add kernel fusions
2. **Features**: Add distillation loss, advanced reasoning states, multi-GPU DDP
3. **Evaluation**: Expand benchmarks, add reasoning-specific metrics
4. **Documentation**: Add more examples, tutorials, architecture docs

Please submit PRs with:
- Clear commit messages
- Tests for new features
- Updated documentation

## Citation

If you use CRSM in your research, please cite:

```bibtex
@software{crsm2025,
  title = {CRSM: Continuous Reasoning State Machine},
  author = {Pomilon},
  year = {2025},
  url = {https://github.com/pomilon/CRSM}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [PyTorch](https://pytorch.org/) and [PyTorch Lightning](https://lightning.ai/)
- Datasets from [Hugging Face](https://huggingface.co/)
- Inspired by recent work on state machines and efficient LLMs

## Roadmap

- [ ] Multi-GPU training (DDP)
- [ ] Distributed training (Slurm integration)
- [ ] Advanced reasoning modules
- [ ] Model quantization (GPTQ, AWQ)
- [ ] Inference optimization (vLLM integration)
- [ ] Web UI for inference
- [ ] Pretrained model checkpoints on Hub

## Contact

For questions or issues, open a GitHub issue or reach out via email.

---

**Note**: This is a proof-of-concept implementation. For production use, additional hardening, testing, and optimization is recommended.

4) Reproducible run harness

Create a JSON config and run:

```bash
python -m crsm.run --config config.json --run-dir runs/my_experiment
```

Notes:
- Only rank 0 writes checkpoints and logs when using distributed training.
- The `distill_pipeline` can optionally compute embeddings using a local HF embedding model if available.
- For large-scale runs, use streaming datasets and shard prompts to multiple workers / machines.

Next steps (planned):
- Add batched GPU-parallel MCTS evaluation
- Integrate distributed training (DDP) and AMP
- Add dataset streaming and large-scale distillation tooling
