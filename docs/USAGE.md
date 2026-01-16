# CRSM Usage Guide

## Installation

```bash
git clone https://github.com/pomilon/CRSM
cd CRSM
pip install -e .
```

## 🚀 Quick Start: Autonomous Inference

> **Note:** Pre-trained weights are not yet available. The code below will run with an initialized (random) model for testing purposes.

```python
import torch
from crsm.core import CRSMModel, CRSMConfig

# Load model
config = CRSMConfig(vocab_size=50257, injection_rate=0.05)
model = CRSMModel(config).cuda()

# Generate with thinking
prompt = torch.tensor([[50, 12, 99]]).cuda() # Example token IDs
output = await model.crsm.think_and_generate(
    prompt, 
    max_length=50, 
    use_deliberation=True,  # Enable MCTS
    deliberation_lag=3      # Plan 3 steps ahead
)
print(output)
```

# ARC-AGI Benchmarking

CRSM includes a specialized benchmarking suite for grid-based reasoning tasks.

### 1. Generate Sanity Tasks
```bash
python crsm/data/arc_gen.py
```
This generates synthetic tasks in `data/arc_sanity/` including Identity, Reflection, and Scaling.

### 2. Run Phase 1 (Sanity)
Evaluates the model's ability to learn deterministic rules and grid syntax.
```bash
python scripts/eval/arc_benchmark.py --config configs/arc_nano.yaml --phase 1 --seeds 42 43 44
```

### 3. Run Phase 2 (Ablations)
Compares the full reasoning model against greedy decoding and models without critics/projection.
```bash
python scripts/eval/arc_benchmark.py --config configs/arc_nano.yaml --phase 2
```

---

## 🛠️ Training Pipeline

### 1. Prepare Data

Before training, you need a text corpus. We provide a script to download and preprocess datasets (FineWeb-Edu, GSM8K) into efficient binary format.

```bash
# Install dependencies
pip install datasets transformers

# Download and prepare FineWeb-Edu (Sample)
python scripts/data/prepare_dataset.py --dataset fineweb --output-dir data/fineweb --shard-size 10000000

# Download and prepare GSM8K
python scripts/data/prepare_dataset.py --dataset gsm8k --output-dir data/gsm8k
```

This will create `*.bin` files in the specified directories.

### 2. Train Backbone (Stage 1)
```bash
python scripts/training/stage_1_backbone.py --config configs/baseline_27m.yaml
```

### 3. Dynamics Distillation (Stage 2)
```bash
python scripts/training/stage_2_dynamics.py --config configs/baseline_27m.yaml
```

### 4. Value Head Training (Stage 3)
```bash
python scripts/training/stage_3_value_head.py --config configs/baseline_27m.yaml
```

### 5. Assembly (Stage 4)
```bash
python scripts/training/stage_4_assembly.py --config configs/baseline_27m.yaml
```

### 6. Configuration
Edit `configs/baseline_27m.yaml` to tune hyperparameters.

**Key Parameters:**
*   `injection_rate`: (Float, 0.0-1.0) How strongly MCTS affects the state. Default `0.05`.
*   `n_simulations`: (Int) MCTS compute budget per step. Default `50`.
*   `autonomous_mode`: (Bool) Whether to run MCTS background loop.

## Monitoring

Use WandB to track:
*   `loss`: Main language modeling loss.
*   `value_loss`: Accuracy of the MCTS value predictor.
*   `dynamics_loss`: Fidelity of the world model.

## Debugging

If the model generates nonsense or the loss explodes:
1.  Check `injection_rate`. If > 0.1, reduce it.
2.  Ensure `dynamics_model` is loaded. MCTS needs it to plan.
3.  Run `python tests/test_architecture_stability.py` to verify the build.
