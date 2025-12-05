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
from crsm.model import CRSMModel, CRSMConfig

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

## Training

### 1. Prepare Data
Ensure you have a text corpus in `data/text_corpus`.

### 2. Train Backbone
```bash
python scripts/training/train_full_crsm.py --config configs/baseline_27m.json
```

### 3. Configuration
Edit `configs/baseline_27m.json` to tune hyperparameters.

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
