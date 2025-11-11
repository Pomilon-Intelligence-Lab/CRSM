# CRSM Usage Guide

Complete guide to training and using CRSM models with learned dynamics.

## Quick Start

```bash
# 1. Test your installation
bash scripts/quickstart.sh

# 2. Train a small model
python scripts/train_full_crsm.py --config configs/small.json

# 3. Use the trained model
python scripts/inference.py --model experiments/full_crsm/crsm_with_dynamics.pt
```

## Training Pipeline

CRSM training has 4 stages:

### Stage 1: Train Base Backbone

Train a standard MambaModel with language modeling objective:

```bash
python -m crsm.cli train \
    --epochs 10 \
    --batch-size 32 \
    --vocab-size 32000 \
    --seq-len 512 \
    --lr 1e-3 \
    --checkpoint-dir experiments/backbone \
    --no-value-loss
```

**Output**: `experiments/backbone/crsm_epoch*.pt`

### Stage 2: Distill Dynamics Model

Train a learned dynamics function `f_θ(h, a) → Δh` from the backbone:

```bash
python scripts/distill_dynamics.py \
    --model-path experiments/backbone/crsm_epoch10.pt \
    --output-path experiments/dynamics.pt \
    --num-samples 50000 \
    --epochs 10 \
    --lr 1e-3 \
    --vocab-size 32000 \
    --d-model 2048 \
    --num-layers 24
```

**Output**: `experiments/dynamics.pt` (learned transition model)

### Stage 3: Create CRSM with Dynamics

Load dynamics into CRSM:

```python
from crsm.model import CRSM
from crsm.load_dynamics import load_dynamics_into_crsm, check_dynamics_quality

# Create CRSM
crsm = CRSM(vocab_size=32000, d_model=2048, num_layers=24)

# Load backbone weights
backbone_ckpt = torch.load('experiments/backbone/crsm_epoch10.pt')
crsm.backbone.load_state_dict(backbone_ckpt['model_state'])

# Load dynamics
load_dynamics_into_crsm(crsm, 'experiments/dynamics.pt')

# Verify quality
stats = check_dynamics_quality(crsm, test_samples=1000)
print(f"Dynamics MSE: {stats['avg_mse']:.6f}")

# Save combined model
from crsm.load_dynamics import save_crsm_with_dynamics
save_crsm_with_dynamics(crsm, 'experiments/crsm_complete.pt')
```

### Stage 4: Fine-tune with Value Head

Train value head for state quality estimation:

```bash
python -m crsm.cli train \
    --epochs 5 \
    --batch-size 16 \
    --vocab-size 32000 \
    --seq-len 512 \
    --lr 5e-4 \
    --checkpoint-dir experiments/final \
    --resume experiments/backbone/crsm_epoch10.pt
    # Value loss enabled by default
```

## Full Pipeline (Automated)

Run all stages with one command:

```bash
python scripts/train_full_crsm.py --config configs/medium.json --output-dir experiments/my_model
```

### Configuration File Format

```json
{
  "vocab_size": 32000,
  "d_model": 2048,
  "d_state": 256,
  "d_ffn": 8192,
  "num_layers": 24,
  "dropout": 0.1,
  
  "backbone_epochs": 10,
  "finetune_epochs": 5,
  "batch_size": 32,
  "seq_len": 512,
  "lr": 1e-3,
  "finetune_lr": 5e-4,
  
  "dynamics_samples": 50000,
  "dynamics_epochs": 10,
  "dynamics_lr": 1e-3,
  
  "c_puct": 1.0,
  "n_simulations": 50,
  
  "seed": 42,
  "device": "cuda"
}
```

### Skip Stages

```bash
# Skip backbone training (use existing)
python scripts/train_full_crsm.py --config configs/medium.json --skip-backbone

# Skip dynamics distillation
python scripts/train_full_crsm.py --config configs/medium.json --skip-dynamics

# Skip fine-tuning
python scripts/train_full_crsm.py --config configs/medium.json --skip-finetune
```

## Inference

### Basic Generation

```python
import torch
from crsm.model import CRSM
from crsm.load_dynamics import load_dynamics_into_crsm

# Load model
crsm = CRSM(vocab_size=32000, d_model=2048, num_layers=24)
checkpoint = torch.load('experiments/crsm_complete.pt')
crsm.load_state_dict(checkpoint['model_state'])

# If dynamics checkpoint exists separately
load_dynamics_into_crsm(crsm, 'experiments/dynamics.pt')

# Generate with reasoning
import asyncio

prompt = torch.tensor([[1, 2, 3, 4, 5]])  # Your tokenized prompt
output = asyncio.run(crsm.think_and_generate(prompt, max_length=100))

print(output)  # Generated token IDs
```

### Compare With/Without Dynamics

```python
# Generate WITHOUT dynamics (SSM forward only)
if hasattr(crsm, 'dynamics'):
    delattr(crsm, 'dynamics')  # Remove dynamics

output_no_dynamics = asyncio.run(crsm.think_and_generate(prompt, max_length=100))

# Reload dynamics
load_dynamics_into_crsm(crsm, 'experiments/dynamics.pt')

# Generate WITH dynamics (learned transitions)
output_with_dynamics = asyncio.run(crsm.think_and_generate(prompt, max_length=100))

# Compare
print("Without dynamics:", output_no_dynamics)
print("With dynamics:", output_with_dynamics)
```

## Evaluation

### Dynamics Quality

```python
from crsm.load_dynamics import check_dynamics_quality

stats = check_dynamics_quality(crsm, test_samples=1000)

print(f"Average MSE: {stats['avg_mse']:.6f}")
print(f"Average MAE: {stats['avg_mae']:.6f}")

# Interpretation:
# MSE < 0.01: Excellent
# MSE < 0.1: Good
# MSE < 1.0: Acceptable
# MSE > 1.0: Poor (retrain dynamics)
```

### Planning Quality

Test on reasoning benchmarks:

```python
# Test on GSM8K, MATH, ARC, etc.
# Compare CRSM vs baseline Mamba

# Measure:
# - Accuracy on reasoning tasks
# - Planning depth (MCTS tree statistics)
# - State delta magnitudes
# - Value head calibration
```

## Advanced Usage

### Custom Dynamics Architecture

Replace `LatentDynamics` with your own:

```python
from crsm.latent_dynamics import LatentDynamics
import torch.nn as nn

class CustomDynamics(nn.Module):
    def __init__(self, d_model, action_dim):
        super().__init__()
        # Your architecture here
        self.net = nn.Sequential(
            nn.Linear(d_model + action_dim, d_model * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model)
        )
    
    def forward(self, state, action_emb):
        return self.net(torch.cat([state, action_emb], -1))

# Use in CRSM
crsm.dynamics = CustomDynamics(d_model=2048, action_dim=2048)
```

### Hierarchical Dynamics (Per-Layer)

```python
class HierarchicalDynamics(nn.Module):
    def __init__(self, layer_dims):
        super().__init__()
        self.layer_dynamics = nn.ModuleList([
            LatentDynamics(d_model=dim) for dim in layer_dims
        ])
    
    def forward(self, states, action_emb):
        # states is a list of per-layer states
        deltas = []
        for layer_dyn, state in zip(self.layer_dynamics, states):
            delta = layer_dyn(state, action_emb)
            deltas.append(delta)
        return deltas
```

### Multi-Step Rollouts

```python
def multi_step_rollout(crsm, initial_state, actions):
    """Rollout dynamics for multiple steps"""
    state = initial_state
    trajectory = [state]
    
    for action in actions:
        action_emb = crsm.backbone.embedding(torch.tensor([[action]])).squeeze(1)
        
        next_state = []
        for layer_state in state:
            delta = crsm.dynamics(layer_state, action_emb)
            next_state.append(layer_state + delta)
        
        state = next_state
        trajectory.append(state)
    
    return trajectory
```

## Troubleshooting

### Dynamics Quality is Poor

**Symptoms**: MSE > 1.0, planning doesn't improve quality

**Solutions**:
1. Collect more transitions: `--num-samples 100000`
2. Train longer: `--epochs 20`
3. Increase model capacity: Larger MLP in `LatentDynamics`
4. Use per-layer dynamics instead of shared
5. Check backbone is properly trained first

### State Deltas Too Small

**Symptoms**: Delta magnitudes < 1e-8, no effect on generation

**Solutions**:
1. Increase delta scaling in `_compute_delta_from_mcts`:
   ```python
   delta = delta * 0.5  # Increase from 0.1
   ```
2. Use more MCTS simulations: `n_simulations=100`
3. Increase `c_puct` for more exploration: `c_puct=2.0`

### MCTS Not Using Dynamics

**Symptoms**: Generation same with/without dynamics

**Verify**:
```python
# Check if dynamics is attached
print(f"Has dynamics: {hasattr(crsm, 'dynamics')}")

# Check if reasoning finds it
print(f"Model has dynamics: {hasattr(crsm.reasoning.model, 'dynamics')}")
```

**Fix**: Ensure `crsm.dynamics` exists before creating reasoning module, or recreate reasoning:
```python
crsm.dynamics = LatentDynamics(d_model=2048)
crsm.reasoning = AsyncDeliberationLoop(crsm.backbone, c_puct=1.0, n_simulations=50)
```

### Memory Issues

**Symptoms**: OOM during dynamics distillation

**Solutions**:
1. Reduce batch size in collection: Modify `collect_transitions()`
2. Reduce `--num-samples`
3. Use CPU for distillation: `--device cpu`
4. Clear cache periodically in training loop

## Best Practices

### Training Schedule

1. **Week 1**: Train backbone to convergence (perplexity plateaus)
2. **Week 2**: Distill dynamics (wait for low MSE < 0.1)
3. **Week 3**: Fine-tune with value head (monitor value calibration)
4. **Week 4**: Evaluate on reasoning benchmarks

### Hyperparameter Tuning

Start with:
- `dynamics_samples`: 50,000 (increase if MSE high)
- `dynamics_epochs`: 10 (early stop on validation)
- `c_puct`: 1.0 (increase for more exploration)
- `n_simulations`: 50 (increase for better planning)
- Delta scaling: 0.1 (tune based on magnitude)

### Validation

Check these during training:
- Backbone perplexity decreasing
- Dynamics MSE < 0.1
- Value head MSE < 0.5
- Delta magnitudes in range [1e-4, 1e-2]
- MCTS tree depth > 3

## Example Workflows

### Research: Quick Iteration

```bash
# Fast cycle for testing
python scripts/train_full_crsm.py --config configs/small.json
# 30 minutes on single GPU
```

### Production: Quality Model

```bash
# Full training for best results
python scripts/train_full_crsm.py --config configs/large.json --output-dir models/crsm_2b
# Several days on 8x A100
```

### Ablation Studies

```bash
# Baseline: No dynamics
python -m crsm.cli train --config configs/base.json --no-value-loss

# CRSM: With dynamics
python scripts/train_full_crsm.py --config configs/base.json

# Compare perplexity and reasoning accuracy
```

## Next Steps

- Check [ROADMAP.md](ROADMAP.md) for planned features
- See [ARCHITECTURE.md](ARCHITECTURE.md) for technical details
- Join discussions on GitHub Issues
- Contribute improvements via Pull Requests