#!/bin/bash
# Quick start script to test the full CRSM pipeline
# This runs a minimal training to verify everything works

set -e  # Exit on error

echo "=================================="
echo "CRSM Quick Start Test"
echo "=================================="
echo ""

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  Warning: No virtual environment detected"
    echo "   Consider running: source venv/bin/activate"
    echo ""
fi

# Create configs directory if it doesn't exist
mkdir -p configs

# Check if small.json exists
if [ ! -f "configs/small.json" ]; then
    echo "Creating configs/small.json..."
    cat > configs/small.json << 'EOF'
{
  "vocab_size": 1000,
  "d_model": 128,
  "d_state": 64,
  "d_ffn": 512,
  "num_layers": 2,
  "dropout": 0.1,
  "backbone_epochs": 2,
  "finetune_epochs": 1,
  "batch_size": 8,
  "seq_len": 32,
  "lr": 0.001,
  "finetune_lr": 0.0005,
  "dynamics_samples": 1000,
  "dynamics_epochs": 3,
  "dynamics_lr": 0.001,
  "c_puct": 1.0,
  "n_simulations": 5,
  "seed": 42,
  "device": null
}
EOF
fi

echo "Step 1: Testing basic imports..."
python -c "from crsm.model import CRSM; from crsm.latent_dynamics import LatentDynamics; print('✓ Imports OK')"

echo ""
echo "Step 2: Running quick training test (1 epoch)..."
python -m crsm.cli train \
    --epochs 1 \
    --batch-size 4 \
    --vocab-size 100 \
    --seq-len 16 \
    --checkpoint-dir experiments/quickstart/backbone \
    --no-wandb

echo ""
echo "Step 3: Testing dynamics distillation..."
python scripts/training/distill_dynamics.py \
    --model-path experiments/quickstart/backbone/crsm_epoch1.pt \
    --output-path experiments/quickstart/dynamics.pt \
    --num-samples 100 \
    --epochs 2 \
    --vocab-size 100 \
    --d-model 128 \
    --num-layers 2

echo ""
echo "Step 4: Testing dynamics loading..."
python << 'PYEOF'
import torch
from crsm.model import CRSM
from crsm.load_dynamics import load_dynamics_into_crsm, check_dynamics_quality

crsm = CRSM(vocab_size=100, d_model=128, num_layers=2)
success = load_dynamics_into_crsm(crsm, 'experiments/quickstart/dynamics.pt')

if success:
    print("✓ Dynamics loaded successfully")
    stats = check_dynamics_quality(crsm, test_samples=10)
    print(f"✓ Quality check passed (MSE: {stats['avg_mse']:.6f})")
else:
    print("✗ Failed to load dynamics")
    exit(1)
PYEOF

echo ""
echo "Step 5: Running pipeline tests..."
python -m pytest tests/test_full_pipeline.py -v --tb=short

echo ""
echo "=================================="
echo "✓ Quick Start Complete!"
echo "=================================="
echo ""
echo "All tests passed. Your CRSM installation is working correctly."
echo ""
echo "Next steps:"
echo "  1. Train a real model: python scripts/training/train_full_crsm.py --config configs/small.json"
echo "  2. Run full test suite: pytest tests/ -v"
echo "  3. Check the documentation: cat docs/USAGE.md"
echo ""