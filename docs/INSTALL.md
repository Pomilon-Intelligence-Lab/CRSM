# Installation & Setup Guide

## System Requirements

- **OS**: Linux, macOS, or Windows (WSL2 recommended)
- **Python**: 3.10, 3.11, or 3.12
- **GPU** (optional): NVIDIA CUDA 11.8+ for GPU training
- **RAM**: 16 GB minimum (32 GB recommended for training)
- **Disk**: 50 GB free space for datasets and checkpoints

## Step 1: Clone the Repository

```bash
git clone https://github.com/Pomilon-Intelligence-Lab/CRSM.git
cd CRSM
```

## Step 2: Create Virtual Environment

### Using venv (recommended)

```bash
# Create environment
python -m venv venv

# Activate environment
source venv/bin/activate          # Linux/macOS
# or
venv\Scripts\activate             # Windows
```

### Using conda

```bash
conda create -n crsm python=3.11
conda activate crsm
```

## Step 3: Install Dependencies

### Option A: Full Installation (Recommended)

```bash
pip install -e ".[dev]"
```

This installs:
- Core dependencies (PyTorch, transformers, etc.)
- Development tools (pytest, black, etc.)

### Option B: Core Only

```bash
pip install -e .
```

### Option C: From requirements.txt

```bash
pip install -r requirements.txt
```

### Option D: GPU Support (if not auto-detected)

```bash
# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Step 4: Verify Installation

```bash
# Check PyTorch
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Run tests
pytest tests/ -v

# Or quick import test
python -c "from crsm.core import CRSMConfig, CRSMModel; print('CRSM imports OK')"
```

## Advanced Configurations

### GPU Setup

#### NVIDIA GPU (CUDA)

1. Install NVIDIA driver:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install nvidia-driver-XXX  # Replace XXX with your GPU driver version

# Verify installation
nvidia-smi
```

2. Verify CUDA is available:
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

#### Apple Silicon (Metal)

```bash
# Install PyTorch with Metal acceleration
pip install torch torchvision torchaudio

# Verify Metal
python -c "import torch; print(torch.backends.mps.is_available())"
```

### Multi-GPU Setup (Optional)

For distributed training on multiple GPUs:

```bash
# Install accelerate
pip install accelerate

# Configure accelerate
accelerate config

# Run training
accelerate launch crsm/train.py --distributed
```

### Docker (Optional)

Create a Docker image for consistent environments:

```dockerfile
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y python3.11 python3.11-venv

WORKDIR /workspace
COPY . .

RUN python3.11 -m venv venv && \
    . venv/bin/activate && \
    pip install -e ".[dev]"

ENTRYPOINT ["/workspace/venv/bin/python"]
```

Build and run:
```bash
docker build -t crsm:latest .
docker run --gpus all -v $(pwd):/workspace crsm:latest -m pytest tests/
```

## Development Setup

### Code Style and Linting

Install development dependencies:
```bash
pip install black isort flake8 pylint
```

Format code:
```bash
black crsm/ tests/
isort crsm/ tests/
```

Check style:
```bash
flake8 crsm/ tests/
```

### Pre-commit Hooks (Optional)

```bash
pip install pre-commit

# Create .pre-commit-config.yaml and install hooks
pre-commit install

# Run hooks on all files
pre-commit run --all-files
```

## Troubleshooting

### ImportError: No module named 'torch'

```bash
# Ensure venv is activated
source venv/bin/activate  # Linux/macOS

# Reinstall PyTorch
pip install --upgrade torch
```

### CUDA not available

```bash
# Check if NVIDIA driver is installed
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Out of Memory during installation

```bash
# Use minimal dependencies first
pip install --no-deps torch transformers datasets

# Then install with --no-cache-dir
pip install --no-cache-dir -e .
```

### Permission denied on Linux

```bash
# Use --user flag
pip install --user -e .

# Or fix venv permissions
chmod -R u+w venv/
```

## Quick Start After Installation

### 1. Test the Installation

```bash
python -c "from crsm.core import CRSMConfig, CRSMModel; print('✓ CRSM ready')"
```

### 2. Run a Simple Example

```python
import torch
from crsm.core import CRSMConfig, CRSMModel

# Create model
config = CRSMConfig()
model = CRSMModel(config)

# Test forward pass
input_ids = torch.randint(0, config.vocab_size, (2, 512))
with torch.no_grad():
    logits, states = model(input_ids)
    
print(f"Input shape: {input_ids.shape}")
print(f"Logits shape: {logits.shape}")
print(f"✓ Forward pass works!")
```

### 3. Run Tests

```bash
pytest tests/ -v --tb=short
```

### 4. Try Colab Training

Open `notebooks/cloud_training/1_train_backbone.ipynb` in Google Colab and run the cells.

## Environment Variables

Optional environment variables for advanced users:

```bash
# GPU memory growth (prevent OOM crashes)
export CUDA_LAUNCH_BLOCKING=0

# Number of threads for data loading
export OMP_NUM_THREADS=8

# Debug CUDA errors
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# Disable cuDNN for reproducibility
export CUDNN_DETERMINISTIC=1

# Wandb project name
export WANDB_PROJECT="crsm-training"
```

## Next Steps

1. **Read the README** for project overview
2. **Check examples/** for usage patterns
3. **Review docs/ARCHITECTURE.md** for technical details
4. **Try the Colab notebook** for end-to-end training
5. **Run tests** to verify your setup

## Getting Help

- Open an issue on GitHub
- Check existing documentation in `docs/`
- Review example notebooks in `notebooks/`
- Check test files in `tests/` for usage patterns
