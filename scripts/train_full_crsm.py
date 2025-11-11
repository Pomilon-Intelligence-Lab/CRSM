"""
Complete CRSM training pipeline:
1. Train base MambaModel
2. Distill dynamics from backbone
3. Create CRSM with learned dynamics
4. Fine-tune with value head

Usage:
    python scripts/train_full_crsm.py --config configs/small.json
"""

import argparse
import json
import torch
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, '.')

from crsm.model import CRSM
from crsm.mamba_ssm import MambaModel
from crsm.load_dynamics import load_dynamics_into_crsm, check_dynamics_quality, save_crsm_with_dynamics
from crsm.utils import set_seed


def train_backbone(config, output_dir):
    """Step 1: Train base MambaModel"""
    print("\n" + "="*60)
    print("STEP 1: Training Base Backbone")
    print("="*60)
    
    cmd = [
        sys.executable, '-m', 'crsm.cli', 'train',
        '--epochs', str(config['backbone_epochs']),
        '--batch-size', str(config['batch_size']),
        '--vocab-size', str(config['vocab_size']),
        '--seq-len', str(config['seq_len']),
        '--lr', str(config['lr']),
        '--checkpoint-dir', str(output_dir / 'backbone'),
        '--no-value-loss',  # Don't train value head yet
    ]
    
    if config.get('device'):
        cmd.extend(['--device', config['device']])
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print("✗ Backbone training failed")
        return False
    
    print("✓ Backbone training complete")
    return True


def distill_dynamics(config, output_dir):
    """Step 2: Distill dynamics model"""
    print("\n" + "="*60)
    print("STEP 2: Distilling Dynamics Model")
    print("="*60)
    
    # Find latest backbone checkpoint
    backbone_dir = output_dir / 'backbone'
    checkpoints = list(backbone_dir.glob('crsm_epoch*.pt'))
    
    if not checkpoints:
        print("✗ No backbone checkpoint found")
        return False
    
    latest_ckpt = max(checkpoints, key=lambda p: int(p.stem.split('epoch')[1]))
    print(f"Using backbone checkpoint: {latest_ckpt}")
    
    cmd = [
        sys.executable, 'scripts/distill_dynamics.py',
        '--model-path', str(latest_ckpt),
        '--output-path', str(output_dir / 'dynamics.pt'),
        '--num-samples', str(config['dynamics_samples']),
        '--epochs', str(config['dynamics_epochs']),
        '--lr', str(config['dynamics_lr']),
        '--vocab-size', str(config['vocab_size']),
        '--d-model', str(config['d_model']),
        '--num-layers', str(config['num_layers']),
    ]
    
    if config.get('device'):
        cmd.extend(['--device', config['device']])
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print("✗ Dynamics distillation failed")
        return False
    
    print("✓ Dynamics distillation complete")
    return True


def create_crsm_with_dynamics(config, output_dir):
    """Step 3: Create CRSM and load dynamics"""
    print("\n" + "="*60)
    print("STEP 3: Creating CRSM with Dynamics")
    print("="*60)
    
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    
    # Create CRSM
    crsm = CRSM(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        d_state=config['d_state'],
        d_ffn=config['d_ffn'],
        num_layers=config['num_layers'],
        dropout=config.get('dropout', 0.1),
        c_puct=config.get('c_puct', 1.0),
        n_simulations=config.get('n_simulations', 50),
    ).to(device)
    
    # Load backbone weights
    backbone_dir = output_dir / 'backbone'
    checkpoints = list(backbone_dir.glob('crsm_epoch*.pt'))
    latest_ckpt = max(checkpoints, key=lambda p: int(p.stem.split('epoch')[1]))
    
    print(f"Loading backbone from {latest_ckpt}...")
    backbone_state = torch.load(latest_ckpt, map_location=device)
    crsm.backbone.load_state_dict(backbone_state['model_state'])
    print("✓ Loaded backbone weights")
    
    # Load dynamics
    dynamics_path = output_dir / 'dynamics.pt'
    success = load_dynamics_into_crsm(crsm, dynamics_path, device)
    
    if not success:
        return None
    
    # Test dynamics quality
    stats = check_dynamics_quality(crsm, test_samples=100)
    
    # Save CRSM with dynamics
    crsm_path = output_dir / 'crsm_with_dynamics.pt'
    save_crsm_with_dynamics(crsm, crsm_path)
    
    return crsm_path


def finetune_with_value(config, output_dir):
    """Step 4: Fine-tune with value head"""
    print("\n" + "="*60)
    print("STEP 4: Fine-tuning with Value Head")
    print("="*60)
    
    # For now, we'll just run training with value loss enabled
    # In production, you'd load the CRSM checkpoint and continue training
    
    cmd = [
        sys.executable, '-m', 'crsm.cli', 'train',
        '--epochs', str(config['finetune_epochs']),
        '--batch-size', str(config['batch_size']),
        '--vocab-size', str(config['vocab_size']),
        '--seq-len', str(config['seq_len']),
        '--lr', str(config['finetune_lr']),
        '--checkpoint-dir', str(output_dir / 'final'),
        # Value loss is enabled by default
    ]
    
    if config.get('device'):
        cmd.extend(['--device', config['device']])
    
    # Resume from backbone checkpoint
    backbone_dir = output_dir / 'backbone'
    checkpoints = list(backbone_dir.glob('crsm_epoch*.pt'))
    if checkpoints:
        latest_ckpt = max(checkpoints, key=lambda p: int(p.stem.split('epoch')[1]))
        cmd.extend(['--resume', str(latest_ckpt)])
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print("✗ Fine-tuning failed")
        return False
    
    print("✓ Fine-tuning complete")
    return True


def main():
    parser = argparse.ArgumentParser(description="Complete CRSM training pipeline")
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config JSON file')
    parser.add_argument('--output-dir', type=str, default='experiments/full_crsm',
                       help='Output directory for all artifacts')
    parser.add_argument('--skip-backbone', action='store_true',
                       help='Skip backbone training (use existing checkpoint)')
    parser.add_argument('--skip-dynamics', action='store_true',
                       help='Skip dynamics distillation')
    parser.add_argument('--skip-finetune', action='store_true',
                       help='Skip final fine-tuning')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = json.load(f)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    set_seed(config.get('seed', 42))
    
    print("\n" + "="*60)
    print("CRSM Full Training Pipeline")
    print("="*60)
    print(f"Config: {args.config}")
    print(f"Output: {output_dir}")
    print(f"Vocab size: {config['vocab_size']}")
    print(f"Model size: {config['d_model']}d x {config['num_layers']} layers")
    
    # Step 1: Train backbone
    if not args.skip_backbone:
        if not train_backbone(config, output_dir):
            print("\n✗ Pipeline failed at backbone training")
            return 1
    else:
        print("\nSkipping backbone training (--skip-backbone)")
    
    # Step 2: Distill dynamics
    if not args.skip_dynamics:
        if not distill_dynamics(config, output_dir):
            print("\n✗ Pipeline failed at dynamics distillation")
            return 1
    else:
        print("\nSkipping dynamics distillation (--skip-dynamics)")
    
    # Step 3: Create CRSM with dynamics
    crsm_path = create_crsm_with_dynamics(config, output_dir)
    if crsm_path is None:
        print("\n✗ Pipeline failed at CRSM creation")
        return 1
    
    # Step 4: Fine-tune with value head
    if not args.skip_finetune:
        if not finetune_with_value(config, output_dir):
            print("\n✗ Pipeline failed at fine-tuning")
            return 1
    else:
        print("\nSkipping fine-tuning (--skip-finetune)")
    
    print("\n" + "="*60)
    print("✓ PIPELINE COMPLETE")
    print("="*60)
    print(f"\nArtifacts saved to: {output_dir}")
    print(f"  - Backbone: {output_dir / 'backbone'}")
    print(f"  - Dynamics: {output_dir / 'dynamics.pt'}")
    print(f"  - CRSM: {output_dir / 'crsm_with_dynamics.pt'}")
    print(f"  - Final: {output_dir / 'final'}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())