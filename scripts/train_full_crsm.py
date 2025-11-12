"""
Complete CRSM training pipeline:
1. Train base MambaModel
2. Distill dynamics from backbone
3. Create CRSM with learned dynamics
4. Fine-tune with value head

Usage:
    python scripts/train_full_crsm.py --config configs/small.json
"""

import shutil
import argparse
import json
import torch
import subprocess
import sys
from pathlib import Path
from typing import Optional

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
        '--epochs', str(config['training']['backbone_epochs']),
        '--batch-size', str(config['training']['batch_size']),
        '--vocab-size', str(config['model']['vocab_size']),
        '--seq-len', str(config['training']['seq_len']),
        '--lr', str(config['training']['lr']),
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

    # find latest epoch checkpoint and copy it to crsm_final.pt for later steps
    backbone_dir = output_dir / 'backbone'
    checkpoints = sorted(backbone_dir.glob('crsm_epoch*.pt'), key=lambda p: int(p.stem.split('epoch')[1]), reverse=True)
    if checkpoints:
        latest = checkpoints[0]
        final_path = backbone_dir / 'crsm_final.pt'
        shutil.copyfile(latest, final_path)
        print(f"✓ Backbone training complete — copied {latest.name} -> {final_path.name}")
    else:
        print("✓ Backbone training complete (no epoch checkpoints found)")
    
    print("✓ Backbone training complete")
    return True


def distill_dynamics(config, output_dir):
    """Step 2: Distill dynamics from backbone and train LatentDynamics model."""
    print("\n" + "="*60)
    print("STEP 2: Distilling Dynamics and Training Model")
    print("="*60)
    
    backbone_path = output_dir / 'backbone' / 'crsm_final.pt' # Assuming a final checkpoint
    dynamics_path = output_dir / 'dynamics.pt'
    
    # --- CRITICAL CHANGE: Add traces-path argument ---
    # NOTE: You must have a 'data' key in your config pointing to a real dataset path
    traces_path = config.get('data', {}).get('traces_path', 'data/train_traces.jsonl')
    
    cmd = [
        sys.executable, 'scripts/distill_dynamics.py',
        '--model-path', str(backbone_path),
        '--output-path', str(dynamics_path),
        '--traces-path', traces_path, # <-- NEW ARGUMENT
        '--num-samples', str(config['dynamics']['dynamics_samples']),
        '--epochs', str(config['dynamics']['dynamics_epochs']),
        '--lr', str(config['dynamics']['dynamics_lr']),
        '--d-model', str(config['model']['d_model']),
        '--vocab-size', str(config['model']['vocab_size']),
        '--num-layers', str(config['model']['num_layers']),
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


# FIX in scripts/train_full_crsm.py

# FINAL FIX in scripts/train_full_crsm.py (Replace existing function)

def create_crsm_with_dynamics(config, output_dir) -> Optional[Path]:
    """Step 3: Create CRSM with dynamics and save in CLI-compatible format."""
    print("\n" + "="*60)
    print("STEP 3: Creating CRSM with Dynamics")
    print("="*60)
    
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    
    # Create CRSM model instance using CORRECT config keys
    crsm = CRSM(
        vocab_size=config['model']['vocab_size'],
        d_model=config['model']['d_model'],
        d_state=config['model']['d_state'],
        d_ffn=config['model']['d_ffn'],
        num_layers=config['model']['num_layers'],
        dropout=config['model'].get('dropout', 0.1),
        c_puct=config['reasoning'].get('c_puct', 1.0),
        n_simulations=config['reasoning'].get('n_simulations', 10),
    ).to(device)

    # Load backbone weights (from crsm_final.pt created in Step 1)
    backbone_dir = output_dir / 'backbone'
    latest_ckpt = backbone_dir / 'crsm_final.pt' 
    print(f"Loading backbone from {latest_ckpt}...")
    backbone_state = torch.load(latest_ckpt, map_location=device)
    # NOTE: Assuming backbone save format is {'model_state': ...}
    crsm.backbone.load_state_dict(backbone_state['model_state_dict'])
    print("✓ Loaded backbone weights")

    # Load dynamics (from dynamics.pt created in Step 2)
    dynamics_path = output_dir / 'dynamics.pt'
    success = load_dynamics_into_crsm(crsm, dynamics_path, device)
    if not success:
        return None
    
    # Test dynamics quality (omitted for brevity, but should run)
    stats = check_dynamics_quality(crsm, test_samples=100)
    
    # --- CRITICAL FIX: Save using standard PyTorch checkpoint keys ---
    final_crsm_path = output_dir / 'crsm_with_dynamics_resume.pt'
    
    try:
        # Renamed keys to model_state_dict and optimizer_state_dict
        checkpoint = {
            'epoch': 0, 
            'model_state_dict': crsm.state_dict(),
            'optimizer_state_dict': None, # Placeholder for compatibility
            'dynamics_stats': stats,
        }
        
        torch.save(checkpoint, final_crsm_path)
        
        print(f"✓ Saved CRSM model (resume format) to {final_crsm_path}")
        return final_crsm_path
        
    except Exception as e:
        print(f"Error saving CRSM model: {e}")
        return None
        
def finetune_with_value(config, output_dir):
    """Step 4: Fine-tune the full CRSM model, including the Value Head."""
    print("\n" + "="*60)
    print("STEP 4: Fine-tuning with Value Head")
    print("="*60)
    
    # 1. Ensure we start fine-tuning from the CRSM with Dynamics
    crsm_with_dynamics_path = output_dir / 'crsm_with_dynamics_resume.pt'
    
    if not crsm_with_dynamics_path.exists():
        print(f"Error: CRSM with dynamics (resume format) not found at {crsm_with_dynamics_path}. Cannot fine-tune.")
        return False

    # 2. Use the parameters from config
    finetune_epochs = config['training']['finetune_epochs']
    finetune_lr = config['training']['finetune_lr']
    
    cmd = [
        sys.executable, '-m', 'crsm.cli', 'train',
        '--epochs', str(finetune_epochs),
        '--batch-size', str(config['training']['batch_size']),
        '--vocab-size', str(config['model']['vocab_size']),
        '--seq-len', str(config['training']['seq_len']),
        '--lr', str(finetune_lr),
        # CRITICAL FIX: Use the fully constructed CRSM as the resume point
        '--resume', str(crsm_with_dynamics_path),
        # Direct the output to the parent directory for easier file finding
        '--checkpoint-dir', str(output_dir), 
    ]
    
    if config.get('device'):
        cmd.extend(['--device', config['device']])
    
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        print("✓ Fine-tuning complete")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Subprocess failed with error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
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
    parser.add_argument('--traces-path', type=str, default=None, help="Path to the real data (JSONL traces) for distillation. Overrides config.") # <-- NEW ARGUMENT
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = json.load(f)

    if args.traces_path:
        config['data'] = config.get('data', {})
        config['data']['traces_path'] = args.traces_path
    
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
    print(f"Vocab size: {config['model']['vocab_size']}")
    print(f"Model size: {config['model']['d_model']}d x {config['model']['num_layers']} layers")
    
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