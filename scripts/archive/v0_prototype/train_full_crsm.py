"""
Complete CRSM training pipeline:
1. Train base MambaModel
2. Distill dynamics from backbone
3. Create CRSM with learned dynamics
4. Fine-tune with value head

Usage:
    python scripts/train_full_crsm.py --config configs/baseline_gpt2.json
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

def get_config_value(config, key, default=None):
    """Get config value from root or system section."""
    # Try root level first
    if key in config:
        return config[key]
    # Try system section
    if 'system' in config and key in config['system']:
        return config['system'][key]
    return default

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
        '--no-value-loss',
    ]
    
    # FIX 1: Add data directory
    data_config = config.get('data', {})
    if 'data_dir' in data_config:
        cmd.extend(['--data-dir', data_config['data_dir']])
    else:
        print("⚠ Warning: No data_dir in config, training may fail!")
    
    tokenizer = get_config_value(config, 'tokenizer')
    if tokenizer:
        cmd.extend(['--tokenizer', tokenizer])

    device = get_config_value(config, 'device')
    if device:
        cmd.extend(['--device', device])
    
    # FIX 3: Add gradient accumulation and other training params
    if 'grad_accum' in config.get('training', {}):
        cmd.extend(['--grad-accum', str(config['training']['grad_accum'])])
    
    if config.get('training', {}).get('use_amp', False):
        cmd.append('--amp')
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print("✗ Backbone training failed")
        return False

    # Find latest epoch checkpoint and copy to crsm_final.pt
    backbone_dir = output_dir / 'backbone'
    checkpoints = sorted(
        backbone_dir.glob('crsm_epoch*.pt'), 
        key=lambda p: int(p.stem.split('epoch')[1]), 
        reverse=True
    )
    
    if checkpoints:
        latest = checkpoints[0]
        final_path = backbone_dir / 'crsm_final.pt'
        shutil.copyfile(latest, final_path)
        print(f"✓ Backbone training complete — copied {latest.name} -> {final_path.name}")
    else:
        print("⚠ Warning: No epoch checkpoints found!")
        return False
    
    print("✓ Backbone training complete")
    return True


def distill_dynamics(config, output_dir):
    """Step 2: Distill dynamics from backbone and train LatentDynamics model."""
    print("\n" + "="*60)
    print("STEP 2: Distilling Dynamics and Training Model")
    print("="*60)
    
    backbone_path = output_dir / 'backbone' / 'crsm_final.pt'
    
    if not backbone_path.exists():
        print(f"✗ Error: Backbone checkpoint not found at {backbone_path}")
        return False
    
    dynamics_path = output_dir / 'dynamics.pt'
    
    # Get traces path
    traces_path = config.get('data', {}).get('traces_path', 'data/train_traces.jsonl')
    
    if not Path(traces_path).exists():
        print(f"⚠ Warning: Traces file not found at {traces_path}")
        print("  Creating bootstrap traces from training data...")
        # You could add bootstrap logic here
    
    cmd = [
        sys.executable, 'scripts/training/distill_dynamics.py',
        '--model-path', str(backbone_path),
        '--output-path', str(dynamics_path),
        '--traces-path', traces_path,
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


def create_crsm_with_dynamics(config, output_dir) -> Optional[Path]:
    """Step 3: Create CRSM with dynamics and save in CLI-compatible format."""
    print("\n" + "="*60)
    print("STEP 3: Creating CRSM with Dynamics")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create CRSM model instance
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

    # Load backbone weights
    backbone_dir = output_dir / 'backbone'
    latest_ckpt = backbone_dir / 'crsm_final.pt'
    
    if not latest_ckpt.exists():
        print(f"✗ Error: Backbone checkpoint not found at {latest_ckpt}")
        return None
    
    print(f"Loading backbone from {latest_ckpt}...")
    backbone_state = torch.load(latest_ckpt, map_location=device)
    
    # Extract state dict
    if 'model_state_dict' in backbone_state:
        state_dict = backbone_state['model_state_dict']
    elif 'model_state' in backbone_state:
        state_dict = backbone_state['model_state']
    else:
        state_dict = backbone_state
    
    crsm.backbone.load_state_dict(state_dict, strict=False)
    print("✓ Loaded backbone weights")

    # Load dynamics
    dynamics_path = output_dir / 'dynamics.pt'
    
    if not dynamics_path.exists():
        print(f"⚠ Warning: Dynamics not found at {dynamics_path}, skipping")
    else:
        success = load_dynamics_into_crsm(crsm, dynamics_path, device)
        if not success:
            print("⚠ Warning: Could not load dynamics, continuing anyway")
        else:
            # Test dynamics quality
            try:
                stats = check_dynamics_quality(crsm, test_samples=100)
                print(f"  Dynamics MSE: {stats['avg_mse']:.6f}")
            except Exception as e:
                print(f"⚠ Warning: Could not check dynamics quality: {e}")
    
    # Save CRSM with proper format
    final_crsm_path = output_dir / 'crsm_with_dynamics_resume.pt'
    
    try:
        checkpoint = {
            'epoch': 0, 
            'model_state_dict': crsm.state_dict(),
            'optimizer_state_dict': None,
        }
        
        torch.save(checkpoint, final_crsm_path)
        print(f"✓ Saved CRSM model to {final_crsm_path}")
        return final_crsm_path
        
    except Exception as e:
        print(f"✗ Error saving CRSM model: {e}")
        return None


def finetune_with_value(config, output_dir):
    """Step 4: Fine-tune the full CRSM model, including the Value Head."""
    print("\n" + "="*60)
    print("STEP 4: Fine-tuning with Value Head")
    print("="*60)
    
    crsm_with_dynamics_path = output_dir / 'crsm_with_dynamics_resume.pt'
    
    if not crsm_with_dynamics_path.exists():
        print(f"✗ Error: CRSM checkpoint not found at {crsm_with_dynamics_path}")
        return False

    finetune_epochs = config['training']['finetune_epochs']
    finetune_lr = config['training']['finetune_lr']

    cmd = [
        sys.executable, '-m', 'crsm.cli', 'train',
        '--epochs', str(finetune_epochs),
        '--batch-size', str(config['training']['batch_size']),
        '--vocab-size', str(config['model']['vocab_size']),
        '--seq-len', str(config['training']['seq_len']),
        '--lr', str(finetune_lr),
        '--resume', str(crsm_with_dynamics_path),
        '--checkpoint-dir', str(output_dir / 'final'),
    ]
    
    # FIX: Add same flags as Stage 1
    data_config = config.get('data', {})
    if 'data_dir' in data_config:
        cmd.extend(['--data-dir', data_config['data_dir']])
    
    if 'tokenizer' in config:
        cmd.extend(['--tokenizer', config['tokenizer']])
    
    if config.get('device'):
        cmd.extend(['--device', config['device']])
    
    if 'grad_accum' in config.get('training', {}):
        cmd.extend(['--grad-accum', str(config['training']['grad_accum'])])
    
    if config.get('training', {}).get('use_amp', False):
        cmd.append('--amp')

    print(f"Running: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print("✓ Fine-tuning complete")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Subprocess failed with error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, 
        formatter_class=argparse.RawTextHelpFormatter
    )
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
    parser.add_argument('--traces-path', type=str, default=None,
                       help="Path to traces (overrides config)")
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = json.load(f)

    # Override traces path if provided
    if args.traces_path:
        if 'data' not in config:
            config['data'] = {}
        config['data']['traces_path'] = args.traces_path
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    set_seed(config.get('seed', 42))
    
    # Print summary
    print("\n" + "="*60)
    print("CRSM Full Training Pipeline")
    print("="*60)
    print(f"Config: {args.config}")
    print(f"Output: {output_dir}")
    print(f"Vocab size: {config['model']['vocab_size']}")
    print(f"Model size: {config['model']['d_model']}d x {config['model']['num_layers']} layers")
    print(f"Tokenizer: {config.get('tokenizer', 'not specified')}")
    print(f"Data dir: {config.get('data', {}).get('data_dir', 'not specified')}")
    
    # Validate config
    if 'data' not in config or 'data_dir' not in config['data']:
        print("\n⚠ WARNING: No data.data_dir in config!")
        print("  Training will likely fail. Add this to your config:")
        print('  "data": { "data_dir": "data/text_corpus" }')
        return 1
    
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
    print(f"  - CRSM: {output_dir / 'crsm_with_dynamics_resume.pt'}")
    print(f"  - Final: {output_dir / 'final'}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())