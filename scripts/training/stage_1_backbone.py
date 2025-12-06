"""
Stage 1: Train the Backbone (System 1)
--------------------------------------
This script trains the MambaModel on a next-token prediction task.
It serves as the "System 1" or intuitive part of the model.

Output: experiments/stage_1/backbone_final.pt
"""

import sys
import yaml
import subprocess
import argparse
from pathlib import Path
import shutil

# Ensure the package is in the path
sys.path.insert(0, '.')

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/training_config.yaml", help="Path to YAML config file")
    
    # Training Overrides
    parser.add_argument('--epochs', type=int, help="Override number of epochs")
    parser.add_argument('--batch-size', type=int, help="Override batch size")
    parser.add_argument('--lr', type=float, help="Override learning rate")
    parser.add_argument('--seq-len', type=int, help="Override sequence length")
    parser.add_argument('--grad-accum', type=int, help="Override gradient accumulation steps")
    
    # System & Data
    parser.add_argument('--seed', type=int, help="Random seed")
    parser.add_argument('--data-dir', type=str, help="Path to data directory")
    parser.add_argument('--tokenizer', type=str, help="Tokenizer name")
    parser.add_argument('--num-workers', type=int, default=0, help="DataLoader workers")
    parser.add_argument('--device', type=str, help="Device (cpu/cuda)")
    
    # Distributed / Resume
    parser.add_argument('--distributed', action='store_true', help="Enable distributed training")
    parser.add_argument('--resume', type=str, help="Path to checkpoint to resume from")
    
    # Logging
    parser.add_argument('--no-wandb', action='store_true', help="Disable wandb")
    parser.add_argument('--wandb-project', type=str, help="WandB project name")

    args = parser.parse_args()

    config = load_config(args.config)
    
    # Setup Output Directory
    output_dir = Path("experiments/stage_1")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("STAGE 1: Backbone Training (System 1)")
    print("="*60)
    
    # Construct Command
    # Start with config values, override if CLI arg is present
    epochs = args.epochs if args.epochs is not None else config['training']['backbone_epochs']
    batch_size = args.batch_size if args.batch_size is not None else config['training']['batch_size']
    seq_len = args.seq_len if args.seq_len is not None else config['training']['seq_len']
    lr = args.lr if args.lr is not None else config['training']['lr']
    grad_accum = args.grad_accum if args.grad_accum is not None else config['training'].get('grad_accum', 1)
    
    cmd = [
        sys.executable, '-m', 'crsm.cli', 'train',
        '--epochs', str(epochs),
        '--batch-size', str(batch_size),
        '--vocab-size', str(config['model']['vocab_size']),
        '--seq-len', str(seq_len),
        '--lr', str(lr),
        '--checkpoint-dir', str(output_dir),
        '--no-value-loss',
        '--d-model', str(config['model']['d_model']),
        '--d-state', str(config['model']['d_state']),
        '--d-ffn', str(config['model']['d_ffn']),
        '--num-layers', str(config['model']['num_layers']),
        '--grad-accum', str(grad_accum),
        '--num-workers', str(args.num_workers)
    ]
    
    # Optional Args
    data_dir = args.data_dir if args.data_dir else config['data'].get('data_dir')
    if data_dir:
        cmd.extend(['--data-dir', data_dir])
    
    if config['training'].get('use_amp', False):
        cmd.append('--amp')
        
    system_config = config.get('system', {})
    
    if args.device:
        cmd.extend(['--device', args.device])
    elif system_config.get('device'):
        cmd.extend(['--device', system_config['device']])

    if args.seed is not None:
        cmd.extend(['--seed', str(args.seed)])
    elif system_config.get('seed'):
        cmd.extend(['--seed', str(system_config['seed'])])

    if args.no_wandb:
        cmd.append('--no-wandb')
    elif args.wandb_project:
        cmd.extend(['--wandb-project', args.wandb_project])
    
    if args.tokenizer:
        cmd.extend(['--tokenizer', args.tokenizer])
    elif config.get('tokenizer'):
        cmd.extend(['--tokenizer', config['tokenizer']])
        
    if args.resume:
        cmd.extend(['--resume', args.resume])
        
    if args.distributed:
        cmd.append('--distributed')

    print(f"Executing: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"✗ Training failed with error: {e}")
        sys.exit(1)
        
    # Standardize Output Name
    checkpoints = sorted(
        output_dir.glob('crsm_epoch*.pt'), 
        key=lambda p: int(p.stem.split('epoch')[1]) if 'epoch' in p.stem else 0, 
        reverse=True
    )
    
    if checkpoints:
        latest = checkpoints[0]
        final_path = output_dir / 'backbone_final.pt'
        shutil.copyfile(latest, final_path)
        print(f"✓ Saved final backbone to {final_path}")
    else:
        print("✗ No checkpoints found!")
        sys.exit(1)

if __name__ == "__main__":
    main()
