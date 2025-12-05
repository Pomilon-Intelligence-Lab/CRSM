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
    args = parser.parse_args()

    config = load_config(args.config)
    
    # Setup Output Directory
    output_dir = Path("experiments/stage_1")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("STAGE 1: Backbone Training (System 1)")
    print("="*60)
    
    # Construct Command
    cmd = [
        sys.executable, '-m', 'crsm.cli', 'train',
        '--epochs', str(config['training']['backbone_epochs']),
        '--batch-size', str(config['training']['batch_size']),
        '--vocab-size', str(config['model']['vocab_size']),
        '--seq-len', str(config['training']['seq_len']),
        '--lr', str(config['training']['lr']),
        '--checkpoint-dir', str(output_dir),
        '--no-value-loss',
        '--d-model', str(config['model']['d_model']),
        '--d-state', str(config['model']['d_state']),
        '--d-ffn', str(config['model']['d_ffn']),
        '--num-layers', str(config['model']['num_layers']),
    ]
    
    # Optional Args
    if 'data_dir' in config['data']:
        cmd.extend(['--data-dir', config['data']['data_dir']])
    
    if config['training'].get('use_amp', False):
        cmd.append('--amp')
        
    if 'grad_accum' in config['training']:
        cmd.extend(['--grad-accum', str(config['training']['grad_accum'])])
        
    if 'device' in config['system']:
        cmd.extend(['--device', config['system']['device']])

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
