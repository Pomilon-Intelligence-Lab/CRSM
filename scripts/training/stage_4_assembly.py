"""
Stage 4: Assembly (Integration & Verification)
----------------------------------------------
This script combines the trained Backbone (with Value Head) and the Distilled Dynamics Model
into the final unified CRSM artifact. It also runs a sanity check to ensure the assembled
model functions correctly.

Inputs:
    - experiments/stage_3/backbone_with_value.pt
    - experiments/stage_2/dynamics_final.pt

Output:
    - experiments/stage_4/crsm_final.pt
"""

import sys
import yaml
import torch
import argparse
from pathlib import Path
import shutil

sys.path.insert(0, '.')

from crsm.model import CRSM, CRSMConfig

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/training_config.yaml", help="Path to YAML config file")
    
    # Overrides
    parser.add_argument('--seed', type=int, help="Random seed")
    parser.add_argument('--device', type=str, help="Device (cpu/cuda)")
    
    args = parser.parse_args()

    config_dict = load_config(args.config)
    
    # Ensure system config exists
    if 'system' not in config_dict:
        config_dict['system'] = {'seed': 42, 'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    
    # Apply Overrides
    if args.seed: config_dict['system']['seed'] = args.seed
    if args.device: config_dict['system']['device'] = args.device

    device = config_dict['system']['device'] if torch.cuda.is_available() else 'cpu'
    
    # Paths
    backbone_path = Path("experiments/stage_3/backbone_with_value.pt")
    dynamics_path = Path("experiments/stage_2/dynamics_final.pt")
    output_dir = Path("experiments/stage_4")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("STAGE 4: Model Assembly & Verification")
    print("="*60)
    
    # Check inputs
    if not backbone_path.exists():
        print(f"✗ Missing Backbone checkpoint: {backbone_path}")
        print("  Please run Stage 3 first.")
        sys.exit(1)
        
    if not dynamics_path.exists():
        print(f"✗ Missing Dynamics checkpoint: {dynamics_path}")
        print("  Please run Stage 2 first.")
        sys.exit(1)
        
    print(f"✓ Found Backbone: {backbone_path}")
    print(f"✓ Found Dynamics: {dynamics_path}")
    
    # Initialize CRSM
    print("\nInitializing CRSM Model structure...")
    model = CRSM(
        vocab_size=config_dict['model']['vocab_size'],
        d_model=config_dict['model']['d_model'],
        d_state=config_dict['model']['d_state'],
        d_ffn=config_dict['model']['d_ffn'],
        num_layers=config_dict['model']['num_layers'],
        c_puct=config_dict['reasoning']['c_puct'],
        n_simulations=config_dict['reasoning']['n_simulations'],
        injection_rate=config_dict['reasoning']['injection_rate']
    ).to(device)
    
    # Load Backbone
    print("Loading Backbone weights...")
    ckpt = torch.load(backbone_path, map_location=device)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    
    missing, unexpected = model.backbone.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  ⚠ Missing keys in backbone: {len(missing)} (e.g., {missing[:3]}...)")
    if unexpected:
        print(f"  ⚠ Unexpected keys in backbone: {len(unexpected)}")
    
    # Load Dynamics
    print("Loading Dynamics weights...")
    success = model.load_dynamics(str(dynamics_path))
    if not success:
        print("✗ Failed to load dynamics!")
        sys.exit(1)
        
    # Verification
    print("\nVerifying Assembly...")
    model.eval()
    
    # 1. Test Forward Pass
    try:
        dummy_input = torch.randint(0, config_dict['model']['vocab_size'], (1, 10)).to(device)
        with torch.no_grad():
            logits, _ = model(dummy_input)
        print(f"✓ Forward pass successful. Logits shape: {logits.shape}")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        sys.exit(1)
        
    # 2. Test Dynamics Connection
    try:
        # Basic check if dynamics is reachable
        dummy_state = torch.randn(1, config_dict['model']['d_model']).to(device)
        with torch.no_grad():
            next_state_rep = model.dynamics(dummy_state)
        print(f"✓ Dynamics model active. Output shape: {next_state_rep.shape}")
    except Exception as e:
        print(f"✗ Dynamics check failed: {e}")
        sys.exit(1)
        
    # Save Assembled Model
    final_path = output_dir / 'crsm_final.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config_dict,
        'meta': {'stage': 4, 'status': 'assembled'}
    }, final_path)
    
    print(f"\n✓ Assembly Complete. Final model saved to: {final_path}")
    print(f"  Ready for inference!")

if __name__ == "__main__":
    main()
