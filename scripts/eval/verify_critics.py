import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
from yaml import safe_load

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from crsm.core.crsm import CRSMModel, CRSMConfig

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    ckpt_path = "checkpoints/rl_checkpoint_10.pt"
    config_path = "configs/arc_small.yaml"
    
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint {ckpt_path} not found.")
        return

    print(f"Loading checkpoint: {ckpt_path}")
    
    with open(config_path, "r") as f:
        config_dict = safe_load(f)
    
    # Filter keys for CRSMConfig
    valid_keys = {
        'vocab_size', 'hidden_size', 'intermediate_size', 'num_hidden_layers', 
        'num_attention_heads', 'max_position_embeddings', 'd_state', 'dropout', 
        'c_puct', 'n_simulations', 'autonomous_mode', 'temperature', 'top_k', 
        'top_p', 'delta_decay', 'max_lag', 'delta_scale', 'injection_rate'
    }
    filtered_config = {k: v for k, v in config_dict.items() if k in valid_keys}
    
    config = CRSMConfig.from_dict(filtered_config)
    model = CRSMModel(config)
    
    try:
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
                
        # Handle prefix mismatch
        first_key = next(iter(new_state_dict.keys()))
        if not first_key.startswith("crsm.") and hasattr(model, 'crsm'):
            # If checkpoint is inner model but we have wrapper
            print("Prefix mismatch detected. Adjusting keys...")
            # This is tricky. Usually we just load strictly=False or check keys.
            # If checkpoint has 'backbone.', it's likely 'crsm.backbone.'
            # Let's try to load to model.crsm if keys don't start with crsm
            if first_key.startswith("backbone."):
                 msg = model.crsm.load_state_dict(new_state_dict, strict=False)
            else:
                 msg = model.load_state_dict(new_state_dict, strict=False)
        else:
            msg = model.load_state_dict(new_state_dict, strict=False)
            
        print(f"Load result: {msg}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    model.to(device)
    model.eval()
    
    print("\n--- Running Value Head Verification ---")
    
    # Create random sequence
    curr_input = torch.randint(0, 10, (1, 10)).to(device)
    states = model.crsm.backbone.init_state(batch_size=1, device=device)
    
    # Prefill
    with torch.no_grad():
        logits, states = model.crsm.backbone(curr_input, states)
    
    value_trace = []
    
    print("Tracing values over 20 steps (Random Walk)...")
    with torch.no_grad():
        for i in range(20):
            # 1. Get values for current state
            layer_values = model.crsm.backbone._compute_layer_values(states)
            v_list = [v.item() for v in layer_values]
            value_trace.append(v_list)
            
            # 2. Step
            next_token = torch.randint(0, 10, (1, 1)).to(device)
            logits, states = model.crsm.backbone.step(next_token, states)

    value_trace = np.array(value_trace) # (Steps, Layers)
    
    print(f"\nValue Trace Shape: {value_trace.shape}")
    print("\nMean Value per Layer:")
    print(value_trace.mean(axis=0))
    
    print("\nStd Dev per Layer:")
    print(value_trace.std(axis=0))
    
    total_variance = value_trace.std(axis=0).mean()
    print(f"\nAverage Per-Layer Standard Deviation: {total_variance:.6f}")
    
    if total_variance < 1e-4:
        print("\n[CONCLUSION] CRITICS ARE FLAT. (Variance near zero)")
        print("This confirms Hypothesis C: The MCTS has no useful gradient to climb.")
    else:
        print("\n[CONCLUSION] Critics show variance.")
        print("Hypothesis C might be incorrect, or variance is not correlated with correctness.")

if __name__ == "__main__":
    main()
