import sys
import os
import torch
import asyncio
import numpy as np
from yaml import safe_load

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from crsm.core.crsm import CRSMModel, CRSMConfig
from crsm.tasks.arc_task import ARCTask

async def run_generation(model, task, injection_rate, device):
    # Set injection rate
    if hasattr(model, 'crsm'):
        model.crsm.injection_rate = injection_rate
    
    # Get a fresh loader to ensure same sample? No, loader shuffle=True by default.
    # We should reuse the same batch.
    
    return
    
async def run_test(model, task, device):
    # Get ONE batch to reuse
    loader, _ = task.get_dataloaders(batch_size=1)
    batch = next(iter(loader))
    x, y, split_indices = batch
    split_idx = split_indices[0].item()
    x_in = x[:, :split_idx].to(device)
    
    # Extract target
    raw_targets = y[0, split_idx-1:].tolist()
    target_tokens = []
    for t in raw_targets:
        target_tokens.append(t)
        if t == 14: break
        
    print(f"\nTest Input Shape: {x_in.shape}")
    
    rates = [0.0, 0.2, 1.0]
    results = {}
    
    for rate in rates:
        print(f"\nGenerating with injection_rate={rate}...")
        
        # Reset state logic inside model if needed?
        # think_and_generate does init_state inside.
        
        if hasattr(model, 'crsm'):
            model.crsm.injection_rate = rate
        
        # Force deterministic randomness for fair comparison?
        # MCTS is stochastic. So exact match is unlikely if temperature > 0.
        # But if rate=0.0, MCTS shouldn't affect state.
        # Wait, rate=0.0 means (1-0)*state + 0*target = state. So pure backbone.
        
        output_ids = await model.crsm.think_and_generate(
            x_in, 
            max_length=min(30, len(target_tokens) + 10), 
            use_deliberation=True, # Always run MCTS
            deliberation_lag=0, 
            fallback_to_sampling=False
        )
        
        pred_tokens = output_ids[split_idx:].tolist()
        results[rate] = pred_tokens
        print(f"Output: {pred_tokens}")

    # Comparison
    base = results[0.0]
    current = results[0.2]
    forced = results[1.0]
    
    print("\n--- Analysis ---")
    if base == current:
        print("Rate 0.0 vs 0.2: Identical output. (MCTS too weak or agreeing with backbone)")
    else:
        print("Rate 0.0 vs 0.2: DIFFERENT output. (MCTS has influence)")
        
    if base == forced:
        print("Rate 0.0 vs 1.0: Identical output. (CRITICAL FAILURE: MCTS cannot steer even when forced)")
    else:
        print("Rate 0.0 vs 1.0: DIFFERENT output. (MCTS mechanism works, plans might be bad)")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    ckpt_path = "checkpoints/rl_checkpoint_10.pt"
    config_path = "configs/arc_small.yaml"
    
    with open(config_path, "r") as f:
        config_dict = safe_load(f)
        
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
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'): new_state_dict[k[7:]] = v
            else: new_state_dict[k] = v
            
        first_key = next(iter(new_state_dict.keys()))
        if not first_key.startswith("crsm.") and hasattr(model, 'crsm'):
            if first_key.startswith("backbone."):
                 model.crsm.load_state_dict(new_state_dict, strict=False)
            else:
                 model.load_state_dict(new_state_dict, strict=False)
        else:
            model.load_state_dict(new_state_dict, strict=False)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
        
    model.to(device)
    model.eval()
    
    task = ARCTask() # Dummy data
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_test(model, task, device))

if __name__ == "__main__":
    main()
