import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
from tqdm import tqdm
from typing import List, Tuple, Any

import sys
sys.path.insert(0, '.')

from crsm.mamba_ssm import MambaModel
from crsm.latent_dynamics import LatentDynamics
from crsm.utils import set_seed
from crsm.tokenizer import Tokenizer
# We'll import the training function from the dedicated file
from crsm.latent_train import train as train_dynamics_model, ShardedDynamicsDataset, collate_dynamics_batch

# --- Distillation Logic (Collect Transitions) ---
def collect_transitions(backbone, traces_path, output_dir, num_samples, vocab_size, device) -> Path:
    """
    Collect (state_t, action_emb, state_delta) tuples by running the backbone on a real dataset.
    The collected data is saved to a temp directory and the path is returned.
    """
    if not Path(traces_path).exists():
        raise FileNotFoundError(f"Traces file not found at: {traces_path}. Cannot distill dynamics.")

    temp_shards_dir = Path(output_dir) / 'dynamics_temp_shards'
    temp_shards_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Collecting {num_samples} state transitions from {traces_path}...")
    backbone.eval()
    
    # FIX from previous step: Remove vocab_size as Tokenizer does not accept it.
    tokenizer = Tokenizer() 
    
    current_shard_size = 0
    max_shard_size = 1000  # Save to new file every 1000 transitions
    shard_idx = 0
    shard_data = []
    total_transitions = 0

    def save_shard():
        nonlocal shard_idx, shard_data
        if not shard_data: return
        shard_path = temp_shards_dir / f'transitions_shard_{shard_idx:05d}.jsonl'
        with shard_path.open('w', encoding='utf-8') as f:
            for entry in shard_data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        shard_idx += 1
        shard_data = []

    with Path(traces_path).open('r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Distilling Data"):
            if total_transitions >= num_samples:
                break
                
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            prompt = obj.get('prompt', '')
            trace = obj.get('trace', '')
            input_text = (prompt + '\n' + trace).strip()
            
            token_ids = tokenizer.encode(input_text)
            
            # Skip short sequences
            if len(token_ids) < 2: continue
            
            # Initialize state (list of per-layer tensors)
            # FIX from previous step: Explicitly set batch_size=1
            state = backbone.init_state(batch_size=1, device=device)
            
            # Process one token at a time to capture transitions
            for t in range(len(token_ids) - 1):
                if total_transitions >= num_samples:
                    break
                    
                # 1. State before the action (h_t)
                # FIX: Squeeze the batch dimension (which is 1) before converting to list.
                # This ensures a 1D list is saved, which becomes a 2D tensor [B, D] after batching, 
                # resolving the 'got 3 and 2' RuntimeError.
                state_t_list = [s.squeeze(0).clone().detach().cpu().numpy().tolist() for s in state if s is not None]

                # 2. Action (token_t) and its embedding (e_a)
                token_t = torch.tensor([[token_ids[t]]], dtype=torch.long, device=device)
                
                # Get action embedding from backbone's embedding layer
                action_emb_tensor = backbone.embedding(token_t).squeeze(0).squeeze(0).detach().cpu()
                
                # 3. Predict the next state (h_{t+1})
                with torch.no_grad():
                    _, next_state = backbone.step(token_t, state)
                
                # 4. Calculate the Delta (h_{t+1} - h_t)
                state_delta_list = []
                for layer_idx, s_curr in enumerate(state):
                    if s_curr is not None:
                         s_next = next_state[layer_idx]
                         # Must squeeze here too to match state_t dimension
                         delta = (s_next.clone().detach() - s_curr.clone().detach()).squeeze(0).cpu().numpy().tolist()
                         state_delta_list.append(delta)
                    
                if len(state_t_list) > 0:
                    # For a single-layer dynamics model, we just use the first layer's state
                    shard_data.append({
                        'state_t': state_t_list[0], 
                        'action_emb': action_emb_tensor.numpy().tolist(), 
                        'state_delta': state_delta_list[0]
                    })
                    total_transitions += 1
                    current_shard_size += 1
                    
                    if current_shard_size >= max_shard_size:
                        save_shard()
                        current_shard_size = 0
                
                # Update state for next iteration
                state = next_state
            
    # Save the last shard
    save_shard()
    
    print(f"\n✓ Collected {total_transitions} real transitions into {shard_idx} shards.")
    return temp_shards_dir


# --- Modified Main function (to use the new collector and separate trainer) ---

def main():
    parser = argparse.ArgumentParser(description="Distill and train Latent Dynamics Model")
    parser.add_argument('--model-path', required=True, help="Path to the trained Mamba backbone checkpoint.")
    parser.add_argument('--output-path', required=True, help="Path to save the trained dynamics model.")
    parser.add_argument('--traces-path', required=True, help="Path to the real data (JSONL traces) for distillation.")
    parser.add_argument('--num-samples', type=int, default=5000, help="Number of state transitions to collect.")
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    # Add optional overrides for model params not in checkpoint metadata
    parser.add_argument('--d-model', type=int, default=128)
    parser.add_argument('--vocab-size', type=int, default=1000)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()
    
    set_seed(42)
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    
    # --- Load Config and Model ---
    loaded_checkpoint = torch.load(args.model_path, map_location='cpu')
    if isinstance(loaded_checkpoint, dict) and 'model_state' in loaded_checkpoint:
        state_dict = loaded_checkpoint['model_state']
    else:
        state_dict = loaded_checkpoint
        
    model_params = {}
    # Extract d_model and vocab_size from state_dict for proper model sizing
    for k, v in state_dict.items():
        if k == 'embedding.weight':
             model_params['vocab_size'] = v.size(0)
             model_params['d_model'] = v.size(1)
             break

    d_model = model_params.get('d_model', args.d_model)
    vocab_size = model_params.get('vocab_size', args.vocab_size)
    num_layers = args.num_layers 
    
    backbone = MambaModel(
        vocab_size=vocab_size, 
        d_model=d_model, 
        d_state=args.d_model // 2, 
        d_ffn=d_model * 4, 
        num_layers=num_layers
    ).to(device)

    # FIX from previous step: Load the state dict directly into the MambaModel instance
    try:
        backbone.load_state_dict(state_dict, strict=True)
        print("Loaded MambaModel state dictionary successfully (strict=True).")
    except RuntimeError as e:
        print(f"Warning: Failed to load state dictionary with strict=True. Trying strict=False. Error: {e}")
        backbone.load_state_dict(state_dict, strict=False)

    print(f"Loaded backbone from {args.model_path} (d_model={d_model}, layers={num_layers}).")

    # --- Step 1: Collect Transitions (The Distillation) ---
    temp_shards_dir = collect_transitions(
        backbone, 
        args.traces_path,
        Path(args.output_path).parent,
        args.num_samples, 
        vocab_size, 
        device
    )
    
    # --- Step 2: Train Dynamics (Calling the other script's logic) ---
    print("\n" + "="*60)
    print("STEP 2: Training Dynamics Model (MSE Loss)")
    print("="*60)

    # Calling the training logic from crsm/latent_train.py
    best_loss = train_dynamics_model(
        shards_dir=str(temp_shards_dir),
        d_model=d_model, 
        epochs=args.epochs, 
        batch_size=args.batch_size, 
        lr=args.lr, 
        device=str(device),
        output_path=args.output_path
    )
    
    # Save metadata
    meta_path = Path(args.output_path).parent / (Path(args.output_path).stem + '_meta.json')
    with open(meta_path, 'w') as f:
        json.dump({
            'model_path': str(args.model_path),
            'output_path': str(args.output_path),
            'num_samples': args.num_samples,
            'epochs': args.epochs,
            'lr': args.lr,
            'd_model': d_model,
            'vocab_size': vocab_size,
            'num_layers': num_layers,
            'best_val_loss': best_loss
        }, f, indent=2)
    
    print(f"\n✓ Saved dynamics model to {args.output_path}")
    print(f"  Best validation loss: {best_loss:.6f}")


if __name__ == '__main__':
    main()