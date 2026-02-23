# crsm/latent_train.py: Trains the Latent Dynamics Model

import os
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader

# CRITICAL CHANGE: Import the Dynamics Model
from .latent_dynamics import LatentDynamics 
from .utils import set_seed

class ShardedDynamicsDataset(IterableDataset):
    """
    Dataset for Latent Dynamics training.
    Loads pre-distilled state transition data.
    """
    def __init__(self, shards_dir: str):
        self.shards = sorted(Path(shards_dir).glob('*.jsonl'))

    def __iter__(self):
        for s in self.shards:
            with s.open('r', encoding='utf-8') as f:
                for line in f:
                    obj = json.loads(line)
                    # Load the state vectors, action embeddings, and delta vectors
                    state_t = torch.tensor(obj['state_t'], dtype=torch.float)
                    action_emb = torch.tensor(obj['action_emb'], dtype=torch.float)
                    state_delta = torch.tensor(obj['state_delta'], dtype=torch.float)
                    yield state_t, action_emb, state_delta

def collate_dynamics_batch(batch):
    # Batch is a list of (state_t, action_emb, state_delta)
    states_t, action_embs, deltas = zip(*batch)
    
    states_t_p = torch.stack(states_t)
    action_embs_p = torch.stack(action_embs)
    deltas_p = torch.stack(deltas)
    
    return states_t_p, action_embs_p, deltas_p


def train(shards_dir, d_model, epochs, batch_size, lr, device, output_path):
    device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Starting Dynamics Training with d_model={d_model}...")
    
    # Dynamics model dimension: state (d_model) + action (d_model from embedding)
    dynamics_model = LatentDynamics(d_model=d_model, action_dim=d_model).to(device) 
    
    # Loss function is MSE for state delta prediction
    crit = nn.MSELoss()
    opt = optim.Adam(dynamics_model.parameters(), lr=lr)
    
    # Data loading
    dataset = ShardedDynamicsDataset(shards_dir)
    dl = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_dynamics_batch, num_workers=0)
    
    best_loss = float('inf')

    for epoch in range(1, epochs + 1):
        total = 0.0
        dynamics_model.train()
        nsteps = 0
        for state_t, action_emb, target_delta in dl:
            state_t = state_t.to(device)
            action_emb = action_emb.to(device)
            target_delta = target_delta.to(device)
            
            opt.zero_grad()
            
            predicted_delta = dynamics_model(state_t, action_emb)
            
            # MSE between predicted delta and target delta
            loss = crit(predicted_delta, target_delta)
            
            loss.backward()
            opt.step()
            total += loss.item()
            nsteps += 1
            
        avg = total / max(1, nsteps)
        print(f'Epoch {epoch} avg_loss={avg:.6f}')
        
        # Save best model
        if avg < best_loss:
            best_loss = avg
            # FIX: Save the state_dict under the key 'dynamics_state'
            torch.save({'dynamics_state': dynamics_model.state_dict()}, output_path)
            print(f'  -> Model saved to {output_path}')
            
    # Return best loss for metadata saving in distill_dynamics.py
    return best_loss


def main():
    parser = argparse.ArgumentParser(description="Train Latent Dynamics Model")
    parser.add_argument('--shards-dir', required=True, help="Directory containing pre-distilled dynamics JSONL shards.")
    parser.add_argument('--d-model', type=int, default=128, help="Dimensionality of the latent state (d_model).")
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--output-path', type=str, default='dynamics.pt', help="Path to save the trained dynamics model.")
    args = parser.parse_args()
    
    # We ignore the returned loss here as it's typically called from the pipeline.
    train(
        args.shards_dir, 
        d_model=args.d_model, 
        epochs=args.epochs, 
        batch_size=args.batch_size, 
        lr=args.lr, 
        device=args.device,
        output_path=args.output_path
    )


if __name__ == '__main__':
    main()