"""
Distill dynamics model from the backbone SSM.

This script trains the LatentDynamics module (f_θ) to predict state transitions
by imitating the backbone's SSM forward passes:

    f_θ(h_t, a) ≈ h_{t+1} - h_t

where h_t is the per-layer state and a is an action (token).

Usage:
    python scripts/distill_dynamics.py --model-path checkpoints/base_model.pt \
                                       --output-path checkpoints/dynamics.pt \
                                       --num-samples 10000 \
                                       --epochs 5
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
from tqdm import tqdm

import sys
sys.path.insert(0, '.')

from crsm.mamba_ssm import MambaModel
from crsm.latent_dynamics import LatentDynamics
from crsm.utils import set_seed


def collect_transitions(backbone, num_samples, vocab_size, device):
    """
    Collect (state, action, next_state) tuples by running the backbone.
    
    Args:
        backbone: Trained MambaModel
        num_samples: Number of transitions to collect
        vocab_size: Vocabulary size
        device: Device to run on
        
    Returns:
        List of (state, action, next_state) tuples
    """
    print(f"Collecting {num_samples} state transitions...")
    backbone.eval()
    
    transitions = []
    batch_size = 32
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for _ in tqdm(range(num_batches), desc="Collecting"):
            # Initialize random states for this batch
            states = backbone.init_state(batch_size=batch_size, device=device)
            
            # Sample random actions
            actions = torch.randint(0, vocab_size, (batch_size,), device=device)
            
            # Get next states from backbone
            _, next_states = backbone.step(actions.unsqueeze(1), states)
            
            # Store transitions for each item in batch
            for b in range(batch_size):
                if len(transitions) >= num_samples:
                    break
                
                # Extract per-layer states for this batch item
                state_b = [s[b:b+1] if s is not None else None for s in states]
                next_state_b = [s[b:b+1] if s is not None else None for s in next_states]
                action_b = actions[b].item()
                
                transitions.append((state_b, action_b, next_state_b))
            
            if len(transitions) >= num_samples:
                break
    
    print(f"Collected {len(transitions)} transitions")
    return transitions[:num_samples]


def train_dynamics(dynamics, backbone, transitions, epochs, lr, device):
    """
    Train the dynamics model to predict state transitions.
    
    Args:
        dynamics: LatentDynamics module
        backbone: MambaModel (for action embeddings)
        transitions: List of (state, action, next_state) tuples
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
    """
    optimizer = optim.Adam(dynamics.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Split into train/val
    split_idx = int(0.9 * len(transitions))
    train_data = transitions[:split_idx]
    val_data = transitions[split_idx:]
    
    print(f"\nTraining dynamics model:")
    print(f"  Train samples: {len(train_data)}")
    print(f"  Val samples: {len(val_data)}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {lr}")
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        dynamics.train()
        train_loss = 0.0
        num_layers_trained = 0
        
        for state, action, next_state in tqdm(train_data, desc=f"Epoch {epoch+1}/{epochs}"):
            # Get action embedding
            action_tensor = torch.tensor([[action]], device=device)
            with torch.no_grad():
                action_emb = backbone.embedding(action_tensor).squeeze(1)  # (1, d_model)
            
            optimizer.zero_grad()
            layer_losses = []
            
            # Train on each layer
            for layer_idx, (s_curr, s_next) in enumerate(zip(state, next_state)):
                if s_curr is None or s_next is None:
                    continue
                
                s_curr = s_curr.to(device)
                s_next = s_next.to(device)
                
                # Predict delta
                pred_delta = dynamics(s_curr, action_emb)
                
                # Target delta
                target_delta = s_next - s_curr
                
                # Loss
                loss = criterion(pred_delta, target_delta)
                layer_losses.append(loss)
                num_layers_trained += 1
            
            if layer_losses:
                total_loss = sum(layer_losses) / len(layer_losses)
                total_loss.backward()
                optimizer.step()
                train_loss += total_loss.item()
        
        avg_train_loss = train_loss / len(train_data)
        
        # Validation
        dynamics.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for state, action, next_state in val_data:
                action_tensor = torch.tensor([[action]], device=device)
                action_emb = backbone.embedding(action_tensor).squeeze(1)
                
                layer_losses = []
                for s_curr, s_next in zip(state, next_state):
                    if s_curr is None or s_next is None:
                        continue
                    
                    s_curr = s_curr.to(device)
                    s_next = s_next.to(device)
                    
                    pred_delta = dynamics(s_curr, action_emb)
                    target_delta = s_next - s_curr
                    
                    loss = criterion(pred_delta, target_delta)
                    layer_losses.append(loss)
                
                if layer_losses:
                    val_loss += sum(layer_losses).item() / len(layer_losses)
        
        avg_val_loss = val_loss / len(val_data)
        
        print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.6f}, val_loss={avg_val_loss:.6f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"  → New best validation loss: {best_val_loss:.6f}")
    
    return best_val_loss


def main():
    parser = argparse.ArgumentParser(description="Distill dynamics model from backbone")
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained backbone checkpoint')
    parser.add_argument('--output-path', type=str, default='checkpoints/dynamics.pt',
                       help='Where to save trained dynamics model')
    parser.add_argument('--num-samples', type=int, default=10000,
                       help='Number of state transitions to collect')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--vocab-size', type=int, default=1000,
                       help='Vocabulary size')
    parser.add_argument('--d-model', type=int, default=128,
                       help='Model dimension')
    parser.add_argument('--num-layers', type=int, default=2,
                       help='Number of layers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Using device: {device}")
    
    # Load backbone
    print(f"\nLoading backbone from {args.model_path}...")
    backbone = MambaModel(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        d_state=64,
        d_ffn=512,
        num_layers=args.num_layers
    ).to(device)
    
    if Path(args.model_path).exists():
        checkpoint = torch.load(args.model_path, map_location=device)
        if 'model_state' in checkpoint:
            backbone.load_state_dict(checkpoint['model_state'])
        else:
            backbone.load_state_dict(checkpoint)
        print("✓ Loaded backbone weights")
    else:
        print("⚠ Checkpoint not found, using randomly initialized backbone")
    
    # Create dynamics model
    dynamics = LatentDynamics(d_model=args.d_model).to(device)
    print(f"\nDynamics model parameters: {sum(p.numel() for p in dynamics.parameters()):,}")
    
    # Collect transitions
    transitions = collect_transitions(
        backbone, 
        args.num_samples, 
        args.vocab_size, 
        device
    )
    
    # Train dynamics
    best_loss = train_dynamics(
        dynamics,
        backbone,
        transitions,
        args.epochs,
        args.lr,
        device
    )
    
    # Save dynamics model
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'dynamics_state': dynamics.state_dict(),
        'config': {
            'd_model': args.d_model,
            'vocab_size': args.vocab_size,
            'num_layers': args.num_layers,
        },
        'best_val_loss': best_loss,
        'num_samples': args.num_samples,
    }, output_path)
    
    print(f"\n✓ Saved dynamics model to {output_path}")
    print(f"  Best validation loss: {best_loss:.6f}")
    
    # Save metadata
    meta_path = output_path.parent / (output_path.stem + '_meta.json')
    with open(meta_path, 'w') as f:
        json.dump({
            'model_path': str(args.model_path),
            'output_path': str(args.output_path),
            'num_samples': args.num_samples,
            'epochs': args.epochs,
            'lr': args.lr,
            'best_val_loss': best_loss,
            'd_model': args.d_model,
            'vocab_size': args.vocab_size,
            'num_layers': args.num_layers,
        }, f, indent=2)
    
    print(f"✓ Saved metadata to {meta_path}")


if __name__ == '__main__':
    main()