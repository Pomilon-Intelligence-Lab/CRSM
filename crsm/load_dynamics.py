"""
Utility functions for loading trained dynamics models into CRSM.
"""

import torch
from pathlib import Path
from .latent_dynamics import LatentDynamics


def load_dynamics_into_crsm(crsm_model, dynamics_path, device=None):
    """
    Load a trained dynamics model into a CRSM instance.
    
    Args:
        crsm_model: CRSM model instance
        dynamics_path: Path to saved dynamics checkpoint
        device: Device to load on (default: model's device)
        
    Returns:
        bool: True if successful, False otherwise
    """
    if device is None:
        device = next(crsm_model.parameters()).device
    
    dynamics_path = Path(dynamics_path)
    
    if not dynamics_path.exists():
        print(f"✗ Dynamics checkpoint not found: {dynamics_path}")
        return False
    
    try:
        checkpoint = torch.load(dynamics_path, map_location=device)
        
        # Get config
        config = checkpoint.get('config', {})
        d_model = config.get('d_model', crsm_model.backbone.embedding.embedding_dim)
        
        # Create dynamics module if not exists
        if not hasattr(crsm_model, 'dynamics'):
            crsm_model.dynamics = LatentDynamics(d_model=d_model).to(device)
        
        # Load weights
        crsm_model.dynamics.load_state_dict(checkpoint['dynamics_state'])
        
        print(f"✓ Loaded dynamics model from {dynamics_path}")
        print(f"  Validation loss: {checkpoint.get('best_val_loss', 'N/A')}")
        print(f"  Trained on {checkpoint.get('num_samples', 'N/A')} samples")
        
        return True
    
    except Exception as e:
        print(f"✗ Failed to load dynamics: {e}")
        return False


def save_crsm_with_dynamics(crsm_model, save_path):
    """
    Save CRSM model including dynamics module.
    
    Args:
        crsm_model: CRSM model instance
        save_path: Path to save to
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'model_state': crsm_model.state_dict(),
    }
    
    # Add dynamics if it exists
    if hasattr(crsm_model, 'dynamics'):
        checkpoint['has_dynamics'] = True
        checkpoint['dynamics_config'] = {
            'd_model': crsm_model.dynamics.net[0].in_features - crsm_model.dynamics.net[0].in_features // 2
        }
    else:
        checkpoint['has_dynamics'] = False
    
    torch.save(checkpoint, save_path)
    print(f"✓ Saved CRSM model to {save_path}")


def check_dynamics_quality(crsm_model, test_samples=100):
    """
    Test the quality of loaded dynamics by comparing predictions to actual SSM transitions.
    
    Args:
        crsm_model: CRSM model with loaded dynamics
        test_samples: Number of samples to test
        
    Returns:
        dict: Statistics about prediction quality
    """
    if not hasattr(crsm_model, 'dynamics'):
        print("✗ No dynamics model loaded")
        return None
    
    print(f"\nTesting dynamics quality on {test_samples} samples...")
    
    device = next(crsm_model.parameters()).device
    vocab_size = crsm_model.backbone.embedding.num_embeddings
    
    crsm_model.eval()
    
    total_mse = 0.0
    total_mae = 0.0
    num_comparisons = 0
    
    with torch.no_grad():
        for _ in range(test_samples):
            # Get random state and action
            states = crsm_model.backbone.init_state(batch_size=1, device=device)
            action = torch.randint(0, vocab_size, (1,), device=device)
            
            # Get actual next state from backbone
            _, actual_next_states = crsm_model.backbone.step(action.unsqueeze(1), states)
            
            # Get predicted next state from dynamics
            action_emb = crsm_model.backbone.embedding(action.unsqueeze(1)).squeeze(1)
            
            for layer_idx, (s_curr, s_actual) in enumerate(zip(states, actual_next_states)):
                if s_curr is None or s_actual is None:
                    continue
                
                # Predict delta
                pred_delta = crsm_model.dynamics(s_curr, action_emb)
                pred_next = s_curr + pred_delta
                
                # Compute errors
                mse = torch.mean((pred_next - s_actual) ** 2).item()
                mae = torch.mean(torch.abs(pred_next - s_actual)).item()
                
                total_mse += mse
                total_mae += mae
                num_comparisons += 1
    
    avg_mse = total_mse / num_comparisons
    avg_mae = total_mae / num_comparisons
    
    stats = {
        'avg_mse': avg_mse,
        'avg_mae': avg_mae,
        'num_comparisons': num_comparisons,
    }
    
    print(f"\nDynamics Quality:")
    print(f"  Average MSE: {avg_mse:.6f}")
    print(f"  Average MAE: {avg_mae:.6f}")
    print(f"  Comparisons: {num_comparisons}")
    
    # Quality assessment
    if avg_mse < 0.01:
        print("  ✓ Excellent quality")
    elif avg_mse < 0.1:
        print("  ✓ Good quality")
    elif avg_mse < 1.0:
        print("  ⚠ Acceptable quality")
    else:
        print("  ✗ Poor quality - consider retraining")
    
    return stats