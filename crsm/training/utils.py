import torch
import numpy as np
import random
from pathlib import Path
from ..core.dynamics import LatentDynamics

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_dynamics_into_crsm(crsm_model, dynamics_path, device=None):
    if device is None:
        device = next(crsm_model.parameters()).device
    
    dynamics_path = Path(dynamics_path)
    if not dynamics_path.exists():
        return False
    
    try:
        checkpoint = torch.load(dynamics_path, map_location=device)
        config = checkpoint.get('config', {})
        d_model = config.get('d_model', crsm_model.backbone.embedding.embedding_dim)
        num_layers = config.get('num_layers', len(crsm_model.backbone.layers))
        
        if not hasattr(crsm_model, 'dynamics') or crsm_model.dynamics is None:
            crsm_model.dynamics = LatentDynamics(d_model=d_model, num_layers=num_layers).to(device)
        
        crsm_model.dynamics.load_state_dict(checkpoint['dynamics_state'])
        return True
    except Exception:
        return False

def check_dynamics_quality(crsm_model, test_samples=100):
    if not hasattr(crsm_model, 'dynamics') or crsm_model.dynamics is None:
        return None
    
    device = next(crsm_model.parameters()).device
    vocab_size = crsm_model.backbone.embedding.num_embeddings
    crsm_model.eval()
    
    total_mse = 0.0
    total_mae = 0.0
    num_comparisons = 0
    
    with torch.no_grad():
        for _ in range(test_samples):
            states = crsm_model.backbone.init_state(batch_size=1, device=device)
            action = torch.randint(0, vocab_size, (1,), device=device)
            _, actual_next_states = crsm_model.backbone.step(action.unsqueeze(1), states)
            action_emb = crsm_model.backbone.embedding(action.unsqueeze(1)).squeeze(1)
            
            # Predict deltas for all layers
            pred_deltas = crsm_model.dynamics(states, action_emb)
            
            for i, (s_curr, s_actual) in enumerate(zip(states, actual_next_states)):
                if s_curr is None or s_actual is None: continue
                pred_next = s_curr + pred_deltas[i]
                mse = torch.mean((pred_next - s_actual) ** 2).item()
                mae = torch.mean(torch.abs(pred_next - s_actual)).item()
                total_mse += mse
                total_mae += mae
                num_comparisons += 1
                
    return {
        'avg_mse': total_mse / max(1, num_comparisons),
        'avg_mae': total_mae / max(1, num_comparisons),
        'num_comparisons': num_comparisons
    }
