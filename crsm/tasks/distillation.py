import torch
import torch.nn as nn
import json
from pathlib import Path
from torch.utils.data import IterableDataset, DataLoader
from .base import BaseTask

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

class DistillationTask(BaseTask):
    """Dynamics Distillation task: Train Dynamics model to predict state residuals."""
    
    def __init__(self, shards_dir):
        self.shards_dir = shards_dir
        self.criterion = nn.MSELoss()

    def get_dataloaders(self, batch_size):
        ds = ShardedDynamicsDataset(self.shards_dir)
        loader = DataLoader(ds, batch_size=batch_size, collate_fn=collate_dynamics_batch)
        return loader, None

    def compute_loss(self, model, batch, device):
        # In distillation, 'model' is actually the dynamics model
        state_t, action_emb, target_delta = batch
        state_t = state_t.to(device)
        action_emb = action_emb.to(device)
        target_delta = target_delta.to(device)
        
        # Dynamics model expects (states, action_emb)
        # Note: current DynamicsModel.forward expects states as List[Tensor] or Tensor
        predicted_delta = model(state_t, action_emb)
        
        return self.criterion(predicted_delta, target_delta)

    def evaluate(self, model, device):
        return {"mse": 0.0}
