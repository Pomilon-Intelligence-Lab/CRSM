import torch

class BaseTask:
    """Abstract interface for all CRSM tasks (LM, Distillation, ARC-AGI)."""
    
    def get_dataloaders(self, batch_size):
        """Returns (train_loader, val_loader)."""
        raise NotImplementedError
        
    def compute_loss(self, model, batch, device):
        """Computes and returns a scalar loss tensor for a batch."""
        raise NotImplementedError
        
    def evaluate(self, model, device):
        """Performs evaluation and returns a dictionary of metrics."""
        raise NotImplementedError
