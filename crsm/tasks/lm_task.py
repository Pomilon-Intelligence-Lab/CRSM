import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .base import BaseTask
from ..data.datasets import RandomTokenDataset, StreamingTextDataset

class LanguageModelingTask(BaseTask):
    """Standard Language Modeling task with Multi-Headed Value Critic support."""
    
    def __init__(self, vocab_size=1000, seq_len=32, data_dir=None, hf_tokenizer_name=None):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.data_dir = data_dir
        self.hf_tokenizer_name = hf_tokenizer_name
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def get_dataloaders(self, batch_size):
        if self.data_dir:
            ds = StreamingTextDataset(
                data_dir=self.data_dir, 
                seq_len=self.seq_len, 
                vocab_size=self.vocab_size,
                hf_tokenizer_name=self.hf_tokenizer_name
            )
            shuffle = False
        else:
            ds = RandomTokenDataset(vocab_size=self.vocab_size, seq_len=self.seq_len, size=2000)
            shuffle = True
            
        loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
        return loader, None # No val_loader for now in random/streaming

    def compute_loss(self, model, batch, device, use_value_loss=True):
        x, y = batch
        x, y = x.to(device), y.to(device)
        
        # Get predictions
        if use_value_loss and hasattr(model, 'predict_policy_value'):
            logits, value, _ = model.predict_policy_value(x, states=None)
        else:
            logits, _ = model(x, states=None)
            value = None
            
        b, t, v = logits.size()
        
        # 1. LM Loss
        lm_loss_per_token = self.criterion(logits.reshape(b * t, v), y.reshape(b * t))
        lm_loss = lm_loss_per_token.mean()
        
        if not use_value_loss or value is None:
            return lm_loss
            
        # 2. Value Loss (MV-Critic)
        with torch.no_grad():
            log_probs = F.log_softmax(logits, dim=-1)
            last_step_log_probs = log_probs[:, -1, :]
            last_step_targets = y[:, -1]
            token_log_probs = last_step_log_probs.gather(1, last_step_targets.unsqueeze(-1)).squeeze(-1)
            target_value = torch.exp(token_log_probs)

        if isinstance(value, list):
            value_loss = sum(F.mse_loss(v, target_value) for v in value)
        else:
            value_loss = F.mse_loss(value, target_value)
            
        total_loss = lm_loss + 1.0 * value_loss
        
        # 3. Hierarchical Weight Supervision: Entropy Loss
        # Prevents early collapse into a single-layer policy.
        if hasattr(model, 'layer_fusion_weights'):
            weights = torch.softmax(model.layer_fusion_weights, dim=0)
            entropy = -torch.sum(weights * torch.log(weights + 1e-8))
            total_loss = total_loss - 0.01 * entropy
            
        return total_loss

    def evaluate(self, model, device):
        # TODO: Implement basic perplexity eval
        return {"loss": 0.0}
