import os
import torch
import torch.nn as nn
from .logger import logger

class Trainer:
    """Task-agnostic training engine for CRSM."""
    
    def __init__(self, model, optimizer, config):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.grad_accum_steps = config.get('grad_accum_steps', 1)
        self.clip_grad_norm = config.get('clip_grad_norm', 1.0)
        self.use_amp = config.get('use_amp', False)
        self.scaler = torch.amp.GradScaler() if self.use_amp else None
        
        self.model.to(self.device)

    def fit(self, task, epochs, checkpoint_dir='checkpoints'):
        os.makedirs(checkpoint_dir, exist_ok=True)
        train_loader, val_loader = task.get_dataloaders(self.config.get('batch_size', 8))
        
        for epoch in range(1, epochs + 1):
            avg_loss = self.train_one_epoch(task, train_loader)
            
            # Validation pass
            val_metrics = {}
            if val_loader:
                val_loss = self.validate(task, val_loader)
                val_metrics['val_loss'] = val_loss
                logger.info(f"Epoch {epoch}/{epochs} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}")
            else:
                logger.info(f"Epoch {epoch}/{epochs} | Train Loss: {avg_loss:.4f}")
            
            # Save checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
            self.save_checkpoint(epoch, checkpoint_path)

    def validate(self, task, dataloader):
        self.model.eval()
        total_loss = 0.0
        step = 0
        
        with torch.no_grad():
            for batch in dataloader:
                loss = task.compute_loss(self.model, batch, self.device)
                total_loss += loss.item()
                step += 1
                
        return total_loss / max(1, step)

    def train_one_epoch(self, task, dataloader):
        self.model.train()
        total_loss = 0.0
        step = 0
        
        for batch in dataloader:
            if step % self.grad_accum_steps == 0:
                self.optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast(device_type=self.device.type):
                    loss = task.compute_loss(self.model, batch, self.device)
                    loss = loss / self.grad_accum_steps
                self.scaler.scale(loss).backward()
                
                if (step + 1) % self.grad_accum_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                loss = task.compute_loss(self.model, batch, self.device)
                loss = loss / self.grad_accum_steps
                loss.backward()
                
                if (step + 1) % self.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    self.optimizer.step()

            total_loss += loss.item() * self.grad_accum_steps
            step += 1
            
        return total_loss / max(1, step)

    def save_checkpoint(self, epoch, path):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)
