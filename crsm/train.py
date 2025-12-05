import os
import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .dataset import RandomTokenDataset, RealTextDataset, StreamingTextDataset
from .mamba_ssm import MambaModel
from .utils import set_seed

def train_one_epoch(model, dataloader, optimizer, device, grad_accum_steps: int = 1, 
                   use_amp: bool = False, use_value_loss: bool = True):
    """
    Train for one epoch with optional value head training.
    
    Args:
        use_value_loss: If True, train value head to predict future loss (CRSM mode)
                       If False, standard LM training only
    """
    model.train()
    total_loss = 0.0
    total_lm_loss = 0.0
    total_value_loss = 0.0
    
    criterion = nn.CrossEntropyLoss(reduction='none')  # Changed to 'none' for per-token loss
    scaler = torch.amp.GradScaler() if use_amp else None
    step = 0
    
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        
        if step % grad_accum_steps == 0:
            optimizer.zero_grad()

        if use_amp and scaler is not None:
            with torch.amp.autocast(device_type=device.type):
                loss = compute_loss_with_value(model, x, y, criterion, use_value_loss)
                loss = loss / grad_accum_steps
            scaler.scale(loss).backward()
            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
        else:
            loss = compute_loss_with_value(model, x, y, criterion, use_value_loss)
            loss = loss / grad_accum_steps
            loss.backward()
            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        total_loss += loss.item() * grad_accum_steps
        step += 1

    avg_loss = total_loss / max(1, step)
    return avg_loss


def compute_loss_with_value(model, x, y, criterion, use_value_loss=True):
    """
    Compute combined LM loss and value loss.
    
    The value head is trained to predict the average next-token loss,
    which serves as a proxy for "how good is this state".
    """
    # Get predictions with value head
    if use_value_loss:
        logits, value, _ = model.predict_policy_value(x, states=None)
    else:
        logits, _ = model(x, states=None)
        value = None
    
    b, t, v = logits.size()
    
    # Standard language modeling loss
    lm_loss_per_token = criterion(logits.reshape(b * t, v), y.reshape(b * t))
    lm_loss = lm_loss_per_token.mean()
    
    # Value loss: train value head to predict probability of correct next token
    # Target: V = P(x_{t+1} | h_t)
    if use_value_loss and value is not None:
        with torch.no_grad():
            # Calculate log probabilities from logits
            log_probs = F.log_softmax(logits, dim=-1)
            
            # We focus on the last step because 'value' is returned for the last step only
            # logits shape: (b, t, v)
            # y shape: (b, t)
            
            # Last step predictions
            last_step_log_probs = log_probs[:, -1, :]  # (b, v)
            last_step_targets = y[:, -1]               # (b,)
            
            # Gather log prob of the true token
            token_log_probs = last_step_log_probs.gather(1, last_step_targets.unsqueeze(-1)).squeeze(-1)
            
            # Target value is the probability (0.0 to 1.0)
            target_value = torch.exp(token_log_probs)

        # Value is (batch,) from last token
        value_loss = F.mse_loss(value, target_value)
        
        # Combined loss with value weight (increased from 0.1 to 1.0 to ensure signal)
        total_loss = lm_loss + 1.0 * value_loss
    else:
        total_loss = lm_loss
    
    return total_loss


def save_checkpoint(model, optimizer, epoch, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),  # Changed from 'model_state'
        'optimizer_state_dict': optimizer.state_dict()  # Changed from 'optim_state'
    }, path)


def main(
    epochs: int = 1,
    batch_size: int = 8,
    vocab_size: int = 1000,
    seq_len: int = 32,
    lr: float = 1e-3,
    seed: int = 42,
    data_dir: str | None = None,
    checkpoint_dir: str = 'checkpoints',
    device_str: str | None = None,
    save_every: int = 1,
    wandb_enabled: bool = True,
    wandb_project: str = 'crsm',
    grad_accum_steps: int = 1,
    use_amp: bool = False,
    num_workers: int = 0,
    resume: str | None = None,
    distributed: bool = False,
    local_rank: int | None = None,
    use_value_loss: bool = True,
    hf_tokenizer_name: str | None = None,  # NEW: Add tokenizer parameter
    d_model: int = 128,
    d_state: int = 64,
    d_ffn: int = 512,
    num_layers: int = 2
):
    set_seed(seed)

    # Allow torchrun to set LOCAL_RANK env var
    if local_rank is None:
        try:
            lr_env = int(os.environ.get('LOCAL_RANK', -1))
            local_rank = lr_env if lr_env >= 0 else None
        except Exception:
            local_rank = None

    # Distributed init if requested and local_rank present
    if distributed and local_rank is not None:
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        torch.distributed.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device('cuda', local_rank)
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(device_str if device_str else ('cuda' if torch.cuda.is_available() else 'cpu'))

    if data_dir:
        # FIXED: Pass vocab_size and hf_tokenizer_name to dataset
        if hf_tokenizer_name:
            ds = StreamingTextDataset(
                data_dir=data_dir, 
                seq_len=seq_len, 
                hf_tokenizer_name=hf_tokenizer_name
            )
        else:
            ds = StreamingTextDataset(
                data_dir=data_dir, 
                seq_len=seq_len, 
                vocab_size=vocab_size
            )
        collate_fn = None
        shuffle = False
    else:
        ds = RandomTokenDataset(vocab_size=vocab_size, seq_len=seq_len, size=2000)
        collate_fn = None
        shuffle = True

    if distributed:
        try:
            from torch.utils.data.distributed import DistributedSampler
            sampler = DistributedSampler(ds)
            dl = DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=False)
        except Exception:
            dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=False)
    else:
        dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=False)

    model = MambaModel(vocab_size=vocab_size, d_model=d_model, d_state=d_state, d_ffn=d_ffn, num_layers=num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if distributed and local_rank is not None:
        try:
            from torch.nn.parallel import DistributedDataParallel as DDP
            model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None)
        except Exception as e:
            print('Failed to wrap model in DDP:', e)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    # resume logic (unchanged)
    if resume is not None and os.path.exists(resume):
        print(f"\n{'='*60}")
        print(f"Resuming from checkpoint: {resume}")
        print('='*60)
        
        ckpt = torch.load(resume, map_location=device)
        
        try:
            # Extract state dict from various formats
            if 'model_state_dict' in ckpt:
                state_dict = ckpt['model_state_dict']
                print("Format: New (model_state_dict)")
            elif 'model_state' in ckpt:
                state_dict = ckpt['model_state']
                print("Format: Old (model_state)")
            else:
                state_dict = ckpt
                print("Format: Raw state_dict")
            
            # Handle CRSM checkpoint (strip 'backbone.' prefix)
            if any(k.startswith('backbone.') for k in state_dict.keys()):
                print("→ Detected CRSM checkpoint, extracting backbone weights...")
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('backbone.'):
                        new_state_dict[k.replace('backbone.', '', 1)] = v
                    elif not k.startswith(('dynamics.', 'reasoning.')):
                        new_state_dict[k] = v
                state_dict = new_state_dict
                print(f"→ Extracted {len(state_dict)} parameters")
            
            # Load model weights
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"⚠ Missing keys: {len(missing)} (this is OK if fine-tuning)")
            if unexpected:
                print(f"⚠ Unexpected keys: {len(unexpected)} (ignored)")
            
            # Load optimizer if available
            opt_loaded = False
            if 'optimizer_state_dict' in ckpt and ckpt['optimizer_state_dict'] is not None:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                opt_loaded = True
            elif 'optim_state' in ckpt:
                optimizer.load_state_dict(ckpt['optim_state'])
                opt_loaded = True
            
            if opt_loaded:
                print("✓ Loaded optimizer state")
            
            start_epoch = ckpt.get('epoch', 0) + 1
            print(f"✓ Successfully resumed from epoch {start_epoch}")
            
        except Exception as e:
            print(f'✗ Failed to load checkpoint: {type(e).__name__}')
            print(f'   {e}')
            print('\n⚠ Continuing from scratch')
            start_epoch = 1
    else:
        start_epoch = 1

    # wandb
    try:
        import wandb
        if wandb_enabled:
            try:
                wandb.init(project=wandb_project, config={
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'lr': lr,
                    'seq_len': seq_len,
                    'use_value_loss': use_value_loss,
                })
            except Exception:
                wandb_enabled = False
    except Exception:
        wandb = None
        wandb_enabled = False

    for epoch in range(start_epoch, epochs + 1):
        # set epoch for distributed sampler
        if distributed and 'sampler' in locals() and hasattr(dl.sampler, 'set_epoch'):
            dl.sampler.set_epoch(epoch)

        loss = train_one_epoch(model, dl, optimizer, device, 
                              grad_accum_steps=grad_accum_steps, 
                              use_amp=use_amp,
                              use_value_loss=use_value_loss)

        try:
            scheduler.step()
        except Exception:
            pass

        # figure rank
        rank = 0
        if distributed:
            try:
                rank = torch.distributed.get_rank()
            except Exception:
                rank = local_rank or 0
        is_rank0 = (rank == 0)

        if is_rank0:
            mode = "with value loss" if use_value_loss else "LM only"
            print(f"Epoch {epoch}/{epochs} loss={loss:.4f} ({mode})")
            if wandb_enabled:
                try:
                    wandb.log({'epoch': epoch, 'loss': loss})
                except Exception:
                    pass

        if epoch % save_every == 0 and is_rank0:
            path = os.path.join(checkpoint_dir, f'crsm_epoch{epoch}.pt')
            tmp_path = path + '.tmp'
            save_checkpoint(model, optimizer, epoch, tmp_path)
            try:
                os.replace(tmp_path, path)
            except Exception:
                try:
                    os.rename(tmp_path, path)
                except Exception:
                    pass

    # cleanup
    if distributed:
        try:
            torch.distributed.barrier()
        except Exception:
            pass
        try:
            torch.distributed.destroy_process_group()
        except Exception:
            pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--vocab-size', type=int, default=1000)
    parser.add_argument('--seq-len', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data-dir', type=str, default=None, help='Path to text files directory')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('--device', type=str, default=None, help='CUDA device or cpu')
    parser.add_argument('--no-value-loss', action='store_true', help='Disable value head training (standard LM only)')
    args = parser.parse_args()
    
    main(epochs=args.epochs, batch_size=args.batch_size, vocab_size=args.vocab_size, 
         seq_len=args.seq_len, lr=args.lr, seed=args.seed, data_dir=args.data_dir, 
         checkpoint_dir=args.checkpoint_dir, device_str=args.device,
         use_value_loss=(not args.no_value_loss))