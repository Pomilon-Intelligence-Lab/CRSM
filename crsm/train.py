"""
Training harness for SLM PoC with checkpointing and simple CLI integration.
Use `crsm.cli` to launch training or run `python -m crsm.train`.
"""
import os
import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from .dataset import RandomTokenDataset, RealTextDataset, StreamingTextDataset
from .mamba_ssm import MambaModel
from .utils import set_seed

def train_one_epoch(model, dataloader, optimizer, device, grad_accum_steps: int = 1, use_amp: bool = False):
    model.train()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    step = 0
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        if step % grad_accum_steps == 0:
            optimizer.zero_grad()

        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                logits, _ = model(x)
                b, t, v = logits.size()
                loss = criterion(logits.reshape(b * t, v), y.reshape(b * t)) / grad_accum_steps
            scaler.scale(loss).backward()
            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
        else:
            logits, _ = model(x)
            b, t, v = logits.size()
            loss = criterion(logits.reshape(b * t, v), y.reshape(b * t)) / grad_accum_steps
            loss.backward()
            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        total_loss += (loss.item() if 'loss' in locals() else 0.0) * grad_accum_steps
        step += 1

    avg_loss = total_loss / max(1, step)
    return avg_loss

def save_checkpoint(model, optimizer, epoch, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optim_state': optimizer.state_dict()
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
        ds = StreamingTextDataset(data_dir=data_dir, seq_len=seq_len)
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
            dl = DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)
        except Exception:
            dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    else:
        dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

    model = MambaModel(vocab_size=vocab_size, d_model=128, d_state=64, d_ffn=512, num_layers=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if distributed and local_rank is not None:
        try:
            from torch.nn.parallel import DistributedDataParallel as DDP
            model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None)
        except Exception as e:
            print('Failed to wrap model in DDP:', e)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    # resume
    if resume is not None and os.path.exists(resume):
        ckpt = torch.load(resume, map_location=device)
        try:
            model.load_state_dict(ckpt['model_state'])
            optimizer.load_state_dict(ckpt['optim_state'])
            start_epoch = ckpt.get('epoch', 0) + 1
            print(f"Resumed from checkpoint {resume}, starting at epoch {start_epoch}")
        except Exception:
            print('Failed to load checkpoint states; continuing from scratch')
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

        loss = train_one_epoch(model, dl, optimizer, device, grad_accum_steps=grad_accum_steps, use_amp=use_amp)

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
            print(f"Epoch {epoch}/{epochs} loss={loss:.4f}")
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
    args = parser.parse_args()
    main(epochs=args.epochs, batch_size=args.batch_size, vocab_size=args.vocab_size, seq_len=args.seq_len,
         lr=args.lr, seed=args.seed, data_dir=args.data_dir, checkpoint_dir=args.checkpoint_dir, device_str=args.device)