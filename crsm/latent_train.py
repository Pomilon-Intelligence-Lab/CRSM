# crsm/latent_train.py
import os
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader

from .mamba_ssm import MambaModel
from .utils import set_seed

class ShardedTokenDataset(IterableDataset):
    def __init__(self, shards_dir: str):
        self.shards = sorted(Path(shards_dir).glob('*.jsonl'))

    def __iter__(self):
        for s in self.shards:
            with s.open('r', encoding='utf-8') as f:
                for line in f:
                    obj = json.loads(line)
                    inp = torch.tensor(obj['input_ids'], dtype=torch.long)
                    tgt = torch.tensor(obj['target_ids'], dtype=torch.long)
                    yield inp, tgt

def collate_batch(batch):
    inps, tgts = zip(*batch)
    max_in = max(x.size(0) for x in inps)
    max_t = max(x.size(0) for x in tgts)
    inps_p = torch.zeros(len(inps), max_in, dtype=torch.long)
    tgts_p = torch.zeros(len(tgts), max_t, dtype=torch.long)
    for i, x in enumerate(inps):
        inps_p[i, :x.size(0)] = x
    for i, x in enumerate(tgts):
        tgts_p[i, :x.size(0)] = x
    return inps_p, tgts_p

def train(shards_dir: str, epochs: int = 1, batch_size: int = 4, lr: float = 1e-4, device: str | None = None):
    set_seed(42)
    device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    ds = ShardedTokenDataset(shards_dir)
    dl = DataLoader(ds, batch_size=batch_size, collate_fn=collate_batch)

    vocab_size = 10000
    model = MambaModel(vocab_size=vocab_size, d_model=256, d_state=64, d_ffn=1024, num_layers=3).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        total = 0.0
        model.train()
        nsteps = 0
        for inp, tgt in dl:
            inp = inp.to(device)
            tgt = tgt.to(device)
            opt.zero_grad()
            logits, _ = model(inp)
            b, t, v = logits.size()
            # If target length != t, pad/truncate to t
            if tgt.size(1) != t:
                # simple align: pad/truncate tgt to t
                if tgt.size(1) < t:
                    pad = torch.zeros(b, t - tgt.size(1), dtype=tgt.dtype, device=tgt.device)
                    tgt2 = torch.cat([tgt, pad], dim=1)
                else:
                    tgt2 = tgt[:, :t]
            else:
                tgt2 = tgt
            loss = crit(logits.view(b * t, v), tgt2.view(b * t))
            loss.backward()
            opt.step()
            total += loss.item()
            nsteps += 1
        avg = total / max(1, nsteps)
        print(f'Epoch {epoch} avg_loss={avg:.4f}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shards-dir', required=True)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()
    train(args.shards_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=args.device)

if __name__ == '__main__':
    main()