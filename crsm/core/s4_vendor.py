"""Vendor fallback S4-like block used when no upstream SSM package is installed.

This module isolates the lightweight PoC S4 implementation so the adapter
can import it without circular dependencies.
"""
import torch
import torch.nn as nn


class S4LiteBlock(nn.Module):
    """A lightweight S4-like block for PoC.

    This is a simple, dependency-free implementation that approximates
    long-range mixing using dilated 1D convolutions and gating. It's
    intentionally small and easy to replace with a real S4/Mamba backend.
    """
    def __init__(self, d_model, kernel_size=3, dropout=0.1, num_layers=4):
        super().__init__()
        self.d_model = d_model
        convs = []
        for i in range(num_layers):
            dilation = 2 ** i
            pad = (kernel_size - 1) * dilation
            convs.append(nn.Conv1d(d_model, d_model * 2, kernel_size, padding=pad, dilation=dilation, groups=1))
        self.convs = nn.ModuleList(convs)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, state=None):
        # x: (batch, seq_len, d_model)
        x = self.norm(x)
        b, t, d = x.shape
        h = x.transpose(1, 2)  # (b, d, t)

        for conv in self.convs:
            out = conv(h)[:, :, -t:]
            out = out.transpose(1, 2).reshape(b, t, 2 * d)
            gate, cand = out.split(d, dim=-1)
            gate = torch.sigmoid(gate)
            cand = torch.tanh(cand)
            h = (gate * cand + (1 - gate) * h.transpose(1, 2)).transpose(1, 2)

        out = h.transpose(1, 2)  # (b, t, d)
        out = self.dropout(out)
        new_state = out[:, -1, :].detach()
        return out, new_state
