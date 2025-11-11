"""
Mamba SSM (State Space Model) backbone implementation.
This module implements the continuous state modeling component of CRSM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional
from .s4_adapter import get_ssm_block_factory
from .s4_vendor import S4LiteBlock


# Attempt to prefer an installed Mamba implementation; fall back to vendor S4LiteBlock
_ssm_factory = get_ssm_block_factory(prefer='mamba')
SSM_BACKEND = getattr(_ssm_factory, '__module__', 's4lite')

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state, d_ffn, dropout=0.1):
        super().__init__()
        # Use production SSM if available, else S4LiteBlock (d_state retained for API)
        try:
            self.ssm = _ssm_factory(d_model, kernel_size=3, dropout=dropout, num_layers=3)
        except Exception:
            self.ssm = S4LiteBlock(d_model, kernel_size=3, dropout=dropout, num_layers=3)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model)
        )
        
    def forward(self, x, state=None):
        # SSM branch
        ssm_out, new_state = self.ssm(x, state)
        x = x + ssm_out
        
        # FFN branch
        ffn_out = self.ffn(x)
        x = x + ffn_out
        
        return x, new_state

class MambaModel(nn.Module):
    def __init__(self, 
                 vocab_size,
                 d_model=256,
                 d_state=64,
                 d_ffn=1024,
                 num_layers=4,
                 dropout=0.1):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_ffn, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)
        # Small value head for MCTS evaluations
        self.value_head = nn.Linear(d_model, 1)
        
    def forward(self, x, states=None):
        x = self.embedding(x)
        
        if states is None:
            states = [None] * len(self.layers)
            
        new_states = []
        for layer, state in zip(self.layers, states):
            x, new_state = layer(x, state)
            new_states.append(new_state)
            
        x = self.norm(x)
        logits = self.output(x)
        
        return logits, new_states

    def init_state(self, batch_size: int = 1, device: Optional[torch.device] = None):
        """Initialize a list of per-layer states for the SSM blocks.

        This attempts to call each layer's SSM init_state if available. If not,
        it falls back to a zero tensor with the block's d_model dimension when
        possible, otherwise None.
        Returns:
            list: new_states per layer (length == num_layers)
        """
        device = device or next(self.parameters()).device
        states = []
        for layer in self.layers:
            ssm = layer.ssm
            # prefer an SSM-provided initializer
            if hasattr(ssm, 'init_state'):
                try:
                    st = ssm.init_state(batch_size=batch_size, device=device)
                except Exception:
                    st = None
            else:
                # fallback to zeros if the block exposes d_model
                if hasattr(ssm, 'd_model'):
                    st = torch.zeros(batch_size, ssm.d_model, device=device)
                else:
                    st = None
            states.append(st)
        return states

    def step(self, x, states=None):
        """Single-step update: run the model on a single token (or short chunk)

        Args:
            x: token ids tensor with shape (batch,) or (batch, 1) or (batch, seq_len)
            states: optional list of previous states
        Returns:
            logits: (batch, seq_len, vocab) or (batch, 1, vocab)
            new_states: list of new states per layer
        """
        # reuse forward implementation by ensuring seq dim
        if x.dim() == 1:
            x_in = x.unsqueeze(1)
        else:
            x_in = x
        logits, new_states = self.forward(x_in, states)
        return logits, new_states

    def predict_policy_value(self, x, states=None):
        """Return policy logits and a scalar value estimate for the input sequence.

        Args:
            x: token ids tensor (batch, seq_len)
        Returns:
            logits: (batch, seq_len, vocab)
            value: (batch,) scalar value estimates (from last token)
            new_states: list of states per layer
        """
        h = self.embedding(x)
        if states is None:
            states = [None] * len(self.layers)
        new_states = []
        for layer, state in zip(self.layers, states):
            h, new_state = layer(h, state)
            new_states.append(new_state)

        h = self.norm(h)
        logits = self.output(h)
        # value from last token
        last_hidden = h[:, -1, :]
        value = self.value_head(last_hidden).squeeze(-1)
        return logits, value, new_states

    def predict_from_states(self, states):
        """Predict policy logits and value directly from per-layer latent states.

        Args:
            states: list of per-layer state tensors or a single tensor.
        Returns:
            logits: (batch, 1, vocab)
            value: (batch,) scalar estimates
            new_states: None (not applicable)
        """
        device = next(self.parameters()).device
        d_model = self.output.in_features

        if states is None:
            h = torch.zeros((1, d_model), device=device)
        elif isinstance(states, list):
            tensors = [s for s in states if isinstance(s, torch.Tensor) and s.numel() > 0]
            if not tensors:
                h = torch.zeros((1, d_model), device=device)
            else:
                # ensure batch dim
                normed = []
                for s in tensors:
                    if s.dim() == 1:
                        s = s.unsqueeze(0)
                    normed.append(s)
                # align dims if necessary by slicing or padding (best-effort)
                batch = normed[0].shape[0]
                aligned = []
                for s in normed:
                    if s.shape[0] != batch:
                        # try to expand or repeat
                        try:
                            s = s.expand(batch, -1)
                        except Exception:
                            s = s.repeat(batch, 1)
                    if s.shape[1] != d_model:
                        # slice or pad with zeros
                        if s.shape[1] > d_model:
                            s = s[:, :d_model]
                        else:
                            pad = torch.zeros((batch, d_model - s.shape[1]), device=device)
                            s = torch.cat([s, pad], dim=1)
                    aligned.append(s)
                h = sum(aligned) / len(aligned)
        elif isinstance(states, torch.Tensor):
            s = states
            if s.dim() == 1:
                s = s.unsqueeze(0)
            if s.shape[1] != d_model:
                if s.shape[1] > d_model:
                    s = s[:, :d_model]
                else:
                    pad = torch.zeros((s.shape[0], d_model - s.shape[1]), device=device)
                    s = torch.cat([s, pad], dim=1)
            h = s
        else:
            # unknown type
            h = torch.zeros((1, d_model), device=device)

        # convert to (batch, seq=1, d_model) so we can reuse norm/output heads
        h_seq = h.unsqueeze(1)
        h_seq = self.norm(h_seq)
        logits = self.output(h_seq)
        last_hidden = h_seq[:, -1, :]
        value = self.value_head(last_hidden).squeeze(-1)
        return logits, value, None