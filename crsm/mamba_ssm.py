"""
Mamba SSM (State Space Model) backbone implementation.
This module implements the continuous state modeling component of CRSM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
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

    def predict_policy_value(self, x):
        """Return policy logits and a scalar value estimate for the input sequence.

        Args:
            x: token ids tensor (batch, seq_len)
        Returns:
            logits: (batch, seq_len, vocab)
            value: (batch,) scalar value estimates (from last token)
            new_states: list of states per layer
        """
        h = self.embedding(x)
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