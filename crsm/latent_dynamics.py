import torch
from torch import nn

class LatentDynamics(nn.Module):
    def __init__(self, d_model, action_dim=None):
        super().__init__() 
        # If action_dim not provided, use d_model (embed actions same as states)
        if action_dim is None:
            action_dim = d_model
        self.net = nn.Sequential(
            nn.Linear(d_model + action_dim, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
    
    def forward(self, state, action_emb):
        return self.net(torch.cat([state, action_emb], -1))