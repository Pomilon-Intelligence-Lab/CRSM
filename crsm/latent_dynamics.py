import torch
from torch import nn

class LatentDynamics(nn.Module):
    def __init__(self, d_model, num_layers, action_dim=None):
        super().__init__() 
        if action_dim is None:
            action_dim = d_model
            
        # Shared Transition Core: Captures the global effect of an action.
        # This keeps the parameter count low while modeling complex transitions.
        self.gru = nn.GRUCell(action_dim, d_model)
        
        # Hierarchical Broadcaster: Projects global intent to layer-specific deltas.
        # Uses Depthwise modulation (element-wise scale and bias) to allow 
        # independent control of each layer's update magnitude and direction.
        # Params: 2 * num_layers * d_model (approx 12k for 24 layers, 256 d_model)
        # Initialized to 0.1 for stability (Warm-up).
        self.layer_scales = nn.Parameter(torch.ones(num_layers, d_model) * 0.1)
        self.layer_biases = nn.Parameter(torch.zeros(num_layers, d_model))
        
        self.num_layers = num_layers
    
    def forward(self, states, action_emb):
        """
        Computes layer-wise deltas for the entire state hierarchy.
        
        Args:
            states: List[Tensor] of len num_layers, or Tensor (Batch, num_layers, d_model)
            action_emb: Tensor (Batch, action_dim)
        Returns:
            List[Tensor] or Tensor: Layer-wise deltas
        """
        is_list = isinstance(states, list)
        
        # 1. Hierarchical Context Aggregation
        if is_list:
            valid_states = [s for s in states if s is not None]
            if not valid_states:
                # Return zero deltas if no state provided
                device = action_emb.device
                res = [torch.zeros_like(action_emb) for _ in range(self.num_layers)]
                return res
            # Aggregate hierarchy into a single context vector for the transition logic
            h_context = torch.stack(valid_states, dim=1).mean(dim=1)
        elif states.dim() == 3:
            h_context = states.mean(dim=1)
        else:
            # Fallback for legacy (Batch, d_model) inputs
            h_context = states

        # Ensure batch dimensions
        if h_context.dim() == 1: h_context = h_context.unsqueeze(0)
        if action_emb.dim() == 1: action_emb = action_emb.unsqueeze(0)

        # 2. Compute Global Delta Intent
        next_h = self.gru(action_emb, h_context)
        global_delta = next_h - h_context # (Batch, d_model)

        # 3. Depthwise Broadcasting
        # Delta_i = Global_Delta * Scale_i + Bias_i
        scales = self.layer_scales.unsqueeze(0)  # (1, num_layers, d_model)
        biases = self.layer_biases.unsqueeze(0)  # (1, num_layers, d_model)
        
        # (Batch, 1, d_model) * (1, N, d_model) -> (Batch, N, d_model)
        layer_deltas = global_delta.unsqueeze(1) * scales + biases
        
        if is_list:
            return [layer_deltas[:, i, :] for i in range(self.num_layers)]
        
        if states.dim() == 2:
            # Return single layer delta for legacy single-layer input
            return layer_deltas[:, 0, :]
            
        return layer_deltas