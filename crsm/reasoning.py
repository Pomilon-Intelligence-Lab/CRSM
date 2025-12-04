"""
MCTS (Monte Carlo Tree Search) based reasoning module for CRSM.
This implements the asynchronous tree search deliberation component.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import asyncio
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import numpy as np

@dataclass
class MCTSNode:
    state: torch.Tensor
    prior_p: float
    children: Dict[int, 'MCTSNode']
    parent: Optional['MCTSNode']
    visit_count: int = 0
    value_sum: float = 0.0
    
    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def expanded(self) -> bool:
        return len(self.children) > 0

class AsyncDeliberationLoop:
    def __init__(self, mamba_model, c_puct=1.0, n_simulations=50):
        """
        Args:
            mamba_model: The underlying Mamba model for state processing
            c_puct: Exploration constant for PUCT algorithm
            n_simulations: Number of MCTS simulations per deliberation
        """
        self.model = mamba_model
        self.c_puct = c_puct
        self.n_simulations = n_simulations
        self.top_k = 16  # Top actions to expand per node
        self.rollout_depth = 5  # Rollout depth for value estimation
        
        # Dynamics model (set by CRSM)
        self.dynamics_model = None
        
        # Sampling parameters (set by CRSM)
        self.use_sampling = True
        self.temperature = 0.8
        
    def select_action(self, node: MCTSNode) -> Tuple[MCTSNode, List[int]]:
        """Select the most promising action using PUCT algorithm"""
        path = []
        
        while node.expanded():
            max_ucb = float('-inf')
            best_action = -1
            best_child = None
            
            for action, child in node.children.items():
                if child.visit_count > 0:
                    q_value = child.value
                    u_value = (self.c_puct * child.prior_p * 
                             math.sqrt(node.visit_count) / (1 + child.visit_count))
                    ucb = q_value + u_value
                else:
                    ucb = float('inf')
                    
                if ucb > max_ucb:
                    max_ucb = ucb
                    best_action = action
                    best_child = child
                    
            path.append(best_action)
            node = best_child
            
        return node, path
    
    def expand_node(self, node: MCTSNode, logits: torch.Tensor, value: float):
        """Expand a leaf node using model predictions"""
        probs = torch.softmax(logits, dim=-1)
        topk = min(self.top_k, logits.size(-1))
        topv, topi = torch.topk(probs, topk)
        for p, a in zip(topv.tolist(), topi.tolist()):
            if p <= 0:
                continue
            next_state = self._get_next_state(node.state, int(a))
            child = MCTSNode(
                state=next_state,
                prior_p=float(p),
                children={},
                parent=node
            )
            node.children[int(a)] = child
                
    def _get_next_state(self, state, action: int):
        """Simulate next state - uses fast dynamics if available."""
        device = next(self.model.parameters()).device

        # Fast dynamics path
        if isinstance(state, list) and self.dynamics_model is not None:
            try:
                token = torch.tensor([[action]], dtype=torch.long, device=device)
                
                if hasattr(self.model, 'backbone') and hasattr(self.model.backbone, 'embedding'):
                    action_emb = self.model.backbone.embedding(token).squeeze(0).squeeze(0)
                else:
                    action_emb = self.model.embedding(token).squeeze(0).squeeze(0)
                
                next_states = []
                for layer_state in state:
                    if layer_state is None:
                        next_states.append(None)
                        continue
                    
                    s_input = layer_state.squeeze(0) if layer_state.dim() > 1 else layer_state
                    delta = self.dynamics_model(s_input, action_emb)
                    
                    if layer_state.dim() > 1:
                        next_layer_state = layer_state + delta.unsqueeze(0)
                    else:
                        next_layer_state = layer_state + delta
                    next_states.append(next_layer_state)
                
                return next_states
            except Exception:
                pass  # Fall through to slow path

        # Slow SSM fallback
        if isinstance(state, list):
            token = torch.tensor([[action]], dtype=torch.long, device=device)
            _, new_states = self.model.step(token, state)
            return new_states

        # Token sequence path
        if isinstance(state, torch.Tensor):
            return torch.cat([state, torch.tensor([action], device=state.device)])

        try:
            return torch.cat([state, torch.tensor([action], device=state.device)])
        except Exception:
            return torch.tensor([action], device=device)
    
    def backpropagate(self, node: MCTSNode, value: float, path: List[int]):
        """Update statistics of visited nodes"""
        cur = node
        while cur is not None:
            cur.visit_count += 1
            cur.value_sum += float(value)
            cur = cur.parent
            
    def deliberate_sync(self, seq: Optional[torch.Tensor], state: Optional[torch.Tensor]) -> Tuple[int, Optional[torch.Tensor], float]:
        """Blocking MCTS deliberation process."""
        # Prepare root state
        if state is not None:
            if isinstance(state, torch.Tensor):
                if state.dim() == 1:
                    root_state = state
                elif state.dim() == 2 and state.size(0) == 1:
                    root_state = state.squeeze(0)
                elif state.dim() == 2:
                    root_state = state[0]
                else:
                    root_state = state.view(-1)
            else:
                root_state = state
        elif seq is not None:
            if seq.dim() == 1:
                root_state = seq
            elif seq.dim() == 2 and seq.size(0) == 1:
                root_state = seq.squeeze(0)
            elif seq.dim() == 2:
                root_state = seq[0]
            else:
                root_state = seq.view(-1)
        else:
            root_state = torch.tensor([], device=next(self.model.parameters()).device)

        root = MCTSNode(state=root_state, prior_p=1.0, children={}, parent=None)

        # Initial expansion
        use_latent = state is not None
        device = next(self.model.parameters()).device
        with torch.no_grad():
            if use_latent:
                logits, value, _ = self.model.predict_from_states(state)
            else:
                logits, value, _ = self.model.predict_policy_value(root.state.unsqueeze(0))
        last_logits = logits[0, -1]
        self.expand_node(root, last_logits, float(value.item()) if hasattr(value, 'item') else float(value))

        # Run simulations
        for _ in range(self.n_simulations):
            leaf, path = self.select_action(root)

            if not leaf.expanded():
                with torch.no_grad():
                    if use_latent:
                        logits, value, _ = self.model.predict_from_states(state)
                    else:
                        logits, value, _ = self.model.predict_policy_value(leaf.state.unsqueeze(0))
                last_logits = logits[0, -1]
                self.expand_node(leaf, last_logits, float(value.item()) if hasattr(value, 'item') else float(value))
                rollout_value = float(value.item()) if hasattr(value, 'item') else float(value)
            else:
                rollout_value = self._rollout_value(leaf)

            self.backpropagate(leaf, rollout_value, path)

        # Select best action
        actions = list(root.children.keys())
        if not actions:
            return 0, None, 0.0

        visit_counts = [root.children[a].visit_count for a in actions]
        best = actions[visit_counts.index(max(visit_counts))]
        
        # Get confidence from the best child's value estimate
        best_child = root.children[best]
        confidence = best_child.value

        delta = self._compute_delta_from_mcts(root, best)
        return int(best), delta, confidence
    
    def _compute_delta_from_mcts(self, root: MCTSNode, best_action: int) -> Optional[List[torch.Tensor]]:
        """
        Compute the update target from MCTS.
        
        Changed: Now returns the TARGET STATE (best child state) directly,
        rather than a difference vector, to support Gated Injection.
        """
        if best_action not in root.children:
            return None
        
        best_child = root.children[best_action]
        
        if not isinstance(root.state, list) or not isinstance(best_child.state, list):
            return None
        
        try:
            # Return the best child's state directly
            targets = []
            for child_layer_state in best_child.state:
                if child_layer_state is None:
                    targets.append(None)
                    continue
                
                if not isinstance(child_layer_state, torch.Tensor):
                    targets.append(None)
                    continue
                
                targets.append(child_layer_state)
            
            return targets
        except Exception:
            return None

    async def deliberate(self, seq: Optional[torch.Tensor], state: Optional[torch.Tensor]) -> Tuple[int, Optional[torch.Tensor], float]:
        """Async wrapper for deliberation."""
        return await asyncio.to_thread(self.deliberate_sync, seq, state)

    def _rollout_value(self, node: MCTSNode) -> float:
        """Improved rollout with optional sampling."""
        state = node.state
        device = next(self.model.parameters()).device
        
        try:
            is_token_seq = state.dtype in (torch.long, torch.int)
        except Exception:
            is_token_seq = False

        if is_token_seq:
            for _ in range(self.rollout_depth):
                with torch.no_grad():
                    logits, value, _ = self.model.predict_policy_value(state.unsqueeze(0))
                last_logits = logits[0, -1]
                
                if self.use_sampling:
                    probs = F.softmax(last_logits / self.temperature, dim=-1)
                    action = torch.multinomial(probs, 1).item()
                else:
                    action = int(torch.argmax(last_logits).item())
                
                state = self._get_next_state(state, action)
            
            with torch.no_grad():
                _, value, _ = self.model.predict_policy_value(state.unsqueeze(0))
            return float(value.item()) if hasattr(value, 'item') else float(value)
        else:
            if isinstance(state, list):
                with torch.no_grad():
                    _, value, _ = self.model.predict_from_states(state)
                return float(value.item()) if hasattr(value, 'item') else float(value)
            else:
                dummy = torch.zeros((1, 1), dtype=torch.long, device=device)
                with torch.no_grad():
                    _, value, _ = self.model.predict_policy_value(dummy)
                return float(value.item()) if hasattr(value, 'item') else float(value)