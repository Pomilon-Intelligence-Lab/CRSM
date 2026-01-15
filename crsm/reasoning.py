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
    prior_p: float
    children: Dict[int, 'MCTSNode']
    parent: Optional['MCTSNode']
    action: Optional[int] = None  # The action that led to this node
    state_cache: Optional[List[torch.Tensor]] = None # Cached state (list of tensors)
    visit_count: int = 0
    value_sum: float = 0.0
    layer_value_sums: Optional[List[float]] = None # Sum of values per layer
    
    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        
        # If we have layer values, implement Weighted Consensus with Uncertainty Penalty
        if self.layer_value_sums:
            layer_means = [s / self.visit_count for s in self.layer_value_sums]
            
            # Compute mean and standard deviation across layers
            # High variance across layers indicates the model's abstraction levels disagree
            # which we treat as "unstable" or "uncertain".
            v_tensor = torch.tensor(layer_means)
            mean_v = v_tensor.mean().item()
            # If only one layer, std is 0
            std_v = v_tensor.std().item() if v_tensor.numel() > 1 else 0.0
            
            # Consensus = Mean Value - Lambda * Uncertainty
            return mean_v - 0.1 * std_v
            
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
        
    def _reconstruct_state(self, node: MCTSNode, root_state: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Reconstructs the state of a node by replaying dynamics from the nearest cached ancestor.
        Solves the 'Memory Wall' by trading compute for VRAM.
        """
        # 1. Find path to nearest cached state
        path = []
        curr = node
        while curr.state_cache is None and curr.parent is not None:
            path.append(curr.action)
            curr = curr.parent
        
        # Base state (from cache or root)
        state = curr.state_cache if curr.state_cache is not None else root_state
        
        # 2. Replay dynamics forward
        # path is reversed (leaf -> root), so reverse it back
        for action in reversed(path):
            state = self._get_next_state(state, action)
            
        # 3. Optional: Cache this state if it's deep in the tree or frequently visited
        # For now, we only cache if it's the node we asked for (to save this computation for children)
        node.state_cache = state
        return state

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

    def project_future_state(self, state: List[torch.Tensor], k_steps: int) -> List[torch.Tensor]:
        """
        Fast-forward a state by k steps using the dynamics model.
        Used to align the planner with a future generation position.
        """
        if k_steps <= 0:
            return state
            
        curr_state = state
        with torch.no_grad():
            for _ in range(k_steps):
                # 1. Predict next token (greedy)
                logits, _, _ = self.model.predict_from_states(curr_state)
                action = torch.argmax(logits[0, -1]).item()
                
                # 2. Advance state
                curr_state = self._get_next_state(curr_state, action)
                
        return curr_state
    
    def expand_node(self, node: MCTSNode, logits: torch.Tensor, value: float):
        """Expand a leaf node using model predictions"""
        probs = torch.softmax(logits, dim=-1)
        topk = min(self.top_k, logits.size(-1))
        topv, topi = torch.topk(probs, topk)
        for p, a in zip(topv.tolist(), topi.tolist()):
            if p <= 0:
                continue
            # Memory Wall Fix: Do NOT compute or store next_state here.
            # Just store the action. State is reconstructed on demand.
            child = MCTSNode(
                prior_p=float(p),
                children={},
                parent=node,
                action=int(a),
                state_cache=None # Lazy
            )
            node.children[int(a)] = child
                
    def _get_next_state(self, state, action: int):
        """Simulate next state - uses fast dynamics if available."""
        device = next(self.model.parameters()).device

        # Fast dynamics path: MULTI-LAYER BROADCASTER
        if isinstance(state, list) and self.dynamics_model is not None:
            try:
                token = torch.tensor([[action]], dtype=torch.long, device=device)
                
                if hasattr(self.model, 'backbone') and hasattr(self.model.backbone, 'embedding'):
                    action_emb = self.model.backbone.embedding(token).squeeze(0).squeeze(0)
                else:
                    action_emb = self.model.embedding(token).squeeze(0).squeeze(0)
                
                # Broadcaster call: Call once for all layers
                layer_deltas = self.dynamics_model(state, action_emb)
                
                next_states = []
                for i, layer_state in enumerate(state):
                    if layer_state is None:
                        next_states.append(None)
                        continue
                    
                    delta = layer_deltas[i]
                    if layer_state.dim() > 1:
                        next_layer_state = layer_state + (delta.unsqueeze(0) if delta.dim() == 1 else delta)
                    else:
                        next_layer_state = layer_state + delta.squeeze(0)
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
            if isinstance(value, list):
                if cur.layer_value_sums is None:
                    # Initialize with zeros for each layer
                    cur.layer_value_sums = [0.0] * len(value)
                for i, v in enumerate(value):
                    cur.layer_value_sums[i] += v
            else:
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

        # Root node always has the state cached
        root = MCTSNode(prior_p=1.0, children={}, parent=None, action=None, state_cache=root_state)

        # Initial expansion
        use_latent = state is not None
        device = next(self.model.parameters()).device
        with torch.no_grad():
            if use_latent:
                logits, values, _ = self.model.predict_from_states(root_state)
            else:
                logits, values, _ = self.model.predict_policy_value(root_state.unsqueeze(0))
        last_logits = logits[0, -1]
        
        # values is list of tensors, convert to list of floats
        value_list = [v.item() for v in values]
        
        self.expand_node(root, last_logits, value_list)

        # Run simulations
        for _ in range(self.n_simulations):
            leaf, path = self.select_action(root)
            
            # Reconstruct state for the leaf node (O(depth) compute, O(1) memory)
            leaf_state = self._reconstruct_state(leaf, root_state)

            if not leaf.expanded():
                with torch.no_grad():
                    if use_latent:
                        logits, values, _ = self.model.predict_from_states(leaf_state)
                    else:
                        logits, values, _ = self.model.predict_policy_value(leaf_state.unsqueeze(0))
                last_logits = logits[0, -1]
                value_list = [v.item() for v in values]
                
                self.expand_node(leaf, last_logits, value_list)
                rollout_value = value_list
            else:
                rollout_value = self._rollout_value(leaf_state)

            self.backpropagate(leaf, rollout_value, path)

        # Select best action
        actions = list(root.children.keys())
        if not actions:
            return 0, None, 0.0

        visit_counts = [root.children[a].visit_count for a in actions]
        best = actions[visit_counts.index(max(visit_counts))]
        
        # Get confidence from the best child's value estimate
        best_child = root.children[best]
        # Return list of confidences if available
        if best_child.layer_value_sums:
            confidence = [s / best_child.visit_count for s in best_child.layer_value_sums]
        else:
            confidence = best_child.value # Fallback scalar

        delta = self._compute_delta_from_mcts(root, best)
        return int(best), delta, confidence
    
    def _compute_delta_from_mcts(self, root: MCTSNode, best_action: int) -> Optional[List[torch.Tensor]]:
        """
        Compute the update target from MCTS.
        """
        if best_action not in root.children:
            return None
        
        best_child = root.children[best_action]
        
        # Memory Wall Fix: Reconstruct states on demand
        # Root must have state_cache
        if root.state_cache is None:
            return None
            
        root_state = root.state_cache
        child_state = self._reconstruct_state(best_child, root_state)
        
        if not isinstance(root_state, list) or not isinstance(child_state, list):
            return None
        
        try:
            # Return the best child's state directly
            targets = []
            for child_layer_state in child_state:
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

    def _rollout_value(self, state: List[torch.Tensor]) -> List[float]:
        """Improved rollout with optional sampling."""
        # state passed in is already reconstructed
        device = next(self.model.parameters()).device
        
        try:
            # Check if it's a token seq (tensor) or latent state (list)
            is_token_seq = isinstance(state, torch.Tensor) and state.dtype in (torch.long, torch.int)
        except Exception:
            is_token_seq = False

        if is_token_seq:
            for _ in range(self.rollout_depth):
                with torch.no_grad():
                    logits, values, _ = self.model.predict_policy_value(state.unsqueeze(0))
                last_logits = logits[0, -1]
                
                if self.use_sampling:
                    probs = F.softmax(last_logits / self.temperature, dim=-1)
                    action = torch.multinomial(probs, 1).item()
                else:
                    action = int(torch.argmax(last_logits).item())
                
                state = self._get_next_state(state, action)
            
            with torch.no_grad():
                _, values, _ = self.model.predict_policy_value(state.unsqueeze(0))
            return [v.item() for v in values]
        else:
            if isinstance(state, list):
                with torch.no_grad():
                    _, values, _ = self.model.predict_from_states(state)
                return [v.item() for v in values]
            else:
                dummy = torch.zeros((1, 1), dtype=torch.long, device=device)
                with torch.no_grad():
                    _, values, _ = self.model.predict_policy_value(dummy)
                return [v.item() for v in values]