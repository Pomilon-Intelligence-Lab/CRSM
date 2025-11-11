"""
MCTS (Monte Carlo Tree Search) based reasoning module for CRSM.
This implements the asynchronous tree search deliberation component.
"""

import torch
import torch.nn as nn
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
        # how many top actions to expand per node (top-k)
        self.top_k = 16
        # rollout depth when leaf has no value
        self.rollout_depth = 5
        # Dynamics model for fast state transitions during MCTS
        self.dynamics_model = None
        
    def select_action(self, node: MCTSNode) -> Tuple[MCTSNode, List[int]]:
        """Select the most promising action using PUCT algorithm"""
        path = []
        
        while node.expanded():
            max_ucb = float('-inf')
            best_action = -1
            best_child = None
            
            # Calculate UCB for all children
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
        # Expand only top-k probable actions for efficiency
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
        """Simulate next state given current state and action.
        
        Prioritizes the lightweight LatentDynamics model for fast MCTS rollouts.
        """
        device = next(self.model.parameters()).device

        # --- CRSM FAST DYNAMICS MODEL STEP (HIGH PRIORITY) ---
        # Check for the lightweight dynamics model, which is attached to the reasoning loop
        if isinstance(state, list) and hasattr(self, 'dynamics_model') and self.dynamics_model is not None:
            try:
                # 1. Get action embedding from the main model's embedding layer
                token = torch.tensor([[action]], dtype=torch.long, device=device)
                
                # CRSM (self.model) is the wrapper; action embedding must come from the backbone.
                if hasattr(self.model, 'backbone') and hasattr(self.model.backbone, 'embedding'):
                     # token -> (1, 1, d_model) -> squeeze to (d_model,)
                     action_emb = self.model.backbone.embedding(token).squeeze(0).squeeze(0)
                else:
                    # Fallback if the embedding is directly on the model
                    action_emb = self.model.embedding(token).squeeze(0).squeeze(0)

            except Exception as e:
                # If we fail to get the embedding (e.g., structure mismatch), fall through to the slow SSM step
                pass 
            else:
                # Execute FAST dynamics prediction
                next_states = []
                for layer_state in state:
                    if layer_state is None:
                        next_states.append(None)
                        continue
                    
                    # Ensure state input is correctly shaped (d_model) for dynamics model
                    s_input = layer_state.squeeze(0)
                    
                    # Predict state delta using the lightweight Dynamics Model
                    delta = self.dynamics_model(s_input, action_emb)
                    
                    # Apply delta (update rule) and keep the batch dimension
                    next_layer_state = layer_state + delta.unsqueeze(0)
                    next_states.append(next_layer_state)
                
                return next_states

        # --- SLOW FALLBACK: Original Mamba step (for MCTS verification/initialization or if dynamics fails) ---
        if isinstance(state, list):
            # latent per-layer state (list) -> use model.step (SLOW)
            token = torch.tensor([[action]], dtype=torch.long, device=device)
            # model.step returns (logits, new_states).
            _, new_states = self.model.step(token, state)
            return new_states

        # token sequence -> append action token
        if isinstance(state, torch.Tensor):
            return torch.cat([state, torch.tensor([action], device=state.device)])

        # fallback: try to coerce
        try:
            return torch.cat([state, torch.tensor([action], device=state.device)])
        except Exception:
            # as a last resort return the action as a tensor
            return torch.tensor([action], device=device)
    
    def backpropagate(self, node: MCTSNode, value: float, path: List[int]):
        """Update statistics of visited nodes in the path"""
        cur = node
        while cur is not None:
            cur.visit_count += 1
            cur.value_sum += float(value)
            cur = cur.parent
            
    def deliberate_sync(self, seq: Optional[torch.Tensor], state: Optional[torch.Tensor]) -> Tuple[int, Optional[torch.Tensor]]:
        """
        Blocking deliberation process using MCTS. Accepts an optional token
        sequence (`seq`) and an optional canonical latent `state` (from SSM).

        This function is CPU/GPU intensive and intended to be executed in a
        background thread (e.g., via asyncio.to_thread).

        Args:
            seq: optional token sequence tensor (batch or 1D)
            state: optional latent state (1D tensor or list-derived tensor)
        Returns:
            (best_action, state_delta): Selected action and optional state delta
        """
        # Prefer using explicit latent state when provided
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
                # non-tensor state (e.g., list) -> keep as-is (per-layer states)
                root_state = state
        elif seq is not None:
            # Fallback: derive root state from token sequence
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

        # Initial expansion of root node using policy/value head
        use_latent = state is not None
        device = next(self.model.parameters()).device
        with torch.no_grad():
            if use_latent:
                # evaluate from provided latent state using the state-only head
                logits, value, _ = self.model.predict_from_states(state)
            else:
                logits, value, _ = self.model.predict_policy_value(root.state.unsqueeze(0))
        last_logits = logits[0, -1]
        self.expand_node(root, last_logits, float(value.item()) if hasattr(value, 'item') else float(value))

        # Run simulations
        for _ in range(self.n_simulations):
            leaf, path = self.select_action(root)

            # If leaf is not expanded, expand it using model's policy/value
            if not leaf.expanded():
                with torch.no_grad():
                    if use_latent:
                        # evaluate from the canonical latent state using the state-only head
                        logits, value, _ = self.model.predict_from_states(state)
                    else:
                        logits, value, _ = self.model.predict_policy_value(leaf.state.unsqueeze(0))
                last_logits = logits[0, -1]
                self.expand_node(leaf, last_logits, float(value.item()) if hasattr(value, 'item') else float(value))
                rollout_value = float(value.item()) if hasattr(value, 'item') else float(value)
            else:
                # If already expanded, run a quick greedy rollout to estimate value
                rollout_value = self._rollout_value(leaf)

            # backpropagate the obtained value
            self.backpropagate(leaf, rollout_value, path)

        # Select action with highest visit count
        actions = list(root.children.keys())
        if not actions:
            return 0, None  # Default action if no children

        visit_counts = [root.children[a].visit_count for a in actions]
        best = actions[visit_counts.index(max(visit_counts))]

        delta = self._compute_delta_from_mcts(root, best)
        return int(best), delta
    
    def _compute_delta_from_mcts(self, root: MCTSNode, best_action: int) -> Optional[List[torch.Tensor]]:
        """
        Compute state delta based on MCTS statistics.
        
        The delta represents how planning suggests the latent state should be adjusted
        based on the tree search. We weight child state differences by visit counts
        to capture which trajectories the search found most promising.
        
        Args:
            root: The MCTS root node
            best_action: The selected action
            
        Returns:
            List of per-layer state deltas, or None if not applicable
        """
        if best_action not in root.children:
            return None
        
        best_child = root.children[best_action]
        
        # Only compute deltas for list-based states (per-layer SSM states)
        if not isinstance(root.state, list) or not isinstance(best_child.state, list):
            return None
        
        # Can't compute delta if structures don't match
        if len(root.state) != len(best_child.state):
            return None
        
        try:
            deltas = []
            # Weight by how much the search explored this path
            # Higher visit count = more confident in this direction
            weight = min(1.0, best_child.visit_count / (root.visit_count + 1e-8))
            
            for parent_layer_state, child_layer_state in zip(root.state, best_child.state):
                # Skip if either is None
                if parent_layer_state is None or child_layer_state is None:
                    deltas.append(None)
                    continue
                
                # Ensure both are tensors
                if not isinstance(parent_layer_state, torch.Tensor) or not isinstance(child_layer_state, torch.Tensor):
                    deltas.append(None)
                    continue
                
                # Compute weighted difference
                # This represents "planning suggests moving the state in this direction"
                delta = (child_layer_state - parent_layer_state) * weight
                
                # Scale down to prevent large jumps (conservative update)
                delta = delta * 0.1
                
                deltas.append(delta)
            
            return deltas
        
        except Exception as e:
            # If anything fails, return None (no delta applied)
            return None


    def _aggregate_mcts_statistics(self, root: MCTSNode) -> Optional[List[torch.Tensor]]:
        """
        Alternative delta computation: aggregate information from all visited nodes.
        
        This is more sophisticated - instead of just using the best child, we create
        a delta that incorporates information from the entire search tree.
        
        You can use this instead of _compute_delta_from_mcts for potentially better results.
        """
        if not isinstance(root.state, list):
            return None
        
        if not root.children:
            return None
        
        try:
            # Collect all children states weighted by visit counts
            total_visits = sum(child.visit_count for child in root.children.values())
            
            if total_visits == 0:
                return None
            
            # Initialize accumulated deltas
            num_layers = len(root.state)
            accumulated_deltas = [None] * num_layers
            
            for action, child in root.children.items():
                if not isinstance(child.state, list) or len(child.state) != num_layers:
                    continue
                
                # Weight by visit count (exploration weight)
                weight = child.visit_count / total_visits
                
                # Also weight by value (exploitation weight)
                value_weight = max(0, child.value) if child.visit_count > 0 else 0
                combined_weight = weight * (1 + value_weight)
                
                for i, (parent_s, child_s) in enumerate(zip(root.state, child.state)):
                    if parent_s is None or child_s is None:
                        continue
                    if not isinstance(parent_s, torch.Tensor) or not isinstance(child_s, torch.Tensor):
                        continue
                    
                    delta = (child_s - parent_s) * combined_weight
                    
                    if accumulated_deltas[i] is None:
                        accumulated_deltas[i] = delta
                    else:
                        accumulated_deltas[i] = accumulated_deltas[i] + delta
            
            # Scale down for conservative updates
            accumulated_deltas = [d * 0.05 if d is not None else None for d in accumulated_deltas]
            
            return accumulated_deltas
        
        except Exception:
            return None

    async def deliberate(self, seq: Optional[torch.Tensor], state: Optional[torch.Tensor]) -> Tuple[int, Optional[torch.Tensor]]:
        """Async wrapper that runs the synchronous deliberation off the event loop.

        Args:
            seq: optional token sequence
            state: optional latent state
        """
        # Run blocking deliberation in a thread pool
        return await asyncio.to_thread(self.deliberate_sync, seq, state)

    def _rollout_value(self, node: MCTSNode) -> float:
        """Perform a short greedy rollout from node and return value estimate."""
        # greedy rollout for rollout_depth steps
        state = node.state
        device = next(self.model.parameters()).device
        # If the node.state looks like a token sequence (LongTensor), roll out on tokens.
        try:
            is_token_seq = state.dtype in (torch.long, torch.int)
        except Exception:
            is_token_seq = False

        if is_token_seq:
            for _ in range(self.rollout_depth):
                with torch.no_grad():
                    logits, value, _ = self.model.predict_policy_value(state.unsqueeze(0))
                last_logits = logits[0, -1]
                action = int(torch.argmax(last_logits).item())
                state = self._get_next_state(state, action)
            # final value estimate
            with torch.no_grad():
                _, value, _ = self.model.predict_policy_value(state.unsqueeze(0))
            return float(value.item()) if hasattr(value, 'item') else float(value)
        else:
            # Non-token latent state: if we have per-layer states use the state-only head
            if isinstance(state, list):
                with torch.no_grad():
                    _, value, _ = self.model.predict_from_states(state)
                return float(value.item()) if hasattr(value, 'item') else float(value)
            else:
                dummy = torch.zeros((1, 1), dtype=torch.long, device=device)
                with torch.no_grad():
                    _, value, _ = self.model.predict_policy_value(dummy)
                return float(value.item()) if hasattr(value, 'item') else float(value)