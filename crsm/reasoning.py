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

        If `state` is a list of per-layer SSM states, call the backbone `step`
        with the action token to obtain the new per-layer states. If `state`
        is a token sequence tensor, append the action to the sequence.
        """
        # latent per-layer state (list) -> use backbone.step to simulate transition
        if isinstance(state, list):
            device = next(self.model.parameters()).device
            token = torch.tensor([[action]], dtype=torch.long, device=device)
            # model.step returns (logits, new_states)
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
            return torch.tensor([action], device=next(self.model.parameters()).device)
    
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

        # For now we don't propose a state delta; return None for state_delta.
        return int(best), None

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