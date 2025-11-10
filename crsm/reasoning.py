"""
MCTS (Monte Carlo Tree Search) based reasoning module for CRSM.
This implements the asynchronous tree search deliberation component.
"""

import torch
import torch.nn as nn
import math
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
                
    def _get_next_state(self, state: torch.Tensor, action: int) -> torch.Tensor:
        """Simulate next state given current state and action"""
        # For now, simply append action to state sequence
        # This should be enhanced based on specific state representation
        return torch.cat([state, torch.tensor([action], device=state.device)])
    
    def backpropagate(self, node: MCTSNode, value: float, path: List[int]):
        """Update statistics of visited nodes in the path"""
        cur = node
        while cur is not None:
            cur.visit_count += 1
            cur.value_sum += float(value)
            cur = cur.parent
            
    async def deliberate(self, state: torch.Tensor) -> int:
        """
        Asynchronous deliberation process using MCTS
        Args:
            state: Current state tensor
        Returns:
            best_action: Selected action after deliberation
        """
        # Normalize state shape to 1-D sequence stored in nodes
        if state.dim() == 1:
            root_state = state
        elif state.dim() == 2 and state.size(0) == 1:
            root_state = state.squeeze(0)
        elif state.dim() == 2:
            root_state = state[0]
        else:
            root_state = state.view(-1)

        root = MCTSNode(state=root_state, prior_p=1.0, children={}, parent=None)

        # Initial expansion of root node using policy/value head
        with torch.no_grad():
            logits, value, _ = self.model.predict_policy_value(root.state.unsqueeze(0))
        last_logits = logits[0, -1]
        self.expand_node(root, last_logits, float(value.item()) if hasattr(value, 'item') else float(value))
        
        # Run simulations
        for _ in range(self.n_simulations):
            leaf, path = self.select_action(root)

            # If leaf is not expanded, expand it using model's policy/value
            if not leaf.expanded():
                with torch.no_grad():
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
            return 0  # Default action if no children
        
        visit_counts = [root.children[a].visit_count for a in actions]
        return actions[visit_counts.index(max(visit_counts))]

    def _rollout_value(self, node: MCTSNode) -> float:
        """Perform a short greedy rollout from node and return value estimate."""
        # greedy rollout for rollout_depth steps
        state = node.state
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