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

def parse_arc_grids(tokens: List[int]) -> List[List[List[int]]]:
    """Extracts all complete grids from a sequence of tokens."""
    grids = []
    i = 0
    while i < len(tokens):
        # ARC Tokens: 13=ROW_END, 14=GRID_END, 16-45=Rows, 46-75=Cols
        if 16 <= tokens[i] <= 45: # Row header
            rows = tokens[i] - 15
            if i + 1 < len(tokens) and 46 <= tokens[i+1] <= 75: # Col header
                cols = tokens[i+1] - 45
                grid = []
                curr_row = []
                idx = i + 2
                while idx < len(tokens) and len(grid) < rows:
                    t = tokens[idx]
                    if 0 <= t <= 9:
                        curr_row.append(t)
                    elif t == 13: # ROW_END
                        if len(curr_row) < cols:
                            curr_row.extend([0] * (cols - len(curr_row)))
                        grid.append(curr_row[:cols])
                        curr_row = []
                    elif t == 14: # GRID_END
                        break
                    idx += 1
                if len(grid) > 0:
                    grids.append(grid)
                i = idx
            else:
                i += 1
        else:
            i += 1
    return grids

class TokenLevelStructuralEvaluator:
    """
    Strictly validates the ARC token protocol during MCTS generation.
    Forces the model to respect:
    1. Dimension tokens (16-45 for rows, 46-75 for cols)
    2. [ROW_END] (13) at correct intervals
    3. [GRID_END] (14) after correct row count
    """
    def __call__(self, path: List[int]) -> float:
        if not path: return 0.5
        
        # We only care about the sequence starting from dimension headers
        # Search for first dimension token (16-75)
        start_idx = -1
        for i, t in enumerate(path):
            if 16 <= t <= 75:
                start_idx = i
                break
        
        if start_idx == -1: return 0.5 # Too early to judge
        
        relevant = path[start_idx:]
        if len(relevant) < 2: return 0.6 # Good start
        
        rows = relevant[0] - 15
        cols = relevant[1] - 45
        
        if not (1 <= rows <= 30 and 1 <= cols <= 30):
            return 0.1 # Invalid dimensions
            
        # Validate grid content
        grid_content = relevant[2:]
        if not grid_content: return 0.7
        
        curr_row_len = 0
        row_count = 0
        
        for i, t in enumerate(grid_content):
            if t == 13: # [ROW_END]
                if curr_row_len != cols:
                    return 0.0 # WRONG COLUMN COUNT
                curr_row_len = 0
                row_count += 1
                # CRITICAL: If we already have enough rows, the NEXT token MUST be 14 or we fail
                continue 
            elif t == 14: # [GRID_END]
                if row_count != rows:
                    return 0.0 # WRONG ROW COUNT
                return 1.0 # Perfect termination!
            
            # If we are past the row count and still seeing tokens that aren't 14
            if row_count == rows and t != 14:
                return 0.0 # Kill path that won't end
            elif 0 <= t <= 9: # Color
                curr_row_len += 1
                if curr_row_len > cols:
                    return 0.0 # Row too long
            else:
                # Unexpected token in grid (like high index noise)
                # Penalize but don't KILL (allow some search noise)
                return 0.1 
                
        # CRITICAL FIX: If we just finished a row's colors, the NEXT token MUST be 13.
        # This is where the model fails (predicts 0 instead of 13).
        if curr_row_len == cols and len(grid_content) > 0 and grid_content[-1] != 13:
             # We just added the last color, the next MCTS step MUST find 13.
             # If we are AT the limit, we give a high structural score,
             # but the NEXT token evaluation will fail if it's not 13.
             return 0.9 
        
        # If we reached column limit but the LAST token isn't 13, and we have MORE tokens, it's an error
        # (This is handled by the loop above if t is a color)

        return 0.8 # So far so good

class ARCContextEvaluator:
    """
    Analyzes ARC prompt context to provide objective rewards for MCTS paths.
    Anchors search to the logic implied by training examples.
    """
    def __init__(self, prompt_tokens: List[int]):
        self.prompt_grids = parse_arc_grids(prompt_tokens)
        # Typically: [I1, O1, I2, O2, ..., Itest]
        # We assume pairs except the last one which is the test input
        self.train_pairs = []
        if len(self.prompt_grids) >= 2:
            for i in range(0, len(self.prompt_grids) - 1, 2):
                if i + 1 < len(self.prompt_grids):
                    self.train_pairs.append((self.prompt_grids[i], self.prompt_grids[i+1]))
        
        self.test_input = self.prompt_grids[-1] if self.prompt_grids else None
        
        # Analyze dimensions
        self.dim_changes = []
        for inp, out in self.train_pairs:
            self.dim_changes.append((len(inp), len(inp[0]), len(out), len(out[0])))
            
        # Analyze if it's an 'Identity' task (The common bias)
        self.is_identity = all(inp == out for inp, out in self.train_pairs) if self.train_pairs else False
        
    def __call__(self, path: List[int]) -> float:
        """
        Combined evaluator: Structural protocol + Logical consistency.
        Returns 0.0 for invalid structure, otherwise logical score [0, 1].
        """
        # 0. Structural Protocol Check (High Priority)
        struct_eval = TokenLevelStructuralEvaluator()
        struct_score = struct_eval(path)
        if struct_score == 0.0:
            return 0.0 # Strict failure
            
        # Parse the grid being generated in imagination
        imagined_grids = parse_arc_grids(path)
        if not imagined_grids:
            return struct_score * 0.5 # Partial reward for correct structure
            
        cand_out = imagined_grids[0]
        r_cand, c_cand = len(cand_out), len(cand_out[0])
        
        reward = 0.5 # Neutral start
        
        # 1. Structural Verification: Dimension Check
        # If all train examples transform dimensions in a specific way, cand_out should too.
        if self.dim_changes and self.test_input:
            r_in, c_in = len(self.test_input), len(self.test_input[0])
            # Check for constant dimension ratio or fixed size
            all_fixed_size = all(d[2] == r_cand and d[3] == c_cand for d in self.dim_changes)
            all_fixed_ratio = all(d[2]/d[0] == r_cand/r_in and d[3]/d[1] == c_cand/c_in for d in self.dim_changes)
            
            if not (all_fixed_size or all_fixed_ratio):
                reward -= 0.2 # Penalty for structural inconsistency
            else:
                reward += 0.1
                
        # 2. Logical Discovery: Break Identity Bias
        # If we KNOW it's not identity, but the model produces identity, penalize heavily.
        if not self.is_identity and self.test_input and cand_out == self.test_input:
            reward -= 0.4 # Strong anti-bias anchor
            
        # 3. Color Palette Consistency
        if self.train_pairs:
            train_out_colors = set()
            for _, out in self.train_pairs:
                for row in out: train_out_colors.update(row)
            
            cand_colors = set()
            for row in cand_out: cand_colors.update(row)
            
            # If candidate uses colors NEVER seen in any output, be suspicious
            unseen_colors = cand_colors - train_out_colors
            if unseen_colors:
                reward -= 0.1 * len(unseen_colors)
                
        return max(0.0, min(1.0, reward))

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
    uncertainty_lambda: float = 0.1 # Default penalty
    use_surprise_reward: bool = False
    external_reward: Optional[float] = None # NEW: CTA Reward [0, 1]
    
    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        
        # If external evaluator (CTA) flagged this as impossible/invalid
        if self.external_reward is not None and self.external_reward == 0.0:
            return -1.0 # Kill this path
        
        # Base value from average (or consensus)
        if self.layer_value_sums:
            layer_means = [s / self.visit_count for s in self.layer_value_sums]
            v_tensor = torch.tensor(layer_means)
            mean_v = v_tensor.mean().item()
            std_v = v_tensor.std().item() if v_tensor.numel() > 1 else 0.0
            val = mean_v - self.uncertainty_lambda * std_v
        else:
            val = self.value_sum / self.visit_count

        # Combine with external reward if present
        if self.external_reward is not None:
            # External reward acts as a strong multiplier or bias
            val = 0.5 * val + 0.5 * (self.external_reward * 2.0 - 1.0)

        # Hypothesis S: Surprise Reward
        if self.use_surprise_reward and self.prior_p > 0:
            surprise = -math.log(self.prior_p + 1e-8)
            val = val * (1.0 + 0.1 * surprise)
            
        return val
    
    def expanded(self) -> bool:
        return len(self.children) > 0

class AsyncDeliberationLoop:
    def __init__(self, mamba_model, c_puct=1.0, n_simulations=50, uncertainty_lambda=0.1, use_surprise_reward=False):
        """
        Args:
            mamba_model: The underlying Mamba model for state processing
            c_puct: Exploration constant for PUCT algorithm
            n_simulations: Number of MCTS simulations per deliberation
            uncertainty_lambda: Penalty/Bonus factor for layer variance
            use_surprise_reward: Boost low-prior high-value nodes
        """
        self.model = mamba_model
        self.c_puct = c_puct
        self.n_simulations = n_simulations
        self.uncertainty_lambda = uncertainty_lambda
        self.use_surprise_reward = use_surprise_reward
        self.path_evaluator = None # Optional function: (path) -> float reward
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
        """Select the most promising action using Adaptive PUCT algorithm"""
        path = []
        
        while node.expanded():
            max_ucb = float('-inf')
            best_action = -1
            best_child = None
            
            # Adaptive PUCT: Increase exploration if the node is over-confident
            # We use the entropy of the priors as a proxy for confidence.
            priors = torch.tensor([child.prior_p for child in node.children.values()])
            entropy = -torch.sum(priors * torch.log(priors + 1e-8)).item()
            # Max entropy for top_k=16 is log(16) approx 2.77
            # If entropy is low (e.g. < 1.0), we are in a high-confidence regime.
            # We boost C_puct to force exploration.
            adaptive_c = self.c_puct
            if entropy < 1.0:
                # Boost C up to 2x as entropy approaches 0
                adaptive_c *= (1.0 + (1.0 - entropy))

            for action, child in node.children.items():
                if child.visit_count > 0:
                    q_value = child.value
                    u_value = (adaptive_c * child.prior_p * 
                             math.sqrt(node.visit_count) / (1 + child.visit_count))
                    ucb = q_value + u_value
                else:
                    # Unvisited nodes get a boost from the adaptive C too
                    ucb = float('inf')
                    
                if ucb > max_ucb:
                    max_ucb = ucb
                    best_action = action
                    best_child = child
                    
            path.append(best_action)
            node = best_child
            
        return node, path

    def _apply_structural_prior(self, logits: torch.Tensor, path: List[int]) -> torch.Tensor:
        """
        Adjusts backbone logits based on the ARC protocol.
        """
        if self.path_evaluator is None or not isinstance(self.path_evaluator, ARCContextEvaluator):
            return logits
            
        # Find where the CURRENT output grid starts (after the last 12)
        output_start_idx = -1
        for i in range(len(path)-1, -1, -1):
            if path[i] == 12: # [OUTPUT_START]
                output_start_idx = i
                break
                
        if output_start_idx == -1:
            return logits
            
        current_output_path = path[output_start_idx + 1:]
        # print(f"  [Prior] Path: {current_output_path}")
        
        # Scenario A: Need to emit Rows
        if len(current_output_path) == 0:
            if self.path_evaluator.dim_changes:
                expected_rows = self.path_evaluator.dim_changes[0][2] + 15
                logits = logits.clone()
                logits[0, :] = -100.0
                logits[0, expected_rows] = 20.0
                logger.debug(f"  [Prior] Scenario A -> Force {expected_rows}")
            return logits
            
        # Scenario B: Need to emit Cols
        if len(current_output_path) == 1:
            if self.path_evaluator.dim_changes:
                expected_cols = self.path_evaluator.dim_changes[0][3] + 45
                logits = logits.clone()
                logits[0, :] = -100.0
                logits[0, expected_cols] = 20.0
                logger.debug(f"  [Prior] Scenario B -> Force {expected_cols}")
            return logits
            
        # Scenario C: In the grid, boost colors and structural tokens
        valid_colors = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 14}
        mask = torch.full_like(logits, -20.0) 
        for c in valid_colors:
            mask[0, c] = 0.0
            
        # Scenario D: FORCE termination if row_count is reached
        grid_content = current_output_path[2:]
        row_count = 0
        for t in grid_content:
            if t == 13: row_count += 1
            
        expected_rows = self.path_evaluator.dim_changes[0][2]
        logits = logits.clone()
        
        if row_count == expected_rows:
            # We finished the rows, GRID_END is REQUIRED
            logits[0, :] = -100.0
            logits[0, 14] = 20.0
        elif row_count < expected_rows:
            # Prevent GRID_END until rows are done
            logits[0, 14] = -100.0
        
        return logits + mask

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
    
    def expand_node(self, node: MCTSNode, logits: torch.Tensor, value: float, path: List[int]):
        """Expand a leaf node using model predictions with Structural Prior Injection"""
        
        # Get the full sequence by prepending the current base sequence
        full_path = (self._current_seq_list if hasattr(self, '_current_seq_list') else []) + path
        
        # Injection: Bias the priors towards correct ARC structure
        biased_logits = self._apply_structural_prior(logits.unsqueeze(0), full_path).squeeze(0)
        
        probs = torch.softmax(biased_logits, dim=-1)
        topk = min(self.top_k, biased_logits.size(-1))
        topv, topi = torch.topk(probs, topk)
        for p, a in zip(topv.tolist(), topi.tolist()):
            if p <= 0:
                continue
            
            child = MCTSNode(
                prior_p=float(p),
                children={},
                parent=node,
                action=int(a),
                state_cache=None, # Lazy
                uncertainty_lambda=self.uncertainty_lambda,
                use_surprise_reward=self.use_surprise_reward
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
        """Update statistics of visited nodes with Structural & Objective Verification"""
        
        # 1. Objective Verification (DTR / CTA)
        if self.path_evaluator is not None:
            external_reward = self.path_evaluator(path)
            
            # Store in leaf node for immediate value property influence
            node.external_reward = external_reward
            
            if external_reward == 0.0:
                # INVALID STRUCTURE: Force value to 0.0 (Strict pruning)
                value = [0.0] * len(value) if isinstance(value, list) else 0.0
            elif external_reward is not None:
                # Blend internal value with external reward
                if isinstance(value, list):
                    value = [(v + external_reward) / 2.0 for v in value]
                else:
                    value = (value + external_reward) / 2.0

        # Update node statistics up the tree
        cur = node
        while cur is not None:
            cur.visit_count += 1
            if isinstance(value, list):
                if cur.layer_value_sums is None:
                    cur.layer_value_sums = [0.0] * len(value)
                for i, v in enumerate(value):
                    cur.layer_value_sums[i] += v
            else:
                cur.value_sum += float(value)
            cur = cur.parent
            
    def deliberate_sync(self, seq: Optional[torch.Tensor], state: Optional[torch.Tensor]) -> Tuple[int, Optional[torch.Tensor], float]:
        """Blocking MCTS deliberation process."""
        # Save sequence for full path reconstruction during MCTS
        if seq is not None:
            self._current_seq_list = seq[0].tolist() if seq.dim() > 1 else seq.tolist()
        else:
            self._current_seq_list = []
            
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
        root = MCTSNode(
            prior_p=1.0, 
            children={}, 
            parent=None, 
            action=None, 
            state_cache=root_state,
            uncertainty_lambda=self.uncertainty_lambda,
            use_surprise_reward=self.use_surprise_reward
        )

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
        
        self.expand_node(root, last_logits, value_list, [])

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
                
                self.expand_node(leaf, last_logits, value_list, path)
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
