import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from .logger import logger

class RLTrainer:
    """
    Reinforcement Learning Trainer for Reasoning (GRPO-style).
    Optimizes the model to generate correct solutions by sampling multiple attempts
    and reinforcing the ones that succeed relative to the group average.
    """
    def __init__(self, model, optimizer, config):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.model.to(self.device)
        
        # Hyperparams
        self.num_generations = config.get('num_generations', 4) # K attempts per prompt
        self.beta_kl = config.get('beta_kl', 0.01) # KL penalty to stay close to SFT model
        self.clip_eps = 0.2
        
        # Reference model (frozen SFT copy) for KL constraint
        # Ideally we clone, but for memory efficiency we might skip strict KL or use approx.
        self.ref_model = None 

    def fit(self, task, epochs, checkpoint_dir='checkpoints'):
        train_loader, _ = task.get_dataloaders(self.config.get('batch_size', 1))
        entropy_coeff = 0.01 
        
        # Handle CRSMModel wrapper
        inner_model = self.model.crsm if hasattr(self.model, 'crsm') else self.model

        for epoch in range(1, epochs + 1):
            total_reward = 0
            total_loss = 0
            steps = 0
            
            self.model.train()
            
            for batch in train_loader:
                x, y, split_indices = batch
                
                # For each item in batch
                for i in range(x.size(0)):
                    split_idx = split_indices[i].item()
                    prompt = x[i, :split_idx].unsqueeze(0).to(self.device)
                    
                    # Extract ground truth robustly
                    raw_target = y[i, split_idx-1:].tolist()
                    gt_tokens = []
                    for t in raw_target:
                        gt_tokens.append(t)
                        if t == 14: break
                    
                    # 1. Fast Generation (K times)
                    completions = []
                    rewards = []
                    
                    for _ in range(self.num_generations):
                        with torch.no_grad():
                            # Generate completion only (no prompt echo)
                            gen_tokens = self._generate_fast(inner_model, prompt, max_len=len(gt_tokens)+10, temp=1.2)
                        
                        # Full sequence for training: [Prompt, Gen]
                        full_seq = torch.cat([prompt, gen_tokens], dim=1)
                        completions.append(full_seq)
                        
                        # Verify
                        pred_tokens = gen_tokens[0].tolist()
                        
                        # Reward Shaping
                        if pred_tokens[:len(gt_tokens)] == gt_tokens:
                            r = 1.0
                        else:
                            if 14 in pred_tokens:
                                p_trunc = pred_tokens[:len(gt_tokens)]
                                matches = sum(1 for p, g in zip(p_trunc, gt_tokens) if p == g)
                                accuracy = matches / max(1, len(gt_tokens))
                                r = 0.1 + (0.4 * accuracy)
                            else:
                                r = 0.0
                        rewards.append(r)
                    
                    # 2. Compute Advantages
                    rewards_t = torch.tensor(rewards, device=self.device)
                    mean_r = rewards_t.mean()
                    std_r = rewards_t.std() + 1e-8
                    advantages = (rewards_t - mean_r) / std_r
                    
                    # 3. Parallel Gradient Computation
                    # We process all K completions in one go if batch size permits, or loop.
                    self.optimizer.zero_grad()
                    policy_loss = 0
                    entropy_loss = 0
                    value_loss_total = 0
                    
                    for k, full_seq in enumerate(completions):
                        # Run forward pass on the FULL generated sequence (like SFT)
                        # We use predict_policy_value to get Value Head outputs
                        logits, values, _ = inner_model.backbone.predict_policy_value(full_seq)
                        
                        # We only care about the log_probs of the GENERATED tokens
                        # Inputs: full_seq[:, :-1]
                        # Targets: full_seq[:, 1:]
                        # Generated part starts at split_idx
                        
                        # Logits for generation steps: logits[:, split_idx-1 : -1] 
                        # corresponding to predictions for full_seq[:, split_idx:]
                        gen_logits = logits[:, split_idx-1:-1, :]
                        gen_targets = full_seq[:, split_idx:]
                        
                        dist = Categorical(logits=gen_logits / 1.2) # Match temp
                        log_probs = dist.log_prob(gen_targets)
                        entropy = dist.entropy()
                        
                        traj_log_prob = log_probs.sum()
                        traj_entropy = entropy.mean()
                        
                        policy_loss += -advantages[k] * traj_log_prob
                        entropy_loss += -traj_entropy
                        
                        # =========================================================
                        # VALUE HEAD TRAINING (Sparse Reward Signal)
                        # =========================================================
                        # We use a strict binary reward for the Value Head to force
                        # System 2 to focus ONLY on correctness, not just syntax.
                        # reward_value = 1.0 if EXACT match, else 0.0
                        # Check correctness using the rewards[k] computed earlier.
                        # Since rewards[k] includes partial credit, we reconstruct strict logic.
                        
                        is_exact_correct = (rewards[k] >= 1.0)
                        target_val = 1.0 if is_exact_correct else 0.0
                        target_val_tensor = torch.tensor(target_val, device=self.device)
                        
                        # values is List[Tensor] (one per layer). 
                        # Each tensor is (batch=1,) or scalar.
                        # We want the value predictions for the generated trajectory.
                        # predict_policy_value gives values for the WHOLE sequence?
                        # It calls _compute_layer_values(new_states).
                        # new_states corresponds to the LAST token's state.
                        # So 'values' represents V(terminal_state).
                        
                        # We regress V(terminal_state) -> Actual Outcome
                        v_loss = sum(F.mse_loss(v.squeeze(), target_val_tensor) for v in values)
                        value_loss_total += v_loss

                    loss = (policy_loss / self.num_generations) + \
                           (entropy_coeff * (entropy_loss / self.num_generations)) + \
                           (0.5 * (value_loss_total / self.num_generations))
                    
                    loss.backward()
                    self.optimizer.step()
                    
                    total_reward += mean_r.item()
                    total_loss += loss.item()
                    steps += 1
            
            avg_reward = total_reward / max(1, steps)
            logger.info(f"RL Epoch {epoch}/{epochs} | Avg Reward: {avg_reward:.4f} | Loss: {total_loss/steps:.4f}")
            
            if epoch % 5 == 0:
                torch.save(self.model.state_dict(), f"{checkpoint_dir}/rl_checkpoint_{epoch}.pt")

    def _generate_fast(self, model, prompt, max_len, temp=1.2):
        """
        Fast autoregressive generation using state caching.
        Returns ONLY the generated tokens (not prompt).
        """
        # Prefill
        logits, states = model.backbone(prompt)
        next_logits = logits[:, -1, :]
        
        gen_tokens = []
        curr_token = torch.multinomial(F.softmax(next_logits / temp, dim=-1), 1)
        gen_tokens.append(curr_token)
        
        for _ in range(max_len):
            if curr_token.item() == 14: break
            
            logits, states = model.backbone.step(curr_token, states)
            next_logits = logits[:, -1, :]
            curr_token = torch.multinomial(F.softmax(next_logits / temp, dim=-1), 1)
            gen_tokens.append(curr_token)
            
        return torch.cat(gen_tokens, dim=1)
