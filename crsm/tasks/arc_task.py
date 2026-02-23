import torch
import torch.nn as nn
import torch.nn.functional as F
import asyncio
from torch.utils.data import DataLoader
from .base import BaseTask
from ..data.datasets import ARCDataset

class ARCTask(BaseTask):
    """
    ARC-AGI Task: Solve grid-based puzzles.
    Focuses loss on the [OUTPUT_START] sequence.
    """
    
    def __init__(self, data_path=None, eval_path=None, samples=None, seq_len=1024):
        self.data_path = data_path
        self.eval_path = eval_path
        self.samples = samples
        self.seq_len = seq_len
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def get_dataloaders(self, batch_size):
        # 1. Training DataLoader
        train_samples = self.samples
        if not self.data_path and not train_samples:
            # Fallback to dummy if nothing provided
            train_samples = [{
                "train": [{"input": [[1, 1], [1, 1]], "output": [[2, 2], [2, 2]]}],
                "test": [{"input": [[1, 1], [1, 1]], "output": [[2, 2], [2, 2]]}]
            }]
            
        train_ds = ARCDataset(data_path=self.data_path, samples=train_samples, seq_len=self.seq_len)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        
        # 2. Validation DataLoader
        val_loader = None
        if self.eval_path:
            val_ds = ARCDataset(data_path=self.eval_path, seq_len=self.seq_len)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
            
        return train_loader, val_loader

    def compute_loss(self, model, batch, device):
        x, y, split_indices = batch
        x, y = x.to(device), y.to(device)
        
        # Forward pass with Value Critic support
        if hasattr(model, 'predict_policy_value'):
            logits, values, _ = model.predict_policy_value(x)
        else:
            logits, _ = model(x)
            values = None
        
        b, t, v = logits.size()
        loss_all = self.criterion(logits.reshape(b * t, v), y.reshape(b * t)).reshape(b, t)
        
        # Mask loss to only count tokens after split_index (the test output part)
        mask = torch.zeros_like(loss_all)
        for i, split_idx in enumerate(split_indices):
            idx = min(split_idx, t)
            mask[i, idx-1:] = 1.0 # -1 because y is shifted
            
        lm_loss = (loss_all * mask).sum() / mask.sum().clamp(min=1.0)
        
        if values is None:
            return lm_loss

        # Outcome-Based Value Supervision
        # For ARC training, we supervise the value heads to predict 1.0 
        # for states that are on the correct solution path.
        # Target value is 1.0 for all steps in the correct path.
        target_value = torch.ones(b, device=device)
        
        # values is List[Tensor] (one per layer), each (batch, 1) or (batch,)
        # Note: predict_policy_value currently returns values for the LAST token
        value_loss = sum(F.mse_loss(v.squeeze(-1), target_value) for v in values)
        
        total_loss = lm_loss + 1.0 * value_loss
        
        # Hierarchical Weight Supervision: Entropy Loss
        inner_model = model.crsm if hasattr(model, 'crsm') else model
        if hasattr(inner_model.backbone, 'layer_fusion_weights'):
            weights = torch.softmax(inner_model.backbone.layer_fusion_weights, dim=0)
            entropy = -torch.sum(weights * torch.log(weights + 1e-8))
            total_loss = total_loss - 0.01 * entropy
            
        return total_loss

    async def evaluate_async(self, model, device):
        """
        Full ARC evaluation with detailed diagnostic metrics.
        """
        loader, _ = self.get_dataloaders(batch_size=1)
        model.eval()
        
        results = {
            "accuracy": 0.0,
            "first_row_rate": 0.0,
            "struct_correct_wrong_color_rate": 0.0,
            "len_ratio_mean": 0.0,
            "premature_end_count": 0,
            "total": 0
        }
        
        correct = 0
        first_row_correct = 0
        struct_correct = 0
        len_ratios = []
        
        inner_model = model.crsm if hasattr(model, 'crsm') else model

        with torch.no_grad():
            for x, y, split_idx in loader:
                results["total"] += 1
                if results["total"] % 10 == 0:
                    logger.info(f"  Evaluating sample {results['total']}...")
                
                x_in = x[:, :split_idx].to(device)
                target_tokens = [t for t in y[0, split_idx-1:].tolist() if t != 0]
                
                # Generate tokens
                if hasattr(inner_model, 'think_and_generate'):
                    # Force fully synchronous deliberation for zero-lag sovereignty
                    output_ids = await inner_model.think_and_generate(
                        x_in, 
                        max_length=len(target_tokens) + 10, 
                        use_deliberation=inner_model.reasoning is not None,
                        deliberation_lag=0, # ZERO LAG: DELIBERATE ON CURRENT TOKEN
                        fallback_to_sampling=False # NEVER FALLBACK: WAIT FOR LOGIC
                    )
                else:
                    output_ids = self._greedy_generate(model, x_in, max_len=len(target_tokens) + 10)
                
                pred_tokens = output_ids[split_idx:].tolist()
                # Remove padding/trailing zeros
                pred_tokens = [t for t in pred_tokens if t != 0]
                
                # 1. Exact Match Accuracy
                if self._match_tokens(pred_tokens, target_tokens):
                    correct += 1
                
                # 2. First Row Check
                if self._match_first_row(pred_tokens, target_tokens):
                    first_row_correct += 1
                    
                # 3. Structure vs Color
                if self._match_structure(pred_tokens, target_tokens):
                    struct_correct += 1
                    
                # 4. Length Ratio
                if len(target_tokens) > 0:
                    len_ratios.append(len(pred_tokens) / len(target_tokens))
                    
                # 5. Premature Termination
                if pred_tokens and pred_tokens[-1] == 14 and len(pred_tokens) < len(target_tokens):
                    results["premature_end_count"] += 1
                elif 14 not in pred_tokens:
                    # Didn't even reach GRID_END
                    pass 

                results["total"] += 1
                
        total = results["total"]
        if total > 0:
            results["accuracy"] = correct / total
            results["first_row_rate"] = first_row_correct / total
            # Only count as "struct correct wrong color" if NOT perfectly correct
            wrong_color = struct_correct - correct
            results["struct_correct_wrong_color_rate"] = max(0, wrong_color) / total
            results["len_ratio_mean"] = sum(len_ratios) / len(len_ratios) if len_ratios else 0.0
            
        return results

    def _greedy_generate(self, model, x, max_len=100):
        curr = x
        for _ in range(max_len):
            logits, _ = model(curr)
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            curr = torch.cat([curr, next_token.unsqueeze(1)], dim=1)
            if next_token.item() == 14: break
        return curr[0]

    def _match_first_row(self, pred, target):
        # First row ends at first token 13
        p_row = []
        for t in pred:
            if t == 13: break
            p_row.append(t)
        
        t_row = []
        for t in target:
            if t == 13: break
            t_row.append(t)
            
        return p_row == t_row if t_row else False

    def _match_structure(self, pred, target):
        # Convert colors to placeholder 1
        p_struct = [1 if 0 <= t <= 9 else t for t in pred]
        t_struct = [1 if 0 <= t <= 9 else t for t in target]
        return p_struct == t_struct

    def evaluate(self, model, device):
        # Detect if we are already inside a running event loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # We are already in an event loop (e.g. from an async test or think_and_generate)
            # We can't use run_until_complete here. 
            # If evaluate is called from sync code, it should use asyncio.run.
            # If evaluate is called from async code, it should be awaited.
            
            # Since evaluate is defined as sync in BaseTask, we have a problem.
            # Best approach for now: return a placeholder or use a nested loop (risky).
            # Actually, most training loops are sync.
            
            # Let's try to run it in a separate thread if loop is running
            import threading
            from concurrent.futures import ThreadPoolExecutor
            
            with ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self.evaluate_async(model, device))
                return future.result()
        else:
            return asyncio.run(self.evaluate_async(model, device))

    def _match_tokens(self, pred, target):
        p = [t for t in pred if t != 0]
        t = [t for t in target if t != 0]
        if not p or not t: return False
        return p[:len(t)] == t
