import torch
import yaml
import asyncio
import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass

sys.path.insert(0, '.')

from crsm.core import CRSMModel, CRSMConfig
from crsm.tasks.arc_task import ARCTask
from crsm.training.logger import logger
from crsm.training.utils import set_seed

@dataclass
class EvalConfig:
    name: str
    use_mcts: bool
    use_critics: bool
    use_dynamics: bool

async def evaluate_config(model, task, config: EvalConfig, device, seeds):
    results = {
        "accuracy": [],
        "reward": [],
        "first_row": [],
        "struct": [],
        "len_ratio": [],
        "premature_end": []
    }
    
    # Store original components to restore later
    inner_model = model.crsm if hasattr(model, 'crsm') else model
    original_dynamics = inner_model.reasoning.dynamics_model
    original_value_heads = inner_model.backbone.value_heads
    
    # Apply Ablations
    if not config.use_dynamics:
        inner_model.reasoning.dynamics_model = None # Disable dynamics projection
    
    if not config.use_critics:
        # Mock value heads to return uniform/zero values if disabled?
        # Or just detach them? 
        # MCTS uses `predict_policy_value` or `predict_from_states`.
        # We can mock the function or zero out weights temporarily.
        # Safest is to zero out weights for this test to simulate "ignorant" critic.
        for head in inner_model.backbone.value_heads:
            head.weight.data.fill_(0.0)
            head.bias.data.fill_(0.0)
            
    # Evaluation Loop over Seeds
    for seed in seeds:
        set_seed(seed)
        # Re-initialize state logic if needed? No, seed controls sampling.
        
        # We need to access the task's data loader to iterate
        loader, _ = task.get_dataloaders(batch_size=1)
        
        total_items = 0
        seed_stats = {k: 0 for k in results.keys()}
        len_ratios = []
        
        with torch.no_grad():
            for batch in loader:
                x, y, split_indices = batch
                split_idx = split_indices[0].item()
                
                x_in = x[:, :split_idx].to(device)
                
                # Robust target extraction
                raw_targets = y[0, split_idx-1:].tolist()
                target_tokens = []
                for t in raw_targets:
                    target_tokens.append(t)
                    if t == 14: break
                
                # Generate
                # Note: think_and_generate uses inner_model state
                output_ids = await inner_model.think_and_generate(
                    x_in, 
                    max_length=len(target_tokens) + 20, 
                    use_deliberation=config.use_mcts,
                    deliberation_lag=0, # Sync for strict eval
                    fallback_to_sampling=False
                )
                
                pred_tokens = output_ids[split_idx:].tolist()
                
                # Metrics Calculation
                # 1. Accuracy (Exact Match)
                is_correct = (pred_tokens[:len(target_tokens)] == target_tokens)
                seed_stats['accuracy'] += 1 if is_correct else 0
                
                # 2. Reward (Dense)
                if is_correct:
                    r = 1.0
                elif 14 in pred_tokens:
                    p_trunc = pred_tokens[:len(target_tokens)]
                    matches = sum(1 for p, g in zip(p_trunc, target_tokens) if p == g)
                    acc = matches / max(1, len(target_tokens))
                    r = 0.1 + 0.4 * acc
                else:
                    r = 0.0
                seed_stats['reward'] += r
                
                # 3. First Row
                # Find first 13
                try:
                    p_row_end = pred_tokens.index(13)
                    t_row_end = target_tokens.index(13)
                    if pred_tokens[:p_row_end+1] == target_tokens[:t_row_end+1]:
                        seed_stats['first_row'] += 1
                except ValueError:
                    pass
                    
                # 4. Structure
                p_struct = [1 if 0 <= t <= 9 else t for t in pred_tokens]
                t_struct = [1 if 0 <= t <= 9 else t for t in target_tokens]
                if p_struct == t_struct and not is_correct:
                    seed_stats['struct'] += 1
                    
                # 5. Length Ratio
                if len(target_tokens) > 0:
                    lr = len(pred_tokens) / len(target_tokens)
                    len_ratios.append(lr)
                    
                # 6. Premature End
                if 14 in pred_tokens and len(pred_tokens) < len(target_tokens) * 0.5:
                    seed_stats['premature_end'] += 1
                    
                total_items += 1
                
        # Average for this seed
        if total_items > 0:
            for k in seed_stats:
                if k == 'len_ratio':
                    results[k].append(np.mean(len_ratios) if len_ratios else 0)
                else:
                    results[k].append(seed_stats[k] / total_items)
    
    # Restore components
    inner_model.reasoning.dynamics_model = original_dynamics
    # Restore value heads (reload from checkpoint? Or clone before?)
    # Since we modified weights in-place, we must reload or clone.
    # Simpler to just re-load weights from original_value_heads if we cloned them,
    # but since they are modules, deepcopy is needed.
    # For now, we will rely on the caller reloading the model between configs if needed,
    # or implement restore logic here.
    
    # Return averaged stats across seeds
    final_metrics = {k: np.mean(v) for k, v in results.items()}
    return final_metrics

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--sft_ckpt', type=str, required=True)
    parser.add_argument('--rl_ckpt', type=str, required=True)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
        
    seeds = [0, 1, 2, 3, 4]
    
    # Define Configurations
    rl_configs = [
        EvalConfig("A. Full System", True, True, True),
        EvalConfig("B. No Planning", False, False, False),
        EvalConfig("C. No Critics", True, False, True),
        EvalConfig("D. No Dynamics", True, True, False)
    ]
    
    sft_configs = [
        EvalConfig("Full System", True, True, True),
        EvalConfig("Greedy Only", False, False, False)
    ]
    
    # Helper to load model fresh
    def load_model(ckpt_path):
        config = CRSMConfig(
            vocab_size=config_dict.get('vocab_size', 100),
            hidden_size=config_dict.get('hidden_size', 256),
            num_hidden_layers=config_dict.get('num_hidden_layers', 8),
            d_state=config_dict.get('d_state', 64),
            intermediate_size=config_dict.get('intermediate_size', 1024),
            injection_rate=config_dict.get('injection_rate', 0.2),
            n_simulations=50 # Reduced for eval speed
        )
        model = CRSMModel(config)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
        model.to(device)
        model.eval()
        return model

    # Task setup
    task = ARCTask(
        data_path=config_dict.get('arc_eval_path'), # Use EVAL split
        seq_len=config_dict.get('seq_len', 4096)
    )
    
    results_table = []

    print("\n--- Running RL Epoch 10 Ablations ---")
    for cfg in rl_configs:
        print(f"Running {cfg.name}...")
        # Reload model each time to reset weights (critical for 'No Critics' ablation)
        model = load_model(args.rl_ckpt)
        metrics = await evaluate_config(model, task, cfg, device, seeds)
        row = {
            "Model": "RL Ep10",
            "Config": cfg.name,
            "MCTS": cfg.use_mcts,
            "Critics": cfg.use_critics,
            "Dynamics": cfg.use_dynamics,
            "ARC Acc": f"{metrics['accuracy']*100:.2f}%",
            "Avg Reward": f"{metrics['reward']:.4f}",
            "First Row %": f"{metrics['first_row']*100:.2f}%",
            "Struct %": f"{metrics['struct']*100:.2f}%",
            "Avg Len": f"{metrics['len_ratio']:.2f}",
            "Premature End": f"{metrics['premature_end']:.2f}"
        }
        results_table.append(row)

    print("\n--- Running SFT Baseline Comparison ---")
    for cfg in sft_configs:
        print(f"Running {cfg.name}...")
        model = load_model(args.sft_ckpt)
        metrics = await evaluate_config(model, task, cfg, device, seeds)
        row = {
            "Model": "SFT Base",
            "Config": cfg.name,
            "MCTS": cfg.use_mcts,
            "Critics": cfg.use_critics,
            "Dynamics": cfg.use_dynamics,
            "ARC Acc": f"{metrics['accuracy']*100:.2f}%",
            "Avg Reward": f"{metrics['reward']:.4f}",
            "First Row %": f"{metrics['first_row']*100:.2f}%",
            "Struct %": f"{metrics['struct']*100:.2f}%",
            "Avg Len": f"{metrics['len_ratio']:.2f}",
            "Premature End": f"{metrics['premature_end']:.2f}"
        }
        results_table.append(row)

    # Print Table
    df = pd.DataFrame(results_table)
    print("\n" + "="*80)
    print("FINAL CAUSALITY EVALUATION REPORT")
    print("="*80)
    print(df.to_markdown(index=False))
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())
