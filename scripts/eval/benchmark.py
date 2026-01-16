"""
Unified CRSM Benchmarking & Validation Suite
--------------------------------------------
This tool provides a complete set of experiments to verify both the 
FUNCTIONAL integrity and OPERATIONAL validity of the CRSM architecture.

Key Features:
1. Support for both Synthetic Sanity (ARC-Gen) and Official ARC-AGI datasets.
2. Automated Ablation Studies (Greedy vs. MCTS).
3. "Learning Proof" metrics to verify if the Subconscious (Value Critic) is actually learning.
4. Hierarchical Diagnostic Metrics (Structure, First Row, Termination Bias).
"""

import os
import sys
import yaml
import torch
import torch.nn.functional as F
import json
import asyncio
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Modular imports
from crsm.core import CRSMModel, CRSMConfig, LatentDynamics
from crsm.tasks.arc_task import ARCTask
from crsm.data.arc_gen import ARCSanityGenerator
from crsm.training.trainer import Trainer
from crsm.training.logger import logger
from crsm.training.utils import set_seed

class CRSMValidator:
    def __init__(self, config_path, output_dir="experiments/benchmark"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.generator = ARCSanityGenerator()

    async def run_benchmark(self, dataset_type="official", ablation=True, seeds=[42]):
        """
        Runs a comprehensive benchmark across specified seeds.
        """
        logger.info("\n" + "="*60)
        logger.info(f"CRSM VALIDATION BENCHMARK: {dataset_type.upper()}")
        logger.info("="*60)

        summary = {}
        
        # Define tasks based on type
        if dataset_type == "sanity":
            tasks = {
                "identity": self.generator.generate_identity,
                "color_perm": self.generator.generate_color_permutation,
                "translation": self.generator.generate_translation
            }
        else:
            # For official, we treat the dataset as one large task
            tasks = {"official_arc": None}

        for task_name, gen_fn in tasks.items():
            logger.info(f"\n>>> Task: {task_name.upper()}")
            task_results = []
            
            for seed in seeds:
                logger.info(f"  [Seed {seed}]")
                set_seed(seed)
                
                # 1. Load Data
                if dataset_type == "sanity":
                    samples = gen_fn(num_samples=100)
                    task = ARCTask(samples=samples, seq_len=self.config.get('seq_len', 512))
                else:
                    task = ARCTask(
                        data_path=self.config.get('arc_data_path'),
                        eval_path=self.config.get('arc_eval_path'),
                        seq_len=self.config.get('seq_len', 1024)
                    )

                # 2. Instantiate Model
                model_config = CRSMConfig(
                    vocab_size=self.config.get('vocab_size', 100),
                    hidden_size=self.config.get('hidden_size', 128),
                    num_hidden_layers=self.config.get('num_hidden_layers', 4),
                    d_state=self.config.get('d_state', 32),
                    intermediate_size=self.config.get('intermediate_size', 512),
                    injection_rate=self.config.get('injection_rate', 0.05)
                )
                model = CRSMModel(model_config).to(self.device)
                
                # 3. Backbone Training (Operational Proof: Does it learn syntax?)
                logger.info("    [Stage 1] Training Backbone...")
                optimizer = torch.optim.AdamW(model.parameters(), lr=float(self.config.get('lr', 1e-4)))
                trainer = Trainer(model, optimizer, self.config)
                trainer.fit(task, epochs=self.config.get('epochs', 5), checkpoint_dir=str(self.output_dir / "backbone"))

                # 4. Subconscious Training (Operational Proof: Does it learn to judge/simulate?)
                logger.info("    [Stage 2] Training Subconscious (MCTS Engine)...")
                learning_metrics = self.train_subconscious(model, task)
                
                # 5. Ablation Study (The Thesis Proof: Does MCTS help?)
                modes = ["greedy", "mcts"] if ablation else ["mcts"]
                seed_metrics = {}
                
                for mode in modes:
                    logger.info(f"    [Eval] Mode: {mode.upper()}")
                    # Configure model
                    model.crsm.reasoning.dynamics_model = model.crsm.dynamics if mode == "mcts" else None
                    
                    metrics = await task.evaluate_async(model, self.device)
                    seed_metrics[mode] = metrics
                    
                    # Log improvement delta
                    if mode == "mcts" and "greedy" in seed_metrics:
                        delta = metrics['accuracy'] - seed_metrics['greedy']['accuracy']
                        logger.info(f"      -> MCTS Improvement Delta: {delta*100:+.2f}% accuracy")
                        struct_delta = metrics['struct_correct_wrong_color_rate'] - seed_metrics['greedy']['struct_correct_wrong_color_rate']
                        logger.info(f"      -> MCTS Structure Delta:   {struct_delta*100:+.2f}% structure")

                task_results.append({
                    "seed": seed,
                    "metrics": seed_metrics,
                    "learning_proof": learning_metrics
                })
                
            summary[task_name] = task_results
            
        self._save_summary(summary, f"{dataset_type}_benchmark_results.json")
        return summary

    def train_subconscious(self, model, task, epochs=5):
        """
        Trains Dynamics and Value heads. 
        Records 'Learning Proof' metrics using DIVERSE LOGICAL NEGATIVES.
        """
        device = self.device
        model.eval()
        
        # 1. Setup
        if model.crsm.dynamics is None:
            model.crsm.dynamics = LatentDynamics(
                d_model=model.config.hidden_size, 
                num_layers=model.config.num_hidden_layers
            ).to(device)
        
        opt_dyn = torch.optim.AdamW(model.crsm.dynamics.parameters(), lr=1e-3)
        
        # Freeze backbone, train values
        for param in model.crsm.backbone.parameters(): param.requires_grad = False
        for v_head in model.crsm.backbone.value_heads:
            for param in v_head.parameters(): param.requires_grad = True
        opt_val = torch.optim.AdamW(model.crsm.backbone.value_heads.parameters(), lr=1e-3)
        
        train_loader, _ = task.get_dataloaders(self.config.get('batch_size', 8))
        learning_metrics = []

        for epoch in range(1, epochs + 1):
            total_dyn_loss = 0
            total_val_acc = 0 
            total_samples = 0
            
            for batch in train_loader:
                x, y, split_indices = batch
                x, y = x.to(device), y.to(device)
                
                # --- A. Positive Sample (Logical Truth) ---
                with torch.no_grad():
                    _, pos_states = model.crsm.backbone(x)
                
                # --- B. Negative Samples (Logical Rejection Suite) ---
                neg_states_list = []
                
                # Neg 1: Premature Termination (Suicidal Bias)
                wrong_x1 = x.clone()
                for i, split_idx in enumerate(split_indices):
                    if split_idx < x.size(1): wrong_x1[i, split_idx] = 14
                
                # Neg 2: Random Logic (Garbage)
                wrong_x2 = x.clone()
                for i, split_idx in enumerate(split_indices):
                    if split_idx < x.size(1): wrong_x2[i, split_idx] = torch.randint(0, 10, (1,)).item()
                
                # Neg 3: Syntax Error (Missing ROW_END)
                wrong_x3 = x.clone()
                wrong_x3[wrong_x3 == 13] = torch.randint(0, 10, (1,)).item()

                with torch.no_grad():
                    _, s1 = model.crsm.backbone(wrong_x1)
                    _, s2 = model.crsm.backbone(wrong_x2)
                    _, s3 = model.crsm.backbone(wrong_x3)
                    neg_states_list = [s1, s2, s3]

                # --- Train Dynamics ---
                opt_dyn.zero_grad()
                batch_dyn_loss = 0
                for i, split_idx in enumerate(split_indices):
                    if split_idx >= x.size(1) - 1: continue
                    s_t = [s[i:i+1] if s is not None else None for s in pos_states]
                    action = y[i, split_idx].item()
                    action_emb = model.crsm.backbone.embedding(torch.tensor([[action]], device=device)).squeeze(1)
                    pred_deltas = model.crsm.dynamics(s_t, action_emb)
                    _, next_states = model.crsm.backbone.step(torch.tensor([[action]], device=device), s_t)
                    for j in range(len(s_t)):
                        if s_t[j] is not None and next_states[j] is not None:
                            batch_dyn_loss += F.mse_loss(pred_deltas[j], next_states[j] - s_t[j])
                
                if isinstance(batch_dyn_loss, torch.Tensor):
                    batch_dyn_loss.backward()
                    opt_dyn.step()
                    total_dyn_loss += batch_dyn_loss.item()

                # --- Train Values (Operational Verifier) ---
                opt_val.zero_grad()
                val_preds_pos = model.crsm.backbone._compute_layer_values(pos_states)
                
                loss_val = 0
                for neg_states in neg_states_list:
                    val_preds_neg = model.crsm.backbone._compute_layer_values(neg_states)
                    for vp, vn in zip(val_preds_pos, val_preds_neg):
                        # Margin ranking: vp must be > vn + 1.0
                        loss_val += F.margin_ranking_loss(vp, vn, torch.ones_like(vp), margin=1.0)
                
                loss_val.backward()
                opt_val.step()
                
                with torch.no_grad():
                    # Average over all negatives
                    v_pos = torch.stack(val_preds_pos).mean().item()
                    v_neg = sum(torch.stack(model.crsm.backbone._compute_layer_values(ns)).mean().item() for ns in neg_states_list) / 3
                    if v_pos > v_neg: total_val_acc += 1
                total_samples += 1

            discrimination_score = total_val_acc / max(1, total_samples)
            logger.info(f"      Subconscious Epoch {epoch}/{epochs} | Dyn Loss: {total_dyn_loss/max(1, total_samples):.4f} | Logical Discrimination: {discrimination_score*100:.1f}%")
            learning_metrics.append({"epoch": epoch, "discrimination": discrimination_score})

        for param in model.crsm.backbone.parameters(): param.requires_grad = True
        return learning_metrics

        for param in model.crsm.backbone.parameters(): param.requires_grad = True
        return learning_metrics

        for param in model.crsm.backbone.parameters(): param.requires_grad = True
        return learning_metrics

        # Re-enable grads
        for param in model.crsm.backbone.parameters(): param.requires_grad = True
        return learning_metrics

    def _save_summary(self, summary, filename):
        with open(self.output_dir / filename, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"\n✓ Validation Summary saved to {self.output_dir / filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/arc_nano.yaml")
    parser.add_argument('--type', type=str, default="official", choices=["sanity", "official"])
    parser.add_argument('--no-ablation', action='store_true')
    args = parser.parse_args()
    
    validator = CRSMValidator(args.config)
    asyncio.run(validator.run_benchmark(
        dataset_type=args.type, 
        ablation=not args.no_ablation,
        seeds=[42] # Start with one seed for time efficiency
    ))