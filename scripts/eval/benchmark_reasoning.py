"""
Benchmark Reasoning Capabilities
--------------------------------
Measures the impact of MCTS by comparing performance with and without deliberation.
Adaptation of the old evaluate_metrics.py.
"""

import sys
import argparse
import json
import torch
import asyncio
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, '.')

from crsm.core import CRSM
from crsm.data.tokenizers import Tokenizer
from crsm.training.utils import set_seed

async def evaluate(model, tokenizer, prompts, device, use_mcts):
    results = []
    
    for prompt in tqdm(prompts, desc=f"Eval (MCTS={use_mcts})"):
        prompt_ids = tokenizer.encode(prompt)
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        
        model.init_latent_state(batch_size=1, device=device)
        
        # We assume if use_mcts is False, we disable it in the model config or call args
        # CRSM logic usually uses MCTS if dynamics are present and n_simulations > 0
        
        try:
            output = await model.think_and_generate(
                prompt=prompt_tensor,
                max_length=50,
                use_deliberation=use_mcts,
                fallback_to_sampling=not use_mcts
            )
            generated_text = tokenizer.decode(output.cpu().tolist())
            results.append(generated_text)
        except Exception as e:
            print(f"Error: {e}")
            results.append("")
            
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--dynamics-path', required=True)
    parser.add_argument('--dataset', default='data/reasoning_tasks.jsonl')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(42)
    
    # Load Model (assuming standard config for now)
    model = CRSM(
        vocab_size=50257,
        d_model=256,
        d_state=128,
        d_ffn=1024,
        num_layers=4,
        n_simulations=10
    ).to(device)
    
    # Load Weights
    ckpt = torch.load(args.model_path, map_location=device)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    
    # Handle key mismatches if loading a raw backbone vs full CRSM
    if 'backbone.embedding.weight' in state_dict:
        model.load_state_dict(state_dict, strict=False)
    else:
        model.backbone.load_state_dict(state_dict, strict=False)
        
    model.load_dynamics(args.dynamics_path)
    model.eval()
    
    tokenizer = Tokenizer()
    
    # Dummy data if file missing
    if not Path(args.dataset).exists():
        prompts = ["1+1=", "if a=b and b=c then a=", "The capital of France is"]
    else:
        with open(args.dataset) as f:
            prompts = [json.loads(line)['prompt'] for line in f]

    print("\n" + "="*60)
    print("BENCHMARK: Reasoning Impact")
    print("="*60)
    
    # 1. Baseline (System 1)
    results_s1 = asyncio.run(evaluate(model, tokenizer, prompts, device, use_mcts=False))
    
    # 2. Centaur (System 2)
    results_s2 = asyncio.run(evaluate(model, tokenizer, prompts, device, use_mcts=True))
    
    print("\nResults:")
    for i, p in enumerate(prompts[:5]):
        print(f"\nPrompt: {p}")
        print(f"  Sys1: {results_s1[i]}")
        print(f"  Sys2: {results_s2[i]}")
        
    # In a real scenario, we would parse answers and compute accuracy.
    
if __name__ == "__main__":
    main()
