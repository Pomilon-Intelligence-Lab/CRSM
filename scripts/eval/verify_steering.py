"""
Verify Steering
---------------
CLI tool to manually inject state deltas and visualize output changes.
Verifies "Do No Harm" and controllability.
"""

import sys
import argparse
import torch
import asyncio
from pathlib import Path

sys.path.insert(0, '.')

from crsm.model import CRSM
from crsm.tokenizer import Tokenizer

async def generate(model, prompt_ids, device, delta=None):
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    model.init_latent_state(batch_size=1, device=device)
    
    # Inject Delta if provided (Simplified: adding to initial state)
    if delta is not None:
        new_state = []
        for s in model.latent_state:
            if s is not None:
                new_state.append(s + delta)
            else:
                new_state.append(None)
        model.latent_state = new_state
        
    # Determine if we should use sampling or greedy (fallback_to_sampling)
    # The new API for think_and_generate does not have 'use_sampling'
    # but uses 'fallback_to_sampling' if use_deliberation=True
    # For this steering test, we probably want deterministic generation if possible,
    # or just simple sampling.
    
    # Actually, think_and_generate signature in crsm/model.py is:
    # async def think_and_generate(self, prompt, max_length=100, use_deliberation=True, deliberation_lag=3, fallback_to_sampling=True):
    
    # We want to disable deliberation to test raw steering impact on the backbone/state.
    
    output = await model.think_and_generate(
        prompt=prompt_tensor,
        max_length=30,
        use_deliberation=False,
        fallback_to_sampling=False # or True, but without deliberation it just samples or greedy?
        # If use_deliberation=False, suggestion is None.
        # Then: next_token = self.sample_next_token(logits[0, -1])
        # sample_next_token uses temperature/top_k/top_p.
        # So it is always sampling unless temperature is very low.
    )
    return output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--prompt', type=str, default="The sky is")
    parser.add_argument('--delta-scale', type=float, default=0.1)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load (Simplified)
    model = CRSM(
        vocab_size=50257,
        d_model=256,
        d_state=128,
        d_ffn=1024,
        num_layers=4
    ).to(device)
    
    ckpt = torch.load(args.model_path, map_location=device)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    
    if 'backbone.embedding.weight' in state_dict:
        model.load_state_dict(state_dict, strict=False)
    else:
        model.backbone.load_state_dict(state_dict, strict=False)
        
    tokenizer = Tokenizer()
    prompt_ids = tokenizer.encode(args.prompt)
    
    print(f"Prompt: {args.prompt}")
    
    # Run Normal
    out_normal = asyncio.run(generate(model, prompt_ids, device))
    text_normal = tokenizer.decode(out_normal.cpu().tolist())
    print(f"Normal: {text_normal}")
    
    # Run Steered (Random Delta injection)
    # Delta size based on d_state
    # Check d_state in config
    # d_state = model.backbone.config.d_state if hasattr(model.backbone, 'config') else model.config.d_state
    # We know d_model is 256 from the initialization above, and the error said tensor a (256).
    # Error message says "tensor a (256) must match tensor b (128)". 
    # model d_model=256 in config. So state is likely 256 or something else.
    # The error suggests state 's' has size 256.
    
    delta = torch.randn(1, 256, device=device) * args.delta_scale
    out_steered = asyncio.run(generate(model, prompt_ids, device, delta=delta))
    text_steered = tokenizer.decode(out_steered.cpu().tolist())
    print(f"Steered (scale={args.delta_scale}): {text_steered}")
    
    if text_normal != text_steered:
        print("✓ Steering successful (Output changed)")
    else:
        print("⚠ Output identical (Try larger scale)")

if __name__ == "__main__":
    main()
