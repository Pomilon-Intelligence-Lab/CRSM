"""
Generate text using a trained CRSM model with saved vocabulary.
Updated to show self-modification metrics during generation.
"""
import argparse
import asyncio
import json
import torch
import sys
from pathlib import Path

sys.path.insert(0, '.')

from crsm.model import CRSM, CRSMConfig
from crsm.tokenizer import Tokenizer


async def generate_text(model, tokenizer, prompt, max_length, device, use_sampling=True, show_metrics=False):
    """Generate text with async deliberation and optional metrics."""
    
    # Tokenize prompt
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        print("⚠ Warning: Empty prompt after tokenization!")
        prompt_ids = [0]
    
    print(f"\nTokenization:")
    print(f"  Prompt: {prompt}")
    print(f"  Token IDs: {prompt_ids[:20]}{'...' if len(prompt_ids) > 20 else ''}")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    
    # Initialize latent state
    model.init_latent_state(batch_size=1, device=device)
    initial_state = [s.clone() if s is not None else None for s in model.latent_state]
    
    if use_sampling:
        print(f"\nGenerating {max_length} tokens with temperature sampling (T={model.temperature})...")
    else:
        print(f"\nGenerating {max_length} tokens with {model.reasoning.n_simulations} MCTS simulations...")
        print("  (Self-modification active)")
    
    # Generate
    import time
    start_time = time.time()
    
    generated = await model.think_and_generate(
        prompt=prompt_tensor,
        max_length=max_length,
        use_sampling=use_sampling
    )
    
    elapsed = time.time() - start_time
    
    # Measure self-modification if MCTS was used
    if not use_sampling and show_metrics:
        final_state = model.latent_state
        total_change = 0.0
        num_layers = 0
        
        for s_init, s_final in zip(initial_state, final_state):
            if s_init is not None and s_final is not None:
                change = torch.norm(s_final - s_init).item()
                total_change += change
                num_layers += 1
        
        avg_change = total_change / max(1, num_layers)
        print(f"\nSelf-Modification Metrics:")
        print(f"  Average state change: {avg_change:.6f}")
        print(f"  Tokens/second: {max_length/elapsed:.2f}")
    
    # Decode
    generated_ids = generated.cpu().tolist()
    output_text = tokenizer.decode(generated_ids)
    
    return output_text, generated_ids, elapsed


def load_model_from_checkpoint(checkpoint_path, config, device):
    """Load model with improved checkpoint handling."""
    checkpoint_path = Path(checkpoint_path)
    
    # Detect checkpoint format
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
        checkpoint_format = "new"
    elif 'model_state' in ckpt:
        state_dict = ckpt['model_state']
        checkpoint_format = "old"
    else:
        state_dict = ckpt
        checkpoint_format = "raw"
    
    is_crsm = any(k.startswith('backbone.') for k in state_dict.keys())
    
    print(f"  Checkpoint format: {checkpoint_format}")
    print(f"  Type: {'CRSM' if is_crsm else 'MambaModel'}")
    
    # Create model
    model = CRSM(
        vocab_size=config.vocab_size,
        d_model=config.hidden_size,
        d_state=config.d_state,
        d_ffn=config.intermediate_size,
        num_layers=config.num_hidden_layers,
        dropout=config.dropout,
        c_puct=config.c_puct,
        n_simulations=config.n_simulations,
        temperature=config.temperature,
        top_k=config.top_k,
        top_p=config.top_p,
    )
    
    # Load weights
    if is_crsm:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"  ✓ Loaded CRSM checkpoint")
        if missing and len(missing) < 10:
            print(f"    Missing: {missing}")
    else:
        model.backbone.load_state_dict(state_dict, strict=False)
        print(f"  ✓ Loaded backbone checkpoint")
    
    model.to(device)
    model.eval()
    
    return model, is_crsm


def main():
    parser = argparse.ArgumentParser(description="Generate text with CRSM")
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--vocab-path', type=str, required=True)
    parser.add_argument('--dynamics-path', type=str, default=None)
    parser.add_argument('--config-path', type=str, default='configs/small.json')
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--max-length', type=int, default=30)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--use-mcts', action='store_true')
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--top-p', type=float, default=0.95)
    parser.add_argument('--show-metrics', action='store_true',
                       help='Show self-modification metrics')
    
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from: {args.config_path}")
    with open(args.config_path, 'r') as f:
        config_dict = json.load(f)
    
    config = CRSMConfig(
        vocab_size=config_dict['model']['vocab_size'],
        hidden_size=config_dict['model'].get('d_model', 128),
        d_state=config_dict['model'].get('d_state', 64),
        intermediate_size=config_dict['model'].get('d_ffn', 512),
        num_hidden_layers=config_dict['model'].get('num_layers', 2),
        dropout=config_dict['model'].get('dropout', 0.1),
        c_puct=config_dict['reasoning'].get('c_puct', 1.0),
        n_simulations=config_dict['reasoning'].get('n_simulations', 10),
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    
    # Setup device
    device = torch.device(args.device if args.device else 
                         ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Using device: {device}")
    
    # Load tokenizer
    print(f"\nLoading vocabulary from: {args.vocab_path}")
    tokenizer = Tokenizer.from_vocab_file(args.vocab_path)
    print(f"✓ Loaded vocabulary with {tokenizer.vocab_size} tokens")
    
    # Load model
    print(f"\nLoading model from: {args.model_path}")
    model, is_crsm = load_model_from_checkpoint(args.model_path, config, device)
    
    # Load dynamics
    has_dynamics = False
    if args.dynamics_path and Path(args.dynamics_path).exists():
        success = model.load_dynamics(args.dynamics_path)
        has_dynamics = success
    elif is_crsm and 'dynamics.net.0.weight' in torch.load(args.model_path, map_location='cpu').get('model_state_dict', {}):
        has_dynamics = True
        print("  ✓ Using dynamics from checkpoint")
    
    if args.use_mcts:
        if has_dynamics:
            print("  ✓ MCTS will use fast dynamics")
        else:
            print("  ⚠ MCTS will use slow SSM fallback (no dynamics)")
    
    # Generate
    print("\n" + "="*70)
    print("GENERATION")
    print("="*70)
    
    async def run():
        return await generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_length=args.max_length,
            device=device,
            use_sampling=(not args.use_mcts),
            show_metrics=args.show_metrics
        )
    
    output, token_ids, elapsed = asyncio.run(run())
    
    print("\n" + "="*70)
    print("RESULT")
    print("="*70)
    print(f"\nGenerated text:\n{output}")
    print(f"\nGeneration time: {elapsed:.2f}s")
    print(f"Tokens: {len(token_ids)}")
    print(f"Speed: {len(token_ids)/elapsed:.2f} tokens/sec")
    print("\n" + "="*70)


if __name__ == '__main__':
    main()