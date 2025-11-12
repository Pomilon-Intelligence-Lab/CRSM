"""
Generate text using a trained CRSM model with saved vocabulary.
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


async def generate_text(model, tokenizer, prompt, max_length, device, use_sampling=True):
    """Generate text with async deliberation."""
    
    # Tokenize prompt
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        print("⚠ Warning: Empty prompt after tokenization!")
        prompt_ids = [0]
    
    print(f"\nTokenization:")
    print(f"  Prompt: {prompt}")
    print(f"  Token IDs: {prompt_ids}")
    print(f"  Decoded: {tokenizer.decode(prompt_ids)}")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    
    # Initialize latent state
    model.init_latent_state(batch_size=1, device=device)
    
    if use_sampling:
        print(f"\nGenerating {max_length} tokens with temperature sampling (T={model.temperature})...")
    else:
        print(f"\nGenerating {max_length} tokens with {model.reasoning.n_simulations} MCTS simulations...")
    
    # Generate
    generated = await model.think_and_generate(
        prompt=prompt_tensor,
        max_length=max_length,
        use_sampling=use_sampling
    )
    
    # Decode
    generated_ids = generated.cpu().tolist()
    output_text = tokenizer.decode(generated_ids)
    
    return output_text, generated_ids


def main():
    parser = argparse.ArgumentParser(description="Generate text with CRSM and saved vocab")
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained CRSM checkpoint')
    parser.add_argument('--vocab-path', type=str, required=True,
                       help='Path to saved vocab.json')
    parser.add_argument('--dynamics-path', type=str, default=None,
                       help='Path to trained dynamics model (optional)')
    parser.add_argument('--config-path', type=str, default='configs/small.json',
                       help='Path to config JSON')
    parser.add_argument('--prompt', type=str, required=True,
                       help='Text prompt to continue')
    parser.add_argument('--max-length', type=int, default=30,
                       help='Maximum tokens to generate')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda/cpu)')
    parser.add_argument('--use-mcts', action='store_true',
                       help='Use MCTS instead of sampling (slower but more deliberate)')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature (lower = more focused)')
    parser.add_argument('--top-k', type=int, default=50,
                       help='Top-k sampling parameter')
    parser.add_argument('--top-p', type=float, default=0.95,
                       help='Nucleus sampling parameter')
    
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from: {args.config_path}")
    with open(args.config_path, 'r') as f:
        config_dict = json.load(f)
    
    config = CRSMConfig(
        vocab_size=config_dict['model']['vocab_size'],
        hidden_size=config_dict['model'].get('d_model', config_dict.get('hidden_size', 128)),
        d_state=config_dict['model'].get('d_state', 64),
        intermediate_size=config_dict['model'].get('d_ffn', config_dict.get('intermediate_size', 512)),
        num_hidden_layers=config_dict['model'].get('num_layers', config_dict.get('num_hidden_layers', 2)),
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
    
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Extract state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    else:
        state_dict = checkpoint
    
    # Load backbone
    if any(k.startswith('backbone.') for k in state_dict.keys()):
        model.load_state_dict(state_dict, strict=False)
        print("✓ Loaded full CRSM checkpoint")
    else:
        model.backbone.load_state_dict(state_dict, strict=False)
        print("✓ Loaded backbone checkpoint")
    
    # Load dynamics if provided
    if args.dynamics_path and Path(args.dynamics_path).exists():
        success = model.load_dynamics(args.dynamics_path)
        if success:
            print("✓ Using fast dynamics for MCTS rollouts")
    else:
        print("⚠ No dynamics model - using slow SSM fallback for MCTS")
    
    model.to(device)
    model.eval()
    
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
            use_sampling=(not args.use_mcts)
        )
    
    output, token_ids = asyncio.run(run())
    
    print("\n" + "="*70)
    print("RESULT")
    print("="*70)
    print(f"\nGenerated text:\n{output}")
    print(f"\nToken IDs ({len(token_ids)} tokens):")
    print(f"{token_ids[:50]}{'...' if len(token_ids) > 50 else ''}")
    print("\n" + "="*70)


if __name__ == '__main__':
    main()