"""
Evaluate CRSM model on standard metrics including self-modification impact.
"""
import torch
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import sys
import asyncio

sys.path.insert(0, '.')

from crsm.model import CRSM, CRSMConfig
from crsm.tokenizer import Tokenizer
from torch.utils.data import DataLoader
from crsm.dataset import RealTextDataset


def compute_perplexity(model, dataloader, device, vocab_size):
    """Compute perplexity on validation set."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Computing perplexity"):
            x = x.to(device)
            y = y.to(device)
            
            logits, _ = model.backbone(x, states=None)
            
            loss = criterion(
                logits.reshape(-1, vocab_size),
                y.reshape(-1)
            )
            
            total_loss += loss.item()
            total_tokens += y.numel()
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    return perplexity, avg_loss


async def evaluate_generation_quality(model, tokenizer, prompts, device, use_mcts=False):
    """Evaluate generation quality metrics."""
    
    results = {
        'repetition_scores': [],
        'diversity_scores': [],
        'length_scores': [],
        'state_changes': [] if use_mcts else None,
    }
    
    async def generate_one(prompt):
        prompt_ids = tokenizer.encode(prompt)
        if not prompt_ids:
            prompt_ids = [0]
        
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        model.init_latent_state(batch_size=1, device=device)
        
        # Capture initial state if using MCTS
        if use_mcts:
            initial_state = [s.clone() if s is not None else None for s in model.latent_state]
        
        try:
            generated = await model.think_and_generate(
                prompt=prompt_tensor,
                max_length=30,
                use_sampling=(not use_mcts)
            )
            
            # Measure state change
            if use_mcts:
                final_state = model.latent_state
                total_change = 0.0
                num_layers = 0
                
                for s_init, s_final in zip(initial_state, final_state):
                    if s_init is not None and s_final is not None:
                        change = torch.norm(s_final - s_init).item()
                        total_change += change
                        num_layers += 1
                
                avg_change = total_change / max(1, num_layers)
                results['state_changes'].append(avg_change)
            
            return generated.cpu().tolist()
        except Exception as e:
            print(f"  Warning: Generation failed: {e}")
            return prompt_ids
    
    for prompt in tqdm(prompts, desc=f"Generating ({'MCTS' if use_mcts else 'Sampling'})"):
        gen_ids = await generate_one(prompt)
        
        if len(gen_ids) < 2:
            continue
        
        # Repetition: check for repeated n-grams
        bigrams = [(gen_ids[i], gen_ids[i+1]) for i in range(len(gen_ids)-1)]
        if bigrams:
            unique_bigrams = len(set(bigrams))
            repetition_score = unique_bigrams / max(1, len(bigrams))
            results['repetition_scores'].append(repetition_score)
        
        # Diversity: unique tokens / total tokens
        diversity_score = len(set(gen_ids)) / max(1, len(gen_ids))
        results['diversity_scores'].append(diversity_score)
        
        # Length
        results['length_scores'].append(len(gen_ids))
    
    metrics = {
        'avg_repetition_score': np.mean(results['repetition_scores']) if results['repetition_scores'] else 0.0,
        'avg_diversity_score': np.mean(results['diversity_scores']) if results['diversity_scores'] else 0.0,
        'avg_length': np.mean(results['length_scores']) if results['length_scores'] else 0.0,
    }
    
    if use_mcts and results['state_changes']:
        metrics['avg_state_change'] = np.mean(results['state_changes'])
        metrics['std_state_change'] = np.std(results['state_changes'])
    
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--vocab-path', required=True)
    parser.add_argument('--config-path', required=True)
    parser.add_argument('--test-data', required=True)
    parser.add_argument('--dynamics-path', default=None)
    parser.add_argument('--output', default='evaluation_results.json')
    parser.add_argument('--num-test-samples', type=int, default=1000)
    parser.add_argument('--skip-generation', action='store_true')
    parser.add_argument('--compare-mcts', action='store_true',
                       help='Compare MCTS vs sampling generation')
    
    args = parser.parse_args()
    
    # Load config
    print("Loading config...")
    with open(args.config_path) as f:
        config_dict = json.load(f)
    
    config = CRSMConfig(
        vocab_size=config_dict['model']['vocab_size'],
        hidden_size=config_dict['model'].get('d_model', 128),
        d_state=config_dict['model'].get('d_state', 64),
        intermediate_size=config_dict['model'].get('d_ffn', 512),
        num_hidden_layers=config_dict['model'].get('num_layers', 2),
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = CRSM(
        vocab_size=config.vocab_size,
        d_model=config.hidden_size,
        d_state=config.d_state,
        d_ffn=config.intermediate_size,
        num_layers=config.num_hidden_layers,
    )
    
    checkpoint = torch.load(args.model_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('model_state', checkpoint))
    
    if any(k.startswith('backbone.') for k in state_dict.keys()):
        model.load_state_dict(state_dict, strict=False)
        print("✓ Loaded full CRSM checkpoint")
    else:
        model.backbone.load_state_dict(state_dict, strict=False)
        print("✓ Loaded backbone checkpoint")
    
    # Load dynamics if available
    has_dynamics = False
    if args.dynamics_path and Path(args.dynamics_path).exists():
        has_dynamics = model.load_dynamics(args.dynamics_path)
    
    model.to(device)
    model.eval()
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_vocab_file(args.vocab_path)
    print(f"✓ Loaded tokenizer with {tokenizer.vocab_size} tokens")
    
    # Prepare test data
    print("\nLoading test data...")
    test_dataset = RealTextDataset(
        data_dir=args.test_data,
        seq_len=32,
        tokenizer=tokenizer,
        vocab_size=config.vocab_size
    )
    
    test_size = min(len(test_dataset), args.num_test_samples)
    test_subset = torch.utils.data.Subset(test_dataset, range(test_size))
    test_loader = DataLoader(test_subset, batch_size=16, shuffle=False)
    
    print(f"  Test samples: {test_size}")
    
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    # 1. Perplexity
    print("\n[1/3] Computing perplexity...")
    perplexity, loss = compute_perplexity(model.backbone, test_loader, device, config.vocab_size)
    print(f"  Perplexity: {perplexity:.2f}")
    print(f"  Loss: {loss:.4f}")
    
    results = {
        'perplexity': float(perplexity),
        'loss': float(loss),
        'model_path': args.model_path,
        'vocab_size': config.vocab_size,
        'test_samples': test_size,
        'has_dynamics': has_dynamics,
    }
    
    # 2. Generation quality
    if not args.skip_generation:
        test_prompts = [
            "The model",
            "Training on",
            "State space",
            "The system",
            "Neural networks",
        ]
        
        # Sampling generation
        print("\n[2/3] Evaluating generation (sampling)...")
        sampling_metrics = asyncio.run(evaluate_generation_quality(
            model, tokenizer, test_prompts, device, use_mcts=False
        ))
        results['sampling'] = sampling_metrics
        
        print(f"  Repetition: {sampling_metrics['avg_repetition_score']:.3f}")
        print(f"  Diversity: {sampling_metrics['avg_diversity_score']:.3f}")
        print(f"  Avg Length: {sampling_metrics['avg_length']:.1f}")
        
        # MCTS generation (if requested and dynamics available)
        if args.compare_mcts and has_dynamics:
            print("\n[3/3] Evaluating generation (MCTS)...")
            mcts_metrics = asyncio.run(evaluate_generation_quality(
                model, tokenizer, test_prompts, device, use_mcts=True
            ))
            results['mcts'] = mcts_metrics
            
            print(f"  Repetition: {mcts_metrics['avg_repetition_score']:.3f}")
            print(f"  Diversity: {mcts_metrics['avg_diversity_score']:.3f}")
            print(f"  Avg Length: {mcts_metrics['avg_length']:.1f}")
            print(f"  State Change: {mcts_metrics.get('avg_state_change', 0):.6f}")
            
            # Comparison
            print("\n  Comparison (MCTS vs Sampling):")
            print(f"    Repetition: {(mcts_metrics['avg_repetition_score'] - sampling_metrics['avg_repetition_score'])*100:+.1f}%")
            print(f"    Diversity: {(mcts_metrics['avg_diversity_score'] - sampling_metrics['avg_diversity_score'])*100:+.1f}%")
        else:
            print("\n[3/3] Skipping MCTS evaluation")
    else:
        print("\n[2/3] Skipping generation evaluation")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {args.output}")
    print("="*70)


if __name__ == '__main__':
    main()