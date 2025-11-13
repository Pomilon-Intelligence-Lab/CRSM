"""
Evaluate CRSM model on standard metrics.
"""
import torch
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import sys

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
            
            logits, _ = model(x, states=None)
            
            loss = criterion(
                logits.reshape(-1, vocab_size),
                y.reshape(-1)
            )
            
            total_loss += loss.item()
            total_tokens += y.numel()
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    return perplexity, avg_loss


def evaluate_generation_quality(model, tokenizer, prompts, device):
    """Evaluate generation quality metrics."""
    import asyncio
    
    results = {
        'repetition_scores': [],
        'diversity_scores': [],
        'length_scores': [],
    }
    
    async def generate_one(prompt):
        prompt_ids = tokenizer.encode(prompt)
        if not prompt_ids:
            prompt_ids = [0]
        
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        model.init_latent_state(batch_size=1, device=device)
        
        try:
            generated = await model.think_and_generate(
                prompt=prompt_tensor,
                max_length=30,
                use_sampling=True
            )
            return generated.cpu().tolist()
        except Exception as e:
            print(f"  Warning: Generation failed for prompt: {e}")
            return prompt_ids  # Return prompt on failure
    
    for prompt in tqdm(prompts, desc="Generating samples"):
        gen_ids = asyncio.run(generate_one(prompt))
        
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
    
    return {
        'avg_repetition_score': np.mean(results['repetition_scores']) if results['repetition_scores'] else 0.0,
        'avg_diversity_score': np.mean(results['diversity_scores']) if results['diversity_scores'] else 0.0,
        'avg_length': np.mean(results['length_scores']) if results['length_scores'] else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--vocab-path', required=True)
    parser.add_argument('--config-path', required=True)
    parser.add_argument('--test-data', required=True, help='Test data directory')
    parser.add_argument('--output', default='evaluation_results.json')
    parser.add_argument('--num-test-samples', type=int, default=1000, help='Max test samples')
    parser.add_argument('--skip-generation', action='store_true', help='Skip generation eval')
    
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
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    if any(k.startswith('backbone.') for k in state_dict.keys()):
        model.load_state_dict(state_dict, strict=False)
        print("✓ Loaded full CRSM checkpoint")
    else:
        model.backbone.load_state_dict(state_dict, strict=False)
        print("✓ Loaded backbone checkpoint")
    
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
    
    # Limit test set size for faster evaluation
    test_size = min(len(test_dataset), args.num_test_samples)
    test_subset = torch.utils.data.Subset(test_dataset, range(test_size))
    test_loader = DataLoader(test_subset, batch_size=16, shuffle=False)
    
    print(f"  Test samples: {test_size}")
    
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    # 1. Perplexity
    print("\n[1/2] Computing perplexity...")
    perplexity, loss = compute_perplexity(model.backbone, test_loader, device, config.vocab_size)
    print(f"  Perplexity: {perplexity:.2f}")
    print(f"  Loss: {loss:.4f}")
    
    gen_metrics = {}
    
    # 2. Generation quality (optional, slow)
    if not args.skip_generation:
        print("\n[2/2] Evaluating generation quality...")
        test_prompts = [
            "The model",
            "Training on",
            "State space",
            "The system",
            "Neural networks",
        ]
        gen_metrics = evaluate_generation_quality(model, tokenizer, test_prompts, device)
        print(f"  Repetition Score: {gen_metrics['avg_repetition_score']:.3f}")
        print(f"  Diversity Score: {gen_metrics['avg_diversity_score']:.3f}")
        print(f"  Avg Length: {gen_metrics['avg_length']:.1f}")
    else:
        print("\n[2/2] Skipping generation evaluation")
    
    # Save results
    results = {
        'perplexity': float(perplexity),
        'loss': float(loss),
        **gen_metrics,
        'model_path': args.model_path,
        'vocab_size': config.vocab_size,
        'test_samples': test_size,
    }
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {args.output}")
    print("="*70)
    
    # Print summary
    print("\nSUMMARY:")
    print(f"  Model: {Path(args.model_path).name}")
    print(f"  Perplexity: {perplexity:.2f}")
    print(f"  Loss: {loss:.4f}")
    if not args.skip_generation:
        print(f"  Diversity: {gen_metrics['avg_diversity_score']:.3f}")
        print(f"  Repetition: {gen_metrics['avg_repetition_score']:.3f}")


if __name__ == '__main__':
    main()