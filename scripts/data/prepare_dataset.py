"""
Prepare Dataset Script
----------------------
Downloads and tokenizes datasets (FineWeb-Edu, GSM8K) into efficient binary format (uint16).
This allows for fast streaming during training with minimal memory overhead.

Usage:
    python scripts/data/prepare_dataset.py --dataset fineweb --output_dir data/fineweb
    python scripts/data/prepare_dataset.py --dataset gsm8k --output_dir data/gsm8k
"""
import os
import sys
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
import multiprocessing as mp

# Ensure the package is in the path
sys.path.insert(0, '.')

def get_tokenizer(model_name="gpt2"):
    return AutoTokenizer.from_pretrained(model_name)

def process_fineweb(example, tokenizer):
    ids = tokenizer.encode(example['text'])
    ids.append(tokenizer.eos_token_id)
    return ids

def process_gsm8k(example, tokenizer):
    # Format: Question: ... Answer: ...
    text = f"Question: {example['question']}\nAnswer: {example['answer']}"
    ids = tokenizer.encode(text)
    ids.append(tokenizer.eos_token_id)
    return ids

def write_to_bin(tokens, output_file):
    # Convert to uint16 (ensure vocab < 65535)
    # GPT2 vocab is 50257, so it fits.
    arr = np.array(tokens, dtype=np.uint16)
    with open(output_file, "wb") as f:
        f.write(arr.tobytes())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['fineweb', 'gsm8k'], help="Dataset to prepare")
    parser.add_argument('--output-dir', type=str, required=True, help="Directory to save binary files")
    parser.add_argument('--tokenizer', type=str, default="gpt2", help="Tokenizer name")
    parser.add_argument('--shard-size', type=int, default=100_000_000, help="Tokens per shard")
    parser.add_argument('--subset', type=str, default="sample-10BT", help="Subset for FineWeb (default: sample-10BT)")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tokenizer = get_tokenizer(args.tokenizer)
    print(f"Loaded tokenizer: {args.tokenizer} (vocab size: {tokenizer.vocab_size})")
    
    if tokenizer.vocab_size >= 65535:
        print("Error: Tokenizer vocab size too large for uint16 storage!")
        sys.exit(1)
        
    # Load Dataset
    print(f"Loading dataset {args.dataset}...")
    if args.dataset == 'fineweb':
        # Use streaming for large datasets
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name=args.subset, split="train", streaming=True)
        process_fn = process_fineweb
        prefix = "fineweb"
    elif args.dataset == 'gsm8k':
        ds = load_dataset("gsm8k", "main", split="train", streaming=True)
        process_fn = process_gsm8k
        prefix = "gsm8k"
        
    token_buffer = []
    shard_idx = 0
    total_tokens = 0
    
    print(f"Processing and tokenizing to {output_dir}...")
    
    pbar = tqdm(desc="Tokens", unit="tok")
    
    for example in ds:
        tokens = process_fn(example, tokenizer)
        token_buffer.extend(tokens)
        
        # Update pbar occasionally
        if len(token_buffer) % 1000 == 0:
             pbar.update(len(tokens)) # Approximate update
        
        if len(token_buffer) >= args.shard_size:
            # Write shard
            filename = output_dir / f"{prefix}_train_{shard_idx:04d}.bin"
            write_to_bin(token_buffer, filename)
            
            total_tokens += len(token_buffer)
            print(f"Saved {filename} ({len(token_buffer)} tokens)")
            
            token_buffer = []
            shard_idx += 1
            
            # For demonstration/testing purposes, we might stop early if running in CI/Test env
            # But this is a script, so let it run.
            
    # Write remaining
    if token_buffer:
        filename = output_dir / f"{prefix}_train_{shard_idx:04d}.bin"
        write_to_bin(token_buffer, filename)
        total_tokens += len(token_buffer)
        print(f"Saved {filename} ({len(token_buffer)} tokens)")
        
    print(f"Done. Total tokens: {total_tokens}")

if __name__ == "__main__":
    main()
