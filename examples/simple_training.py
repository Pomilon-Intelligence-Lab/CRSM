#!/usr/bin/env python3
"""
Simple example: Training a CRSM model

This example demonstrates basic model initialization, forward pass, and a minimal training loop.
For a full training pipeline, see notebooks/colab_train_crsm_2b.ipynb
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from crsm.model import CRSMConfig, CRSMModel
from crsm.tokenizer import Tokenizer


class SimpleTextDataset(IterableDataset):
    """Simple in-memory text dataset for demonstration."""
    
    def __init__(self, texts, tokenizer, seq_len=128):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        
        # Tokenize all texts and concatenate
        all_ids = []
        for text in texts:
            ids = tokenizer.encode(text)
            all_ids.extend(ids)
        
        self.ids = torch.tensor(all_ids, dtype=torch.long)
    
    def __iter__(self):
        # Yield sequences of length seq_len
        for i in range(0, len(self.ids) - self.seq_len, self.seq_len):
            seq = self.ids[i:i + self.seq_len]
            input_ids = seq[:-1]
            labels = seq[1:]
            yield {'input_ids': input_ids, 'labels': labels}


def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model configuration
    config = CRSMConfig(
        vocab_size=10000,           # Small vocab for demo
        hidden_size=512,            # Small hidden dim
        num_hidden_layers=4,        # Few layers for quick training
        intermediate_size=2048,
        max_position_embeddings=256,
        d_state=64,
    )
    print(f"\nModel config: {config}")
    
    # Initialize model and tokenizer
    model = CRSMModel(config).to(device)
    tokenizer = Tokenizer()  # Uses SimpleVocab fallback
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    
    # Sample training data
    sample_texts = [
        "The quick brown fox jumps over the lazy dog. " * 5,
        "Natural language processing is fascinating. " * 5,
        "Machine learning models learn from data. " * 5,
        "Deep neural networks are powerful tools. " * 5,
        "Training language models requires GPUs. " * 5,
    ]
    
    # Create dataset and dataloader
    dataset = SimpleTextDataset(sample_texts, tokenizer, seq_len=64)
    dataloader = DataLoader(dataset, batch_size=2)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    
    # Training loop
    print("\nStarting training...")
    model.train()
    
    total_loss = 0.0
    num_batches = 0
    
    for epoch in range(2):
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 5:  # Limit batches for demo
                break
            
            # Move to device
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            logits, _ = model(input_ids)
            
            # Compute loss
            logits_flat = logits.reshape(-1, config.vocab_size)
            labels_flat = labels.reshape(-1)
            loss = loss_fn(logits_flat, labels_flat)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if (batch_idx + 1) % 2 == 0:
                avg_loss = total_loss / num_batches
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {avg_loss:.4f}")
    
    print(f"\nTraining complete! Final loss: {total_loss / num_batches:.4f}")
    
    # Inference example
    print("\nInference example:")
    model.eval()
    
    prompt = "The quick"
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
    
    with torch.no_grad():
        logits, _ = model(input_ids)
        # Get last token logits
        next_logits = logits[0, -1, :]
        next_token_probs = F.softmax(next_logits, dim=-1)
        next_token_id = torch.argmax(next_logits).item()
    
    print(f"Prompt: '{prompt}'")
    print(f"Predicted next token ID: {next_token_id}")
    print(f"Top 5 token probabilities: {torch.topk(next_token_probs, 5).values.cpu().numpy()}")


if __name__ == "__main__":
    main()
