"""
Reinforcement learning fine-tuning for CRSM value head.
Uses actual generation outcomes to train better value predictions.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import asyncio
from tqdm import tqdm

from crsm.model import CRSM
from crsm.tokenizer import Tokenizer


class RLTrainer:
    """RL trainer for CRSM value head."""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.optimizer = optim.Adam(model.backbone.value_head.parameters(), lr=1e-4)
    
    async def generate_and_score(self, prompt_ids, max_length=50):
        """Generate text and compute reward."""
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        self.model.init_latent_state(batch_size=1, device=self.device)
        
        # Generate
        generated = await self.model.think_and_generate(
            prompt=prompt_tensor,
            max_length=max_length,
            use_sampling=True
        )
        
        # Compute reward (multiple criteria)
        gen_ids = generated.cpu().tolist()
        
        # 1. Length reward (prefer reasonable lengths)
        length_reward = 1.0 - abs(len(gen_ids) - 30) / 30.0
        
        # 2. Diversity reward (unique tokens / total)
        diversity_reward = len(set(gen_ids)) / len(gen_ids)
        
        # 3. Repetition penalty (penalize repeated bigrams)
        bigrams = [(gen_ids[i], gen_ids[i+1]) for i in range(len(gen_ids)-1)]
        repetition_penalty = len(set(bigrams)) / max(1, len(bigrams))
        
        # Combined reward
        reward = (length_reward + diversity_reward + repetition_penalty) / 3.0
        
        return generated, reward
    
    def train_step(self, prompts, num_samples=4):
        """Single RL training step."""
        total_loss = 0.0
        
        for prompt in prompts:
            prompt_ids = self.tokenizer.encode(prompt)
            
            # Generate multiple samples and compute rewards
            samples = []
            for _ in range(num_samples):
                gen, reward = asyncio.run(
                    self.generate_and_score(prompt_ids, max_length=30)
                )
                samples.append((gen, reward))
            
            # Compute advantage (reward - baseline)
            rewards = [r for _, r in samples]
            baseline = sum(rewards) / len(rewards)
            
            # Update value head
            for gen, reward in samples:
                # Get value prediction for prompt
                prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
                
                self.optimizer.zero_grad()
                
                with torch.no_grad():
                    _, states = self.model.backbone(prompt_tensor)
                
                # Predict value
                _, value, _ = self.model.backbone.predict_policy_value(prompt_tensor)
                
                # RL loss: minimize |V - reward|
                advantage = reward - baseline
                loss = (value - reward) ** 2
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
        
        return total_loss / (len(prompts) * num_samples)
    
    def train(self, prompts, epochs=10):
        """Train value head with RL."""
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            loss = self.train_step(prompts, num_samples=4)
            print(f"  RL Loss: {loss:.4f}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--vocab-path', required=True)
    parser.add_argument('--output-path', required=True)
    parser.add_argument('--prompts-file', required=True)
    parser.add_argument('--epochs', type=int, default=10)
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    from crsm.model import CRSMConfig
    import json
    
    config = CRSMConfig(vocab_size=1000, hidden_size=128)
    model = CRSM(
        vocab_size=config.vocab_size,
        d_model=config.hidden_size,
        d_state=64,
        d_ffn=512,
        num_layers=2
    )
    
    checkpoint = torch.load(args.model_path, map_location=device)
    model.backbone.load_state_dict(checkpoint.get('model_state_dict', checkpoint), strict=False)
    model.to(device)
    
    # Load tokenizer
    tokenizer = Tokenizer.from_vocab_file(args.vocab_path)
    
    # Load prompts
    with open(args.prompts_file) as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    print(f"Training RL on {len(prompts)} prompts...")
    
    # Train
    trainer = RLTrainer(model, tokenizer, device)
    trainer.train(prompts, epochs=args.epochs)
    
    # Save
    torch.save({
        'model_state_dict': model.backbone.state_dict(),
    }, args.output_path)
    
    print(f"\n✓ Saved RL-trained model to {args.output_path}")


if __name__ == '__main__':
    main()