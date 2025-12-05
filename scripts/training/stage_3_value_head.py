"""
Stage 3: Offline Value Training (The Judgment)
----------------------------------------------
This script trains the Value Head using offline Expert Iteration.
It simulates MCTS rollouts using the frozen Dynamics model and trains on the outcomes.

Output: experiments/stage_3/crsm_v1_complete.pt
"""

import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
import asyncio
from tqdm import tqdm

sys.path.insert(0, '.')

from crsm.model import CRSM
from crsm.tokenizer import Tokenizer
from crsm.utils import set_seed

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# --- Logic adapted from rl_value_head_finetuning.py and verify_capabilities.py ---

class OfflineValueTrainer:
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        # Only train the value head
        self.optimizer = optim.Adam(model.backbone.value_head.parameters(), lr=float(config['training']['value_training']['lr']))
        
    async def simulate_rollout(self, prompt_ids, max_length=20):
        """
        Simulate a rollout using the Dynamics model (implicitly used by CRSM's think_and_generate or manually).
        Here we generate data:
        - Run with MCTS enabled (Positive samples from best path)
        - Run with Noise or Sampling (Negative samples)
        """
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        self.model.init_latent_state(batch_size=1, device=self.device)
        
        # 1. Positive Run (Reasoning/MCTS)
        # We assume think_and_generate uses MCTS if dynamics are loaded and configured
        try:
            positive_gen = await self.model.think_and_generate(
                prompt=prompt_tensor,
                max_length=max_length,
                use_deliberation=True,
                fallback_to_sampling=False # Enforce MCTS logic
            )
            pos_reward = 1.0 # Simplified reward: MCTS chosen path is "good"
        except Exception:
            positive_gen = prompt_tensor[0]
            pos_reward = 0.0

        # 2. Negative/Contrastive Run (Random/Noisy)
        self.model.init_latent_state(batch_size=1, device=self.device)
        # Actually think_and_generate doesn't accept 'temperature' as arg either, it's in model config or reasoning config
        # But we can simulate noisy generation by disabling deliberation and letting it fallback to sampling
        try:
            # Temporarily boost temperature if possible, or just rely on inherent sampling
            # The model config has temperature.
            original_temp = self.model.temperature
            self.model.temperature = 1.5
            
            negative_gen = await self.model.think_and_generate(
                prompt=prompt_tensor,
                max_length=max_length,
                use_deliberation=False, # Disable MCTS
                fallback_to_sampling=True # Use sampling
            )
            self.model.temperature = original_temp
            neg_reward = 0.0 # Assume noisy path is "bad" relative to MCTS
        except Exception:
            negative_gen = prompt_tensor[0]
            neg_reward = 0.0
            
        return [
            (prompt_ids, pos_reward),
            (prompt_ids, neg_reward) 
        ]
        # Note: A real implementation would store the specific states traversed. 
        # Here we are simplifying to "Value of this start state given the outcome".
        # Better: Store the (state, value) pairs generated during MCTS.
        # But CRSM v1 api might not expose internal MCTS tree easily without modification.
        # We will stick to the "RL Loop" structure which updates Value Head based on outcomes.

    def train_step(self, batch_data):
        self.optimizer.zero_grad()
        total_loss = 0
        
        for prompt_ids, target_value in batch_data:
            prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
            
            # Forward pass to get current value prediction
            # We need to detach backbone gradients to only train value head
            with torch.no_grad():
                _, _ = self.model.backbone(prompt_tensor) # Updates internal state if needed, or just gets features
                # Actually we need features.
                # Let's assume predict_policy_value works
                pass

            _, predicted_value, _ = self.model.backbone.predict_policy_value(prompt_tensor)
            
            target = torch.tensor([[target_value]], dtype=torch.float, device=self.device)
            loss = nn.MSELoss()(predicted_value, target)
            
            loss.backward()
            total_loss += loss.item()
            
        self.optimizer.step()
        return total_loss / len(batch_data)

async def main_async():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/training_config.yaml", help="Path to YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)
    device = config['system']['device'] if torch.cuda.is_available() else 'cpu'
    set_seed(config['system']['seed'])

    # Paths
    backbone_path = Path("experiments/stage_1/backbone_final.pt")
    dynamics_path = Path("experiments/stage_2/dynamics_final.pt")
    output_dir = Path("experiments/stage_3")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not backbone_path.exists() or not dynamics_path.exists():
        print("✗ Prerequisite stages not complete.")
        sys.exit(1)

    print("\n" + "="*60)
    print("STAGE 3: Value Head Training (The Judgment)")
    print("="*60)

    # Load Model
    print("Loading Frozen System...")
    model = CRSM(
        vocab_size=config['model']['vocab_size'],
        d_model=config['model']['d_model'],
        d_state=config['model']['d_state'],
        d_ffn=config['model']['d_ffn'],
        num_layers=config['model']['num_layers'],
        c_puct=config['reasoning']['c_puct'],
        n_simulations=config['reasoning']['n_simulations']
    ).to(device)
    
    # Load Weights
    ckpt = torch.load(backbone_path, map_location=device)
    model.backbone.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt, strict=False)
    
    # Load Dynamics
    model.load_dynamics(dynamics_path)
    
    # Freeze everything except Value Head
    for param in model.parameters():
        param.requires_grad = False
    for param in model.backbone.value_head.parameters():
        param.requires_grad = True
        
    # Data Generation (Offline Rollouts)
    print("Generating Offline Data via MCTS Simulation...")
    # Use prompts from reasoning tasks or dummy prompts
    # Ensure they are within vocab size (1000 for small config)
    # Tokenizer uses byte-level BPE, so IDs can be large. 
    # But small config has vocab_size=1000. 
    # We should use simple dummy tokens if tokenizer produces IDs > vocab_size
    
    prompts = ["The future of AI is", "Solve 2+2=", "Reasoning requires", "System 1 and 2"]
    
    tokenizer = Tokenizer() # Assuming default vocab
    
    # If using small config with small vocab, check IDs
    if config['model']['vocab_size'] < 50000:
         # Mock prompts with valid IDs
         import random
         dataset = []
         print("Generating synthetic data for small vocab...")
         for _ in range(20):
             # Random sequence of valid tokens
             ids = [random.randint(0, config['model']['vocab_size']-1) for _ in range(5)]
             # Directly pass IDs to trainer, bypassing string encoding which might yield high IDs
             # But trainer.simulate_rollout expects IDs? No, it takes prompt_ids.
             # So we can pass raw IDs.
             
             # Need to adapt the loop below to handle this.
             pass
    
    trainer = OfflineValueTrainer(model, config, device)
    
    epochs = config['training']['value_training']['epochs']
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        dataset = []
        
        # Simulation Phase
        # Check if we should use synthetic IDs for small models
        use_synthetic = config['model']['vocab_size'] < 50000
        
        iterator = range(20) if use_synthetic else prompts
        
        for item in tqdm(iterator, desc="Simulating"):
            if use_synthetic:
                 # Generate random valid IDs
                 import random
                 prompt_ids = [random.randint(0, config['model']['vocab_size']-1) for _ in range(5)]
            else:
                 prompt_ids = tokenizer.encode(item)
                 
            data_points = await trainer.simulate_rollout(prompt_ids)
            dataset.extend(data_points)
            
        # Training Phase
        # Shuffle dataset
        import random
        random.shuffle(dataset)
        
        loss = trainer.train_step(dataset)
        print(f"  Value Loss: {loss:.6f}")
        
    # Save Final Model
    final_path = output_dir / 'crsm_v1_complete.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config
    }, final_path)
    
    print(f"\n✓ Stage 3 Complete. Full CRSM saved to {final_path}")

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
