import torch
import torch.optim as optim
from crsm.core.crsm import CRSMModel, CRSMConfig
from crsm.tasks.arc_task import ARCTask
from crsm.training.trainer import Trainer
from crsm.core.dynamics import LatentDynamics
from crsm.core.reasoning import ARCContextEvaluator
import sys

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("crsm")

async def proof_of_correctness():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. FIXED 2x2 IDENTITY TASK
    samples = []
    for _ in range(20):
        grid = [[1, 2], [3, 4]]
        samples.append({
            "train": [{"input": grid, "output": grid}],
            "test": [{"input": grid, "output": grid}]
        })
    task = ARCTask(samples=samples, seq_len=256)
    
    # 2. MODEL CONFIG
    config = CRSMConfig(
        vocab_size=100,
        hidden_size=128,
        num_hidden_layers=2,
        d_state=32,
        intermediate_size=256,
        injection_rate=0.05,
        n_simulations=100, # Reduced for faster debug, still enough for identity
        use_context_anchoring=True
    )
    model = CRSMModel(config).to(device)
    
    # 3. FAST BACKBONE TRAINING (100 Epochs for total convergence)
    print("Training Backbone on Fixed Identity (100 Epochs)...")
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    trainer = Trainer(model, optimizer, {"batch_size": 4})
    trainer.fit(task, epochs=100)
    
    # 4. INSTANTIATE DYNAMICS
    model.crsm.dynamics = LatentDynamics(d_model=128, num_layers=2).to(device)
    model.crsm.reasoning.dynamics_model = model.crsm.dynamics
    
    # 5. MCTS EVALUATION (The Proof)
    model.eval()
    print("\nEvaluating with MCTS (Reasoning Engine Enabled, No Sampling Fallback)...")
    
    # Custom eval loop to see tokens
    train_loader, _ = task.get_dataloaders(1)
    for i, (x, y, split_idx) in enumerate(train_loader):
        if i >= 1: break
        split_idx = split_idx[0].item() if isinstance(split_idx, torch.Tensor) else split_idx
        prompt = x[:, :split_idx].to(device)
        print(f"Full sequence sample (first 30 tokens): {x[0, :30].tolist()}")
        print(f"Prompt ends at idx {split_idx}, tokens: {prompt[0, -5:].tolist()}")
        
        # Target should start from the first generated token
        target = y[0, split_idx-1:].tolist()
        # Clean up target: remove trailing zeros (padding) but KEEP internal zeros (color 0)
        # Find GRID_END (14) and cut there
        if 14 in target:
            target = target[:target.index(14)+1]
        
        output_ids = await model.crsm.think_and_generate(
            prompt, max_length=20, use_deliberation=True, deliberation_lag=0, fallback_to_sampling=False
        )
        # Generated tokens are those AFTER the prompt length
        pred = output_ids[prompt.size(1):].tolist()
        # Clean up pred: cut at first 14
        if 14 in pred:
            pred = pred[:pred.index(14)+1]
        
        print(f"Sample {i}:")
        print(f"Target: {target}")
        print(f"Pred:   {pred}")
        print(f"Match:  {pred == target}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(proof_of_correctness())