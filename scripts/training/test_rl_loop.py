import sys
import os
import torch
import torch.optim as optim
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from crsm.core.crsm import CRSMModel, CRSMConfig
from crsm.training.rl_trainer import RLTrainer
from crsm.tasks.arc_task import ARCTask

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Tiny model for speed
    config = CRSMConfig(vocab_size=100, hidden_size=64, num_hidden_layers=2, d_state=16) 
    model = CRSMModel(config)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    trainer_config = {'batch_size': 1, 'num_generations': 2, 'device': device}
    trainer = RLTrainer(model, optimizer, trainer_config)
    
    task = ARCTask() # Dummy data
    
    print("Testing RL Step...")
    try:
        trainer.fit(task, epochs=1, checkpoint_dir='checkpoints')
        print("RL Step Completed Successfully.")
    except Exception as e:
        print(f"RL Step Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
