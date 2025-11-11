import torch
import asyncio
import os
import sys
import time
import json
from typing import Dict, Any
from pathlib import Path

# ----------------------------------------------------------------------
# FIX: Add the project root to sys.path
# This allows the script in 'scripts/' to find the 'crsm' package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
# ----------------------------------------------------------------------

# Import the necessary classes. Fatal exit if core modules are missing.
try:
    from crsm.model import CRSM
    from crsm.latent_dynamics import LatentDynamics
    from crsm.tokenizer import Tokenizer
except ImportError as e:
    # If any core module is missing, exit
    print(f"FATAL IMPORT ERROR: {e}. Please ensure crsm/ is in your project root and all modules are present.")
    sys.exit(1)

# --- Configuration ---
MODEL_PATH = "experiments/full_crsm/crsm_with_dynamics.pt"
CONFIG_PATH = "configs/small.json"

# --- Utility Function to Load Model Correctly ---

def load_crsm_from_config(model_path: str, config_path: str) -> CRSM:
    """Correctly loads the CRSM model by reading config and loading state_dict."""
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}.")

    with open(config_path, 'r') as f:
        full_config: Dict = json.load(f)

    # 1. Extract and merge parameters needed for CRSM.__init__
    params = full_config.get("model", {})
    params.update(full_config.get("reasoning", {}))
    params['autonomous_mode'] = True 

    # 2. Instantiate the model using UNPACKED keyword arguments
    model = CRSM(**params)
    
    # 3. Load the weights
    loaded_checkpoint: Any = torch.load(model_path, map_location='cpu')
    
    # Extract state_dict from wrapper if needed
    if isinstance(loaded_checkpoint, dict) and 'model_state' in loaded_checkpoint:
        print("Checkpoint is wrapped. Extracting 'model_state'...")
        state_dict = loaded_checkpoint['model_state']
    else:
        state_dict = loaded_checkpoint
        
    # 4. Separate keys for the main model and the dynamics model
    model_state_dict = {}
    dynamics_state_dict = {}
    
    for k, v in state_dict.items():
        if k.startswith('dynamics.'):
            # Keys like "dynamics.net.0.weight" are for LatentDynamics. Remove "dynamics." prefix.
            dynamics_state_dict[k.replace('dynamics.', '', 1)] = v 
        else:
            model_state_dict[k] = v

    # 5. Load the cleaned main model state_dict
    model.load_state_dict(model_state_dict, strict=False) 
    
    # 6. Manually load the dynamics model into the reasoning component
    d_model = params.get('d_model', 128)
    # Action embedding dimension is typically d_model
    action_dim = d_model 
    
    dynamics_model = LatentDynamics(d_model=d_model, action_dim=action_dim)
    dynamics_model.load_state_dict(dynamics_state_dict)
    
    # Attach the dynamics model to the MCTS loop
    model.reasoning.dynamics_model = dynamics_model
    
    print("✓ Model and dynamics weights loaded successfully.")
    return model

# --- The Test Function ---

async def test_autonomous_generation():
    print(f"Loading CRSM model from: {MODEL_PATH}")
    
    try:
        model = load_crsm_from_config(MODEL_PATH, CONFIG_PATH)
    except Exception as e:
        print(f"FATAL ERROR during model loading: {e}")
        return

    # 2. Set reasoning parameters
    model.reasoning.n_simulations = 50 
    model.autonomous_mode = True       
    
    # 3. Initialize state and tokenizer
    device = torch.device("cpu")
    model.to(device)
    model.init_latent_state(batch_size=1, device=device) 
    
    tokenizer = Tokenizer()

    prompt_text = "The most important next step for this CRSM project is to"
    # FIX: Convert the list output of encode() to a tensor before unsqueeze(0)
    prompt = torch.tensor(tokenizer.encode(prompt_text), dtype=torch.long).unsqueeze(0).to(device)
    
    print("\n==============================================")
    print("Starting Autonomous Reasoning Test")
    print(f"Prompt: \"{prompt_text}\"")
    print(f"N_Simulations: {model.reasoning.n_simulations}")
    print("==============================================")
    
    # 4. START THE AUTONOMOUS THINKING LOOP
    model.start_autonomous_thinking()
    
    print("✓ Autonomous thinking task started in background. Waiting 0.5s...")
    await asyncio.sleep(0.5) 
    
    # 5. Run generation
    print("Running token generation loop (MCTS runs in parallel)...")
    
    start_time = time.time()
    generated_tokens = await model.think_and_generate(
        prompt, 
        max_length=30
    )
    end_time = time.time()
    
    # 6. Stop the autonomous thinking
    model.stop_autonomous_thinking()

    # FINAL FIX: Robustly convert the generated tensor output to a list of IDs for decoding
    # Squeeze() handles batch (0) and any extraneous length dimensions.
    sequence_ids = generated_tokens.cpu().squeeze().tolist()
    
    # Handle the case where the output was a scalar (e.g., L=1, .tolist() returns int)
    if isinstance(sequence_ids, int):
        sequence_ids = [sequence_ids]
        
    decoded_output = tokenizer.decode(sequence_ids)
    
    print("\n==============================================")
    print("RESULTS")
    print(f"Total time: {end_time - start_time:.3f}s")
    print(f"Final output: {decoded_output}")
    print("==============================================")
    print("✅ Autonomous generation complete. The CRSM architecture is validated!")

if __name__ == '__main__':
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(test_autonomous_generation())