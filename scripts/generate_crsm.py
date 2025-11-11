import argparse
import torch
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Add the project root to the path to allow crsm module imports
sys.path.insert(0, '.')

from crsm.model import CRSM, CRSMConfig 
from crsm.tokenizer import Tokenizer

def load_model_from_checkpoint(checkpoint_path: str, config_path: str, device: str) -> CRSM:
    """Loads the CRSM model from a checkpoint, applying the configuration."""
    
    # 1. Load the configuration dictionary from JSON
    print(f"Loading config from: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
        
    # 2. Perform CRITICAL KEY MAPPING AND SANITIZATION
    MODEL_KEY_MAP = {
        "d_model": "hidden_size",
        "d_ffn": "intermediate_size",
        "num_layers": "num_hidden_layers",
    }
    
    config_data = {}
    for k, v in config_dict.get('model', {}).items():
        new_k = MODEL_KEY_MAP.get(k, k)
        if new_k in ['vocab_size', 'hidden_size', 'intermediate_size', 'num_hidden_layers', 'd_state']:
            config_data[new_k] = int(v) 
        else:
            config_data[new_k] = v
        
    config_data.update(config_dict.get('reasoning', {}))

    if 'd_state' not in config_data:
        config_data['d_state'] = CRSMConfig.d_state
    
    # 3. Instantiate the CRSMConfig object
    print(f"Final CRSMConfig data: {config_data}")
    config = CRSMConfig.from_dict(config_data)
    config.autonomous_mode = True 
    
    # 4. Instantiate the CRSM model
    print("Instantiating CRSM model...")
    model = CRSM(
        vocab_size=config.vocab_size,
        d_model=config.hidden_size,
        d_state=config.d_state,
        d_ffn=config.intermediate_size,
        num_layers=config.num_hidden_layers,
        dropout=config.dropout,
        c_puct=config.c_puct,
        n_simulations=config.n_simulations,
        autonomous_mode=config.autonomous_mode
    )
    
    # FIX: Assign the config object to the model instance
    model.config = config
    
    # 5. Load the final checkpoint state with Mismatch Fix
    print(f"Loading final CRSM model from: {checkpoint_path}")
    state_dict_loaded = torch.load(checkpoint_path, map_location=device)

    if 'model_state' in state_dict_loaded:
        state_dict_loaded = state_dict_loaded['model_state']
    elif 'state_dict' in state_dict_loaded:
        state_dict_loaded = state_dict_loaded['state_dict']

    state_dict_fixed = {}
    for k, v in state_dict_loaded.items():
        if k.startswith(('embedding.', 'layers.', 'norm.', 'output.', 'value_head.')):
            state_dict_fixed['backbone.' + k] = v
        else:
            state_dict_fixed[k] = v 
            
    try:
        model.load_state_dict(state_dict_fixed, strict=True)
        print("✓ Loaded model in STRICT mode.")
    except RuntimeError as e:
        print(f"Warning: Loading in NON-STRICT mode due to missing/extra keys (This is expected if loading a partial model or an older backbone checkpoint): {e}")
        model.load_state_dict(state_dict_fixed, strict=False)

    model.to(device)
    model.eval()
    return model

def generate_text(model: CRSM, tokenizer: Tokenizer, prompt: str, max_tokens: int, temperature: float = 0.8, top_k: int = 40) -> str:
    """Generates text from a prompt using the CRSM model with MCTS deliberation."""
    device = next(model.parameters()).device
    
    # 1. Encode the prompt
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    # FIX: The pad token ID for SimpleVocab is 0
    pad_id = 0 
    
    # 2. Initialize the CRSM's latent state
    model.init_latent_state(device=device)

    # Pass the prompt through the model to initialize the state
    with torch.no_grad():
        outputs = model(input_tensor)
        
        # Robust unpacking (fixes previous ValueError)
        if len(outputs) == 3:
            _, _, final_states = outputs
        elif len(outputs) == 2:
            _, final_states = outputs
        else:
            print("\nWarning: Unexpected model output length. Assuming state is implicit...")
            # FIX: Use direct attribute access
            final_states = model.latent_state

        # FIX: Direct attribute assignment
        model.latent_state = final_states

    n_sims = getattr(model.config, 'n_simulations', 10) 
    print(f"Initial State Set. Starting generation with {n_sims} MCTS simulations per step.")
    
    output_ids = []
    
    for _ in range(max_tokens):
        
        # FIX: Use direct attribute access
        current_states = model.latent_state
        
        # FIX: Use the resolved pad_id
        dummy_input = torch.tensor([[pad_id]], dtype=torch.long, device=device)
        
        # CRITICAL FIX: The keyword argument must be 'states', not 'state'
        with torch.no_grad():
            outputs = model(dummy_input, states=current_states) 
        
        # Robust unpacking in the loop
        if len(outputs) == 3:
            logits, _, next_states = outputs
        elif len(outputs) == 2:
            logits, next_states = outputs
        else:
            print("\nError: Unexpected model output length in generation loop.")
            break
            
        # Sample from logits
        next_logits = logits[0, -1] / temperature
        
        if top_k > 0:
            v, i = torch.topk(next_logits, top_k)
            out = torch.softmax(v, dim=-1)
            idx = torch.multinomial(out, num_samples=1)
            current_token = i[idx].item()
        else:
            probs = torch.softmax(next_logits, dim=-1)
            current_token = torch.multinomial(probs, num_samples=1).item()
            
        # FIX: Direct attribute assignment
        model.latent_state = next_states
        
        # FIX: Use the resolved pad_id
        if current_token == pad_id:
            break
        output_ids.append(current_token)
        
        print(tokenizer.decode([current_token]), end=' ', flush=True)

    print("\n--- Generation Complete ---")
    return tokenizer.decode(output_ids)

def main():
    parser = argparse.ArgumentParser(description="CRSM Text Generation with MCTS Deliberation")
    parser.add_argument('--model-path', type=str, 
                        default='experiments/full_crsm/final/crsm_final.pt',
                        help="Path to the final CRSM checkpoint.")
    parser.add_argument('--config-path', type=str, 
                        default='configs/small.json',
                        help="Path to the model configuration JSON file.")
    parser.add_argument('--prompt', type=str, 
                        default='The key to solving the puzzle is',
                        help="The starting text prompt.")
    parser.add_argument('--max-tokens', type=int, default=50, help="Maximum number of tokens to generate.")
    parser.add_argument('--device', type=str, default=None, help="Device to use (e.g., 'cuda:0' or 'cpu').")
    
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        model = load_model_from_checkpoint(args.model_path, args.config_path, device)
    except FileNotFoundError as e:
        print(f"Error: Model checkpoint not found. {e}")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    tokenizer = Tokenizer() 

    print("="*60)
    print(f"PROMPT: {args.prompt}")
    print("="*60)
    
    print(f"GENERATING: ", end='')
    generate_text(model, tokenizer, args.prompt, args.max_tokens)


if __name__ == '__main__':
    main()