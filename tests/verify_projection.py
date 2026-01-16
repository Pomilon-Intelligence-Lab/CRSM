import torch
import asyncio
from crsm.core import CRSM, CRSMConfig
from crsm.core.reasoning import AsyncDeliberationLoop

async def test_forward_projection():
    print("Initializing CRSM for Forward Projection Verification...")
    config = CRSMConfig(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=2,
        d_state=16
    )
    
    model = CRSM(
        vocab_size=config.vocab_size,
        d_model=config.hidden_size,
        d_state=config.d_state,
        num_layers=config.num_hidden_layers
    )
    
    # Initialize state
    state = model.init_latent_state(batch_size=1)
    
    # Run projection
    lag = 5
    print(f"Projecting state forward {lag} steps...")
    projected_state = model.reasoning.project_future_state(state, lag)
    
    # Check if state changed
    changed = False
    for s_orig, s_proj in zip(state, projected_state):
        if not torch.equal(s_orig, s_proj):
            changed = True
            break
            
    if changed:
        print("SUCCESS: State successfully changed during forward projection.")
    else:
        print("FAILURE: State remained identical after projection.")
        return

    # Verify that it's deterministic (if greedy)
    projected_state_2 = model.reasoning.project_future_state(state, lag)
    deterministic = True
    for s1, s2 in zip(projected_state, projected_state_2):
        if not torch.equal(s1, s2):
            deterministic = False
            break
            
    if deterministic:
        print("SUCCESS: Projection is deterministic (Greedy).")
    else:
        print("FAILURE: Projection is non-deterministic.")

if __name__ == "__main__":
    asyncio.run(test_forward_projection())