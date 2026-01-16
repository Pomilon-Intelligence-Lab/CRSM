import torch
import torch.nn as nn
import math
import asyncio
from crsm.core import CRSM, CRSMConfig
from crsm.core.mamba import MambaModel

def test_sparse_gating():
    print("Initializing CRSM for Sparse Gating Verification...")
    config = CRSMConfig(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=4,
        d_state=16,
        injection_rate=0.5 # High rate to make effect visible
    )
    
    model = CRSM(
        vocab_size=config.vocab_size,
        d_model=config.hidden_size,
        d_state=config.d_state,
        num_layers=config.num_hidden_layers,
        injection_rate=config.injection_rate
    )
    
    # Initialize state
    model.init_latent_state(batch_size=1)
    
    # Mock latent state (all zeros)
    # model.latent_state is list of tensors
    # Set them to 0.0
    for i in range(len(model.latent_state)):
        if model.latent_state[i] is not None:
            model.latent_state[i] = torch.zeros_like(model.latent_state[i])
        
    # Create a "Delta" (Target State) -> All Ones
    delta = []
    for s in model.latent_state:
        if s is not None:
            delta.append(torch.ones_like(s))
        else:
            delta.append(None)
    
    # Mock Confidence (Logits)
    # Layer 0: Low confidence (should preserve 0.0) -> Logit -5 (Sigmoid ~0.006)
    # Layer 3: High confidence (should pull to 1.0) -> Logit +5 (Sigmoid ~0.993)
    
    confidences = [-5.0, -1.0, 1.0, 5.0]
    
    # Expected Alpha
    # alpha = injection_rate * sigmoid(conf)
    # Layer 0: 0.5 * 0.0067 = 0.00335
    # Layer 3: 0.5 * 0.9933 = 0.49665
    
    # Update manually (simulating _apply_pending_deltas logic)
    final_scale = []
    for c in confidences:
        val = 1.0 / (1.0 + math.exp(-c))
        final_scale.append(config.injection_rate * val)
        
    print(f"Computed Scales: {final_scale}")
    
    # Apply
    model.apply_state_delta(delta, scale=final_scale)
    
    # Check results
    # New State = (1 - alpha) * Old(0) + alpha * Target(1) = alpha
    
    print("\n--- Results ---")
    for i, state in enumerate(model.latent_state):
        if state is None:
            continue
        mean_val = state.mean().item()
        expected = final_scale[i]
        print(f"Layer {i}: Mean State = {mean_val:.5f} (Expected ~{expected:.5f})")
        
        if i == 0:
            assert mean_val < 0.01, f"Layer 0 should be stable (low update), got {mean_val}"
        if i == 3:
            assert mean_val > 0.4, f"Layer 3 should be updated (high update), got {mean_val}"

    print("\nSUCCESS: Sparse Gating behavior verified.")

if __name__ == "__main__":
    test_sparse_gating()
