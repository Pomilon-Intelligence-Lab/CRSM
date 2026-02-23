import asyncio
import torch
import sys
import os
import uuid

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from crsm.core import CRSMConfig, CRSMModel, CRSM

# Ensure device
device = torch.device("cpu") # sufficient for logic test

async def run_lag_tests():
    print(f"\n{'='*60}")
    print(f"VERIFYING LAG-AWARE DELTA CORRECTION")
    print(f"{'='*60}")
    
    # 1. Setup
    config = CRSMConfig(
        vocab_size=100,
        hidden_size=16,
        num_hidden_layers=1,
        injection_rate=0.1,
        delta_decay=0.9,
        max_lag=5
    )
    # We instantiate CRSM directly to test internal methods
    model = CRSM(
        vocab_size=config.vocab_size,
        d_model=config.hidden_size,
        num_layers=config.num_hidden_layers,
        injection_rate=config.injection_rate,
        delta_decay=config.delta_decay,
        max_lag=config.max_lag
    )
    # Init async components
    model._ensure_async_components()
    model.current_generation_id = str(uuid.uuid4())
    
    # Helper to reset state
    def reset_state():
        # Init to Zeros
        model.latent_state = [torch.zeros(1, config.hidden_size)]
    
    # Helper to create target (Ones)
    def get_target():
        return [torch.ones(1, config.hidden_size)]
    
    # =================================================================
    # TEST 1: Zero Lag (Synchronous equivalent)
    # Expected: alpha = 0.1 * 1.0 (conf) * 1.0 (lag) = 0.1
    # State: 0 -> 0.1
    # =================================================================
    reset_state()
    target = get_target()
    
    # Queue: (pos, delta, gen_id, confidence, captured_step)
    # Use 20.0 logit to get sigmoid(20) ~ 1.0 confidence
    await model.state_update_queue.put((0, target, model.current_generation_id, 20.0, 10))
    
    # Process at step 10 (Lag 0)
    await model._apply_pending_deltas(current_step=10)
    
    # Check buffer instead of state
    assert 0 in model._targeted_deltas
    delta, scales = model._targeted_deltas.pop(0)
    val = scales[0] if isinstance(scales, list) else scales
    print(f"Test 1 (Lag 0): Exp=0.1000, Got={val:.4f} -> {'PASS' if abs(val-0.1)<1e-4 else 'FAIL'}")

    # =================================================================
    # TEST 2: Lag 1 (Decay)
    # Expected: alpha = 0.1 * 1.0 * 0.9^1 = 0.09
    # State: 0 -> 0.09
    # =================================================================
    reset_state()
    target = get_target()
    
    await model.state_update_queue.put((0, target, model.current_generation_id, 20.0, 10))
    
    # Process at step 11 (Lag 1)
    await model._apply_pending_deltas(current_step=11)
    
    assert 0 in model._targeted_deltas
    delta, scales = model._targeted_deltas.pop(0)
    val = scales[0] if isinstance(scales, list) else scales
    print(f"Test 2 (Lag 1): Exp=0.0900, Got={val:.4f} -> {'PASS' if abs(val-0.09)<1e-4 else 'FAIL'}")

    # =================================================================
    # TEST 3: Lag 5 (Max Lag)
    # Expected: alpha = 0.1 * 0.9^5 = 0.1 * 0.59049 = 0.0590
    # State: 0 -> 0.0590
    # =================================================================
    reset_state()
    target = get_target()
    
    await model.state_update_queue.put((0, target, model.current_generation_id, 20.0, 10))
    
    # Process at step 15 (Lag 5)
    await model._apply_pending_deltas(current_step=15)
    
    assert 0 in model._targeted_deltas
    delta, scales = model._targeted_deltas.pop(0)
    val = scales[0] if isinstance(scales, list) else scales
    expected = 0.1 * (0.9**5)
    print(f"Test 3 (Lag 5): Exp={expected:.4f}, Got={val:.4f} -> {'PASS' if abs(val-expected)<1e-4 else 'FAIL'}")

    # =================================================================
    # TEST 4: Lag 6 (Over Limit)
    # Expected: Pruned -> 0.0
    # =================================================================
    reset_state()
    target = get_target()
    
    await model.state_update_queue.put((0, target, model.current_generation_id, 20.0, 10))
    
    # Process at step 16 (Lag 6)
    await model._apply_pending_deltas(current_step=16)
    
    # Should NOT be in buffer
    val = 0.0
    if 0 in model._targeted_deltas:
        _, scales = model._targeted_deltas.pop(0)
        val = scales[0] if isinstance(scales, list) else scales
        
    print(f"Test 4 (Lag 6): Exp=0.0000, Got={val:.4f} -> {'PASS' if abs(val-0.0)<1e-4 else 'FAIL'}")

    # =================================================================
    # TEST 5: Wrong Generation ID
    # Expected: Rejected -> 0.0
    # =================================================================
    reset_state()
    target = get_target()
    
    wrong_id = str(uuid.uuid4())
    await model.state_update_queue.put((0, target, wrong_id, 1.0, 10))
    
    # Process at step 10
    await model._apply_pending_deltas(current_step=10)
    
    val = 0.0
    if 0 in model._targeted_deltas:
        _, scales = model._targeted_deltas.pop(0)
        val = scales[0] if isinstance(scales, list) else scales
        
    print(f"Test 5 (Bad GenID): Exp=0.0000, Got={val:.4f} -> {'PASS' if abs(val-0.0)<1e-4 else 'FAIL'}")

if __name__ == "__main__":
    asyncio.run(run_lag_tests())
