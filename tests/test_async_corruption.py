
import asyncio
import pytest
import torch
import time
from crsm.model import CRSM, CRSMConfig

class MockBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cpu')

    def init_state(self, batch_size=1, device=None):
        return [torch.zeros(1, 1)] # Simple state

    def forward(self, x, states=None):
        # Always predict token 1
        logits = torch.zeros(1, 1, 100)
        logits[0, 0, 1] = 10.0
        return logits, states

    def embedding(self, x):
        return torch.randn(1, 1, 16) # Dummy embedding

@pytest.mark.asyncio
async def test_queue_flushing_prevents_corruption():
    """
    Test that deltas present before generation starts are flushed.
    """
    model = CRSM(100)
    model.backbone = MockBackbone()
    model._ensure_async_components()

    # Inject poison delta (no gen_id)
    poison_delta = [torch.ones(1, 1) * 9999.0]
    await model.state_update_queue.put((0, poison_delta))

    prompt = torch.tensor([[1]])

    # Run generation
    try:
        await asyncio.wait_for(model.think_and_generate(prompt, max_length=2, use_deliberation=False), timeout=5.0)
    except asyncio.TimeoutError:
        pytest.fail("think_and_generate timed out")

    final_state_val = model.latent_state[0].item()

    # Should be 0 because queue was flushed
    assert final_state_val < 1.0, "Queue flushing failed! Poison delta was applied."

@pytest.mark.asyncio
async def test_deltas_are_applied_validly_with_id():
    """
    Test that valid deltas injected during generation are applied.
    """
    model = CRSM(100)
    model.backbone = MockBackbone()
    model._ensure_async_components()

    prompt = torch.tensor([[1]])

    # Start generation in background task
    task = asyncio.create_task(model.think_and_generate(prompt, max_length=5, use_deliberation=False))

    # Wait for it to start and flush queues
    await asyncio.sleep(0.1)

    # Get the active generation ID
    gen_id = model.current_generation_id
    assert gen_id is not None, "Generation ID was not set!"

    # Inject valid delta
    valid_delta = [torch.ones(1, 1) * 100.0]
    # We inject at position 0. By now step might be 0 or 1.
    # We want to ensure it is applied.
    await model.state_update_queue.put((0, valid_delta, gen_id))

    try:
        await asyncio.wait_for(task, timeout=5.0)
    except asyncio.TimeoutError:
        pytest.fail("think_and_generate task timed out")

    final_state_val = model.latent_state[0].item()

    # Should be > 50 (it might decay if step advanced, but 100 * 0.9^k is still high)
    assert final_state_val > 50.0, "Valid delta was not applied!"

@pytest.mark.asyncio
async def test_lag_decay():
    """
    Test that stale deltas are decayed.
    """
    model = CRSM(100)
    model.backbone = MockBackbone()
    model._ensure_async_components()

    model.current_generation_id = "test_gen"
    model.latent_state = [torch.zeros(1, 1)]

    delta = [torch.ones(1, 1) * 100.0]
    await model.state_update_queue.put((0, delta, "test_gen"))

    # Apply at step 2 (Lag = 2)
    # Decay = 0.9^2 = 0.81
    # Expected value = 81.0
    async with model._state_lock:
        await model._apply_pending_deltas(current_step=2)

    val = model.latent_state[0].item()
    assert abs(val - 81.0) < 0.1, f"Decay logic incorrect. Expected 81.0, got {val}"

    # Test rejection logic (Lag > max_lag)
    model.max_lag = 5
    await model.state_update_queue.put((0, delta, "test_gen"))

    # Apply at step 10 (Lag = 10 > 5)
    async with model._state_lock:
        await model._apply_pending_deltas(current_step=10)

    val2 = model.latent_state[0].item()
    # Should remain 81.0 (no new addition)
    assert abs(val2 - 81.0) < 0.1, "Stale delta was not rejected!"
