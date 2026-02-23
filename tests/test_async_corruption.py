
import asyncio
import pytest
import torch
import time
from crsm.core import CRSM, CRSMConfig

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
    confidence = 1.0
    # Tuple format: (pos, delta, gen_id, confidence)
    await model.state_update_queue.put((0, poison_delta, None, confidence))

    prompt = torch.tensor([[1]])

    # Run generation
    try:
        await asyncio.wait_for(model.think_and_generate(prompt, max_length=2, use_deliberation=False), timeout=5.0)
    except asyncio.TimeoutError:
        pytest.fail("think_and_generate timed out")

    final_state_val = model.latent_state[0].item()

    # Should be 0 because queue was flushed
    assert final_state_val < 1.0, "Queue flushing failed! Poison delta was applied."

from unittest.mock import patch

@pytest.mark.asyncio
async def test_deltas_are_applied_validly_with_id():
    """
    Test that valid deltas injected during generation are applied.
    """
    model = CRSM(100)
    model.backbone = MockBackbone()
    model._ensure_async_components()

    prompt = torch.tensor([[1]])
    
    # Fix race condition: Force a known Generation ID and prevent queue flushing
    known_id = "known-uuid-1234"
    
    with patch('uuid.uuid4', return_value=known_id), \
         patch.object(model, '_flush_queues', return_value=None):
        
        # Pre-inject delta so it's waiting when generation starts
        # Inject valid delta (Target State)
        valid_delta = [torch.ones(1, 1) * 100.0]
        confidence = 1.0
        
        # Tuple format: (pos, delta, gen_id, confidence)
        await model.state_update_queue.put((0, valid_delta, known_id, confidence))
        
        # Start generation
        # It will pick up the delta on the first step (pos 0)
        await model.think_and_generate(prompt, max_length=5, use_deliberation=False)

    final_state_val = model.latent_state[0].item()

    # With Gated Injection (alpha=0.05):
    # New = (0.95 * 0) + (0.05 * 100) = 5.0
    # It should be significantly non-zero.
    assert final_state_val > 1.0, f"Valid delta was not applied! Value: {final_state_val}"
