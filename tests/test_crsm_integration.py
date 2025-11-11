import torch
import pytest
import pytest_asyncio

# Assuming the 'crsm' package is in the root or installed
from crsm.model import CRSM

# --- Pytest Fixture ---

@pytest.fixture
def small_crsm_model():
    """
    Provides a small, initialized CRSM model for testing.
    """
    # Create small CRSM model
    model = CRSM(
        vocab_size=100,
        d_model=32,
        d_state=16,
        d_ffn=128,
        num_layers=2,
        c_puct=1.0,
        n_simulations=5  # Keep small for fast testing
    )
    # Initialize latent state
    model.init_latent_state(batch_size=1)
    return model

# --- Pytest Test Functions ---

@pytest.mark.asyncio
async def test_think_and_generate_modifies_state(small_crsm_model):
    """
    Test 1: Verifies that think_and_generate (with deltas)
    actually modifies the model's canonical latent state.
    """
    model = small_crsm_model
    
    # Store a deep copy of the original state
    state_before = [s.clone() if s is not None else None for s in model.latent_state]
    prompt = torch.randint(0, 100, (1, 5))

    # Act
    output_with_deltas = await model.think_and_generate(prompt, max_length=10)
    state_after = model.latent_state

    # Assert
    assert output_with_deltas.shape[0] == 15  # 5 prompt + 10 generated
    assert state_after is not None

    state_changed = False
    for s_before, s_after in zip(state_before, state_after):
        if s_before is not None and s_after is not None:
            # Check if the sum of absolute differences is non-trivial
            diff = torch.abs(s_after - s_before).sum().item()
            if diff > 1e-6:
                state_changed = True
                break
    
    assert state_changed is True, "Latent state was not modified by think_and_generate"

@pytest.mark.asyncio
async def test_apply_state_delta_works(small_crsm_model):
    """
    Test 2: Verifies that the apply_state_delta method
    correctly modifies the latent state.
    """
    model = small_crsm_model
    
    # Store a deep copy of the original state
    initial_state = [s.clone() for s in model.latent_state]
    
    # Create a test delta
    test_delta = [torch.randn_like(s) * 0.1 for s in model.latent_state]
    
    # Act
    model.apply_state_delta(test_delta)
    
    # Assert
    delta_applied = False
    for s_init, s_after, delta in zip(initial_state, model.latent_state, test_delta):
        # The difference should be exactly equal to the applied delta
        diff = torch.abs(s_after - s_init).sum().item()
        expected_diff = torch.abs(delta).sum().item()
        
        # Use approx for floating point comparison
        assert diff == pytest.approx(expected_diff), f"State change {diff} does not match delta {expected_diff}"
        
        if diff > 1e-6:
            delta_applied = True
            
    assert delta_applied is True, "apply_state_delta did not modify the state"

@pytest.mark.asyncio
async def test_deliberation_produces_deltas(small_crsm_model):
    """
    Test 3: Verifies that the reasoning.deliberate method
    successfully computes and returns a non-None state delta.
    """
    model = small_crsm_model
    
    seq = torch.randint(0, 100, (1, 8))
    states = model.backbone.init_state(batch_size=1)
    
    # Act
    action, delta = await model.reasoning.deliberate(seq, states)
    
    # Assert
    assert isinstance(action, int)
    assert delta is not None, "Deliberation returned a None delta"
    
    delta_has_magnitude = False
    for i, d in enumerate(delta):
        if d is not None:
            magnitude = torch.abs(d).sum().item()
            # This can be zero if the model is untrained, but we
            # can at least assert that *if* it's non-zero, it's a tensor
            assert isinstance(d, torch.Tensor)
            if magnitude > 0:
                delta_has_magnitude = True
    
    # This is a weaker assertion, as an untrained model might
    # (correctly) produce a zero delta. The main test is that delta is not None.
    # If this fails, it's not critical, but good to know.
    assert delta_has_magnitude is True, "Delta was produced, but all layer deltas were zero"