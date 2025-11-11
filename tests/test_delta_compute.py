import torch
import pytest

# Assuming the 'crsm' package is in the root or installed
from crsm.mamba_ssm import MambaModel
from crsm.reasoning import AsyncDeliberationLoop

# --- Pytest Fixture ---

@pytest.fixture
def model_and_reasoning():
    """
    Provides an initialized MambaModel and its corresponding
    AsyncDeliberationLoop reasoning module.
    """
    model = MambaModel(
        vocab_size=100,
        d_model=32,
        d_state=16,
        num_layers=2  # Keep small for testing
    )
    reasoning = AsyncDeliberationLoop(model, n_simulations=10)
    return model, reasoning

# --- Pytest Test Functions ---

def test_mamba_model_has_init_state(model_and_reasoning):
    """
    Verifies that the MambaModel class has the 'init_state' method,
    which is critical for the reasoning loop.
    """
    model, _ = model_and_reasoning
    assert hasattr(model, 'init_state'), \
        "MambaModel is missing the 'init_state' method. " \
        "This is required for the MCTS loop."

def test_init_state_returns_list(model_and_reasoning):
    """
    Verifies that model.init_state() returns a list of states,
    which is the expected format for delta computation.
    """
    model, _ = model_and_reasoning
    states = model.init_state(batch_size=1)
    
    assert isinstance(states, list), \
        "model.init_state() should return a list (one entry per layer)"
    assert len(states) == 2, \
        f"Expected {2} layer states, but got {len(states)}"

def test_get_next_state_preserves_list_structure(model_and_reasoning):
    """
    Verifies that the reasoning._get_next_state method correctly
    processes a list of states and returns a new list of states.
    This was a key check in the original script's debug block.
    """
    model, reasoning = model_and_reasoning
    
    # GIVEN a list of initial states
    initial_states = model.init_state(batch_size=1)
    assert isinstance(initial_states, list)
    
    # WHEN we get the next state
    action = 10  # An arbitrary action token
    next_state = reasoning._get_next_state(initial_states, action)
    
    # THEN the new state should also be a list of the same length
    assert isinstance(next_state, list), \
        "_get_next_state did not return a list when given a list"
    assert len(next_state) == len(initial_states), \
        "The new state list has a different length than the original"

def test_deliberate_sync_produces_non_none_delta(model_and_reasoning):
    """
    The main integration test: Verifies that a full run of
    deliberate_sync produces a non-None delta.
    """
    model, reasoning = model_and_reasoning
    
    # GIVEN an input sequence and initialized states
    seq = torch.randint(0, 100, (1, 10))
    states = model.init_state(batch_size=1)
    assert isinstance(states, list)  # Prerequisite check
    
    # WHEN we run synchronous deliberation
    action, delta = reasoning.deliberate_sync(seq, states)
    
    # THEN the action should be an int and delta should be a non-None list
    assert isinstance(action, int)
    assert delta is not None, "deliberate_sync returned a None delta"
    
    assert isinstance(delta, list), "Delta was not a list"
    assert len(delta) == len(states), "Delta list has incorrect number of layers"