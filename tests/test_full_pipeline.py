"""
Integration test for the full CRSM pipeline.
This is a smoke test - it doesn't train to convergence, just verifies everything runs.
"""

import pytest
import torch
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, '.')

from crsm.mamba_ssm import MambaModel
from crsm.latent_dynamics import LatentDynamics
from crsm.model import CRSM
from crsm.load_dynamics import load_dynamics_into_crsm, check_dynamics_quality


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


def test_latent_dynamics_module():
    """Test that LatentDynamics module works"""
    d_model = 32
    dynamics = LatentDynamics(d_model=d_model)
    
    # Test forward pass
    state = torch.randn(1, d_model)
    action_emb = torch.randn(1, d_model)
    
    delta = dynamics(state, action_emb)
    
    assert delta.shape == (1, d_model)
    assert not torch.isnan(delta).any()


def test_dynamics_distillation_minimal():
    """Test minimal dynamics distillation (few samples)"""
    device = torch.device('cpu')
    vocab_size = 100
    d_model = 32
    
    # Create small backbone
    backbone = MambaModel(
        vocab_size=vocab_size,
        d_model=d_model,
        d_state=16,
        d_ffn=64,
        num_layers=2
    ).to(device)
    
    # Create dynamics
    dynamics = LatentDynamics(d_model=d_model).to(device)
    
    # Collect a few transitions
    transitions = []
    with torch.no_grad():
        for _ in range(10):
            states = backbone.init_state(batch_size=1, device=device)
            action = torch.randint(0, vocab_size, (1,), device=device)
            _, next_states = backbone.step(action.unsqueeze(1), states)
            transitions.append((states, action.item(), next_states))
    
    # Train for 1 epoch
    optimizer = torch.optim.Adam(dynamics.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()
    
    for state, action, next_state in transitions:
        action_tensor = torch.tensor([[action]], device=device)
        action_emb = backbone.embedding(action_tensor).squeeze(1)
        
        optimizer.zero_grad()
        losses = []
        
        for s_curr, s_next in zip(state, next_state):
            if s_curr is None or s_next is None:
                continue
            
            pred_delta = dynamics(s_curr, action_emb)
            target_delta = s_next - s_curr
            loss = criterion(pred_delta, target_delta)
            losses.append(loss)
        
        if losses:
            total_loss = sum(losses) / len(losses)
            total_loss.backward()
            optimizer.step()
    
    # Verify it produces reasonable outputs
    with torch.no_grad():
        test_state = torch.randn(1, d_model)
        test_action = torch.randn(1, d_model)
        test_delta = dynamics(test_state, test_action)
        assert not torch.isnan(test_delta).any()


def test_crsm_with_dynamics():
    """Test that CRSM works with dynamics module"""
    vocab_size = 100
    d_model = 32
    
    # Create CRSM
    crsm = CRSM(
        vocab_size=vocab_size,
        d_model=d_model,
        d_state=16,
        d_ffn=64,
        num_layers=2,
        n_simulations=2  # Small for speed
    )
    
    # Add dynamics
    crsm.dynamics = LatentDynamics(d_model=d_model)
    
    # Test that reasoning uses dynamics
    device = next(crsm.parameters()).device
    seq = torch.randint(0, vocab_size, (1, 10), device=device)
    states = crsm.backbone.init_state(batch_size=1, device=device)
    
    # Run deliberation (should use dynamics if available)
    action, delta = crsm.reasoning.deliberate_sync(seq, states)
    
    assert isinstance(action, int)
    # Delta might still be None if MCTS didn't expand nodes, but shouldn't crash


def test_reasoning_with_dynamics():
    """Test that reasoning._get_next_state uses dynamics when available"""
    vocab_size = 100
    d_model = 32
    device = torch.device('cpu')
    
    # Create model with dynamics
    backbone = MambaModel(
        vocab_size=vocab_size,
        d_model=d_model,
        d_state=16,
        d_ffn=64,
        num_layers=2
    ).to(device)
    
    # Add dynamics attribute (simulating CRSM structure)
    backbone.dynamics = LatentDynamics(d_model=d_model).to(device)
    
    from crsm.reasoning import AsyncDeliberationLoop
    reasoning = AsyncDeliberationLoop(backbone, n_simulations=2)
    
    # Get initial state
    states = backbone.init_state(batch_size=1, device=device)
    action = 42
    
    # Get next state (should use dynamics)
    next_states = reasoning._get_next_state(states, action)
    
    # Verify it's a list (per-layer states)
    assert isinstance(next_states, list)
    assert len(next_states) == 2  # num_layers
    
    # Verify states changed
    for s_old, s_new in zip(states, next_states):
        if s_old is not None and s_new is not None:
            # States should be different (dynamics applied)
            assert not torch.allclose(s_old, s_new, atol=1e-6)


def test_load_dynamics_into_crsm(temp_dir):
    """Test loading dynamics checkpoint into CRSM"""
    vocab_size = 100
    d_model = 32
    
    # Create and save dynamics
    dynamics = LatentDynamics(d_model=d_model)
    dynamics_path = temp_dir / 'dynamics.pt'
    
    torch.save({
        'dynamics_state': dynamics.state_dict(),
        'config': {'d_model': d_model},
        'best_val_loss': 0.1,
        'num_samples': 100,
    }, dynamics_path)
    
    # Create CRSM
    crsm = CRSM(
        vocab_size=vocab_size,
        d_model=d_model,
        d_state=16,
        d_ffn=64,
        num_layers=2
    )
    
    # Load dynamics
    success = load_dynamics_into_crsm(crsm, dynamics_path)
    
    assert success
    assert hasattr(crsm, 'dynamics')
    assert isinstance(crsm.dynamics, LatentDynamics)


def test_check_dynamics_quality():
    """Test dynamics quality checking"""
    vocab_size = 100
    d_model = 32
    
    # Create CRSM with dynamics
    crsm = CRSM(
        vocab_size=vocab_size,
        d_model=d_model,
        d_state=16,
        d_ffn=64,
        num_layers=2
    )
    crsm.dynamics = LatentDynamics(d_model=d_model)
    
    # Check quality (with few samples for speed)
    stats = check_dynamics_quality(crsm, test_samples=5)
    
    assert stats is not None
    assert 'avg_mse' in stats
    assert 'avg_mae' in stats
    assert stats['num_comparisons'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])