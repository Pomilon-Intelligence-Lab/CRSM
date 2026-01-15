import torch

from crsm.mamba_ssm import MambaModel
from crsm.reasoning import AsyncDeliberationLoop, MCTSNode


def test_mcts_expansion_from_latent_state():
    # small model for fast test
    m = MambaModel(vocab_size=64, d_model=32, d_state=16, d_ffn=64, num_layers=3, dropout=0.0)
    loop = AsyncDeliberationLoop(mamba_model=m, c_puct=1.0, n_simulations=2)

    # initialize canonical latent state (per-layer list)
    latent = m.init_state(batch_size=1, device=torch.device('cpu'))
    assert isinstance(latent, list)

    # Create root node with latent-state list (using state_cache)
    root = MCTSNode(state_cache=latent, prior_p=1.0, children={}, parent=None)
    
    # Verify it holds the state
    assert root.state_cache is latent

    # Get policy logits and value from latent states
    logits, values, _ = m.predict_from_states(latent)
    last_logits = logits[0, -1]
    
    # values is now a list of tensors for Multi-Headed Critic
    value_list = [v.item() for v in values]

    # Expand root; children states should be produced via model.step inside _get_next_state
    loop.expand_node(root, last_logits, value_list)

    assert len(root.children) > 0

    # At least one child should have per-layer state list and differ from parent
    some_diff = False
    for action, child in root.children.items():
        # Reconstruct state
        child_state = loop._reconstruct_state(child, latent)
        assert isinstance(child_state, list)
        
        # compare each layer: at least one layer must change
        diffs = 0
        for p_layer, c_layer in zip(latent, child_state):
            if p_layer is None or c_layer is None:
                continue
            # shapes must match
            assert p_layer.shape == c_layer.shape
            if not torch.allclose(p_layer, c_layer):
                diffs += 1
        if diffs > 0:
            some_diff = True

    assert some_diff, "No child latent state differed from the parent; step() may not have been used"
