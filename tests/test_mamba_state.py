import torch

from crsm.core.mamba import MambaModel


def test_init_state_and_step():
    m = MambaModel(vocab_size=32, d_model=16, d_state=8, d_ffn=32, num_layers=2, dropout=0.0)
    states = m.init_state(batch_size=1, device=torch.device('cpu'))
    assert isinstance(states, list)
    assert len(states) == len(m.layers)

    # single-step with a single token
    logits, new_states = m.step(torch.tensor([[1]], dtype=torch.long), states)
    assert logits.shape[0] == 1
    assert isinstance(new_states, list)
    assert len(new_states) == len(m.layers)
