import torch

from crsm.mamba_ssm import MambaModel
from crsm.reasoning import AsyncDeliberationLoop


def test_deliberate_sync_returns_action():
    m = MambaModel(vocab_size=32, d_model=16, d_state=8, d_ffn=32, num_layers=1, dropout=0.0)
    loop = AsyncDeliberationLoop(mamba_model=m, c_puct=1.0, n_simulations=2)
    state = torch.tensor([1, 2, 3], dtype=torch.long)
    action, delta = loop.deliberate_sync(None, state)
    assert isinstance(action, int)
    assert delta is None
