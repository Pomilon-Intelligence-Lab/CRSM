import torch

from crsm.core import CRSM


def test_crsm_init_latent_and_apply():
    crsm = CRSM(vocab_size=32, d_model=16, d_state=8, d_ffn=32, num_layers=2, dropout=0.0)
    latent = crsm.init_latent_state(batch_size=1, device=torch.device('cpu'))
    assert latent is not None
    assert isinstance(latent, list)

    # create a zero delta matching latent structure
    delta = []
    for s in latent:
        if s is None:
            delta.append(None)
        else:
            delta.append(torch.zeros_like(s))

    # apply delta and ensure no error and shapes preserved
    crsm.apply_state_delta(delta)
    for s in crsm.latent_state:
        if s is not None:
            assert s.shape[0] == 1
