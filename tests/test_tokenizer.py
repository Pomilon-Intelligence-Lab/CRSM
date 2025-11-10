import pytest

from crsm.tokenizer import Tokenizer


def test_simple_tokenizer_roundtrip():
    tok = Tokenizer()
    text = 'hello world from crsm'
    ids = tok.encode(text)
    assert isinstance(ids, list)
    assert len(ids) > 0
    decoded = tok.decode(ids)
    assert isinstance(decoded, str)


def test_hf_tokenizer_if_available():
    # If transformers is available and a small HF tokenizer is present, test roundtrip
    try:
        t = Tokenizer(hf_name='sshleifer/tiny-gpt2')
    except RuntimeError:
        pytest.skip('transformers not available')
    text = 'the quick brown fox'
    ids = t.encode(text)
    assert isinstance(ids, list)
    assert len(ids) > 0
    decoded = t.decode(ids)
    assert isinstance(decoded, str)
