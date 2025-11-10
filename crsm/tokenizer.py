"""Tokenizer wrapper that prefers Hugging Face AutoTokenizer but falls back to a simple whitespace vocab.

Provides a small, consistent API: Tokenizer.encode(text) -> List[int], Tokenizer.decode(ids) -> str, and vocab_size.
"""
from typing import List, Optional

try:
    from transformers import AutoTokenizer
    HAS_HF = True
except Exception:
    AutoTokenizer = None  # type: ignore
    HAS_HF = False


class SimpleVocab:
    """Very small whitespace vocab used as a fallback.

    It assigns ids on the fly and keeps a growing mapping. Not suitable
    for production, but useful for tests and quick PoC.
    """
    def __init__(self):
        self.itos = ['<pad>', '<unk>']
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def encode(self, text: str) -> List[int]:
        toks = text.split()
        ids = []
        for t in toks:
            if t not in self.stoi:
                self.stoi[t] = len(self.itos)
                self.itos.append(t)
            ids.append(self.stoi[t])
        return ids

    def decode(self, ids: List[int]) -> str:
        return ' '.join(self.itos[i] if i < len(self.itos) else '<unk>' for i in ids)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)


class Tokenizer:
    def __init__(self, hf_name: Optional[str] = None):
        self._hf = None
        self._simple = None
        if hf_name is not None:
            if not HAS_HF:
                raise RuntimeError('transformers not available to load HF tokenizer')
            self._hf = AutoTokenizer.from_pretrained(hf_name, use_fast=True)
        else:
            # If transformers is available but no name provided, still allow HF lazy use
            if HAS_HF and self._hf is None:
                # don't auto-load a large tokenizer without explicit name
                pass
            self._simple = SimpleVocab()

    def encode(self, text: str) -> List[int]:
        if self._hf is not None:
            out = self._hf(text, add_special_tokens=False)
            if isinstance(out, dict) and 'input_ids' in out:
                return out['input_ids']
            # fallback
            return list(map(int, out.get('input_ids', [])))
        return self._simple.encode(text)

    def decode(self, ids: List[int]) -> str:
        if self._hf is not None:
            return self._hf.decode(ids)
        return self._simple.decode(ids)

    @property
    def vocab_size(self) -> int:
        if self._hf is not None:
            return self._hf.vocab_size
        return self._simple.vocab_size
