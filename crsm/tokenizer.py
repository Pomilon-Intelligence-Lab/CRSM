"""Tokenizer wrapper with vocab persistence support."""
from typing import List, Optional
import json
from pathlib import Path

try:
    from transformers import AutoTokenizer
    HAS_HF = True
except Exception:
    AutoTokenizer = None  # type: ignore
    HAS_HF = False


class SimpleVocab:
    """Very small whitespace vocab used as a fallback."""
    
    def __init__(self, fixed_vocab_size: Optional[int] = None, prepopulate: bool = False):
        """
        Args:
            fixed_vocab_size: If provided, limits vocab to this size.
            prepopulate: If True, pre-fill with placeholder tokens up to fixed_vocab_size.
                        Set to False when building vocab from training data.
        """
        self.itos = ['<pad>', '<unk>']
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self._fixed_vocab_size = fixed_vocab_size
        
        # Only pre-populate with placeholders if explicitly requested
        # This is useful for decoding but NOT for building vocab from training data
        if fixed_vocab_size and prepopulate:
            while len(self.itos) < fixed_vocab_size:
                placeholder = f'<tok_{len(self.itos)}>'
                self.itos.append(placeholder)
                self.stoi[placeholder] = len(self.itos) - 1

    # From tokenizer.py, SimpleVocab.encode
    def encode(self, text: str) -> List[int]:
        toks = text.split()
        ids = []
        for t in toks:
            if t not in self.stoi:
                # Check if we are enforcing a fixed vocab size
                if self._fixed_vocab_size is not None:
                    # If fixed vocab is set and full, ALWAYS use <unk>
                    if len(self.itos) >= self._fixed_vocab_size:
                        # Vocab is at its limit or we are beyond it
                        ids.append(self.stoi['<unk>'])
                    else:
                        # Still room to grow within the fixed limit (training phase)
                        self.stoi[t] = len(self.itos)
                        self.itos.append(t)
                        ids.append(self.stoi[t])
                else:
                    # No fixed vocab size, always grow (training phase)
                    self.stoi[t] = len(self.itos)
                    self.itos.append(t)
                    ids.append(self.stoi[t])
            else:
                ids.append(self.stoi[t])
        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        words = []
        for i in ids:
            if i < len(self.itos):
                words.append(self.itos[i])
            else:
                # ID is beyond current vocab - create placeholder
                placeholder = f'<tok_{i}>'
                words.append(placeholder)
                # Optionally expand vocab (only if not fixed size)
                if not self._fixed_vocab_size:
                    while len(self.itos) <= i:
                        new_placeholder = f'<tok_{len(self.itos)}>'
                        self.itos.append(new_placeholder)
                        self.stoi[new_placeholder] = len(self.itos) - 1
        return ' '.join(words)

    @property
    def vocab_size(self) -> int:
        if self._fixed_vocab_size:
            return self._fixed_vocab_size
        return len(self.itos)
    
    def save(self, path: str):
        """Save vocabulary to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'itos': self.itos,
                'fixed_vocab_size': self._fixed_vocab_size
            }, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'SimpleVocab':
        """Load vocabulary from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        vocab = cls(fixed_vocab_size=data.get('fixed_vocab_size'))
        vocab.itos = data['itos']
        vocab.stoi = {t: i for i, t in enumerate(vocab.itos)}
        return vocab


class Tokenizer:
    def __init__(self, hf_name: Optional[str] = None, vocab_size: Optional[int] = None, prepopulate: bool = False):
        """
        Args:
            hf_name: HuggingFace tokenizer name (e.g., 'gpt2')
            vocab_size: Fixed vocab size for SimpleVocab fallback
            prepopulate: If True, pre-fill SimpleVocab with placeholders (for decoding only)
        """
        self._hf = None
        self._simple = None
        if hf_name is not None:
            if not HAS_HF:
                raise RuntimeError('transformers not available to load HF tokenizer')
            self._hf = AutoTokenizer.from_pretrained(hf_name, use_fast=True)
        else:
            self._simple = SimpleVocab(fixed_vocab_size=vocab_size, prepopulate=prepopulate)

    def encode(self, text: str) -> List[int]:
        if self._hf is not None:
            # Use add_special_tokens=False and truncation=False
            # Let the dataset handle chunking
            out = self._hf(text, add_special_tokens=False, truncation=False)
            if isinstance(out, dict) and 'input_ids' in out:
                return out['input_ids']
            return list(map(int, out.get('input_ids', [])))
        return self._simple.encode(text)

    def decode(self, ids: List[int]) -> str:
        if self._hf is not None:
            return self._hf.decode(ids, skip_special_tokens=False)
        return self._simple.decode(ids)

    @property
    def vocab_size(self) -> int:
        if self._hf is not None:
            return self._hf.vocab_size
        return self._simple.vocab_size
    
    def save_vocab(self, path: str):
        """Save vocabulary (only for SimpleVocab)."""
        if self._simple is not None:
            self._simple.save(path)
        elif self._hf is not None:
            print("Warning: Cannot save HuggingFace tokenizer vocab this way")
    
    @classmethod
    def from_vocab_file(cls, path: str) -> 'Tokenizer':
        """Load tokenizer from saved vocab file."""
        tok = cls()
        tok._simple = SimpleVocab.load(path)
        return tok