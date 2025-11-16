"""
Fixed dataset module with proper tokenizer handling.
"""
import gc
from collections import deque
from typing import Iterator, Optional
import torch
from torch.utils.data import Dataset, IterableDataset
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Iterable

from .tokenizer import Tokenizer

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except Exception:
    load_dataset = None  # type: ignore
    HAS_DATASETS = False


class RandomTokenDataset(Dataset):
    """Generates random token sequences for a quick PoC training loop."""
    def __init__(self, vocab_size: int = 1000, seq_len: int = 32, size: int = 1000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        seq = torch.randint(0, self.vocab_size, (self.seq_len,), dtype=torch.long)
        # inputs and targets (next-token prediction)
        return seq[:-1], seq[1:]


class Vocab:
    def __init__(self, tokens: List[str]):
        self.itos = ['<pad>', '<unk>'] + tokens
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.stoi.get(t, self.stoi['<unk>']) for t in tokens]


class RealTextDataset(Dataset):
    """Loads text files from a directory, builds a simple token-level vocab,
    and yields sequences for next-token prediction.
    """
    def __init__(self, data_dir: str, seq_len: int = 128, min_freq: int = 2, 
                 tokenizer: Optional[object] = None, hf_tokenizer_name: Optional[str] = None,
                 vocab_size: Optional[int] = None):
        """
        Args:
            data_dir: directory with .txt files
            seq_len: length of sequence (including next-token target)
            min_freq: minimum token frequency for building simple vocab (ignored if tokenizer provided)
            tokenizer: optional callable/tokenizer object
            hf_tokenizer_name: optional Hugging Face tokenizer name
            vocab_size: vocab size for simple tokenizer
        """
        self.data_dir = Path(data_dir)
        files = list(self.data_dir.glob('**/*.txt'))

        # If an HF tokenizer name is provided, build a Tokenizer wrapper
        if hf_tokenizer_name is not None:
            tokenizer = Tokenizer(hf_name=hf_tokenizer_name)
        elif tokenizer is None:
            # Create tokenizer with vocab_size
            tokenizer = Tokenizer(vocab_size=vocab_size, prepopulate=False)

        self._use_hf = tokenizer is not None

        if self._use_hf:
            # Tokenize each file and concatenate ids
            data = []
            for p in files:
                txt = p.read_text(encoding='utf-8')
                # Tokenize with truncation disabled (we'll chunk it ourselves)
                # Use add_special_tokens=False to avoid adding extra tokens
                if hasattr(tokenizer, '_hf') and tokenizer._hf is not None:
                    # Direct HF tokenizer
                    ids = tokenizer._hf.encode(txt, add_special_tokens=False)
                else:
                    # Tokenizer wrapper
                    ids = tokenizer.encode(txt)
                data.extend(ids)
            self.data = torch.tensor(data, dtype=torch.long)
            self.seq_len = seq_len
            self.vocab = None
            return

        # Legacy simple whitespace tokenizer path (shouldn't reach here anymore)
        all_tokens = []
        for p in files:
            txt = p.read_text(encoding='utf-8')
            toks = txt.split()
            all_tokens.extend(toks)

        counts = Counter(all_tokens)
        tokens = [t for t, c in counts.items() if c >= min_freq]
        self.vocab = Vocab(tokens)

        # concatenate all tokens into a flat list of ints
        data = []
        for p in files:
            toks = p.read_text(encoding='utf-8').split()
            data.extend(self.vocab.encode(toks))

        self.data = torch.tensor(data, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.data[idx: idx + self.seq_len]
        return seq[:-1].clone(), seq[1:].clone()


class StreamingTextDataset(IterableDataset):
    """An iterable dataset that streams tokens from large text corpora.

    It supports two modes:
      - Using the `datasets` library in streaming mode when available.
      - Falling back to simple file-based streaming (line-by-line tokenization).

    The dataset yields (input_seq, target_seq) tensors of length seq_len-1 each
    where input is tokens[:-1] and target is tokens[1:].
    """
    def __init__(self, data_dir: Optional[str] = None, file_patterns: Optional[Iterable[str]] = None,
                 seq_len: int = 128, tokenizer: Optional[object] = None, hf_tokenizer_name: Optional[str] = None,
                 dataset_name: Optional[str] = None, vocab_size: Optional[int] = None):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        
        # Create tokenizer if not provided
        if hf_tokenizer_name is not None:
            self.tokenizer = Tokenizer(hf_name=hf_tokenizer_name)
        elif self.tokenizer is None:
            # Use simple tokenizer with vocab_size
            self.tokenizer = Tokenizer(vocab_size=vocab_size, prepopulate=False)

        # data source selection
        self.data_dir = Path(data_dir) if data_dir else None
        self.file_patterns = list(file_patterns) if file_patterns is not None else None
        self.dataset_name = dataset_name

    def _stream_tokens_from_files(self) -> Iterable[int]:
        # iterate over files and yield token ids lazily
        files = []
        if self.file_patterns and self.data_dir:
            for pat in self.file_patterns:
                files.extend(self.data_dir.glob(pat))
        elif self.data_dir:
            files = list(self.data_dir.glob('**/*.txt'))

        for p in files:
            print(f"Streaming file: {p.name}")
            with p.open(encoding='utf-8') as f:
                for line in f:
                    # Tokenize only the current line
                    ids = self.tokenizer.encode(line)
                    for i in ids:
                        yield i
                    # CRITICAL: Delete the temporary list and prompt GC
                    del ids 
                    gc.collect() # <-- ADD THIS LINE AFTER EACH LINE TOKENIZATION

            # CRITICAL: Delete file buffer and force GC after processing entire file
            gc.collect()

    def token_stream(self):
        if HAS_DATASETS and self.dataset_name is not None:
            yield from self._stream_tokens_from_datasets()
        else:
            yield from self._stream_tokens_from_files()

    def __iter__(self):
        # FIX: Replace the infinitely growing list [] with a fixed-size deque
        # The buffer only ever holds self.seq_len tokens.
        buf = deque(maxlen=self.seq_len) 

        for tok in self.token_stream():
            buf.append(tok)
            
            # We yield a sequence as soon as the buffer is full.
            if len(buf) == self.seq_len:
                # Convert deque to a list/tuple for tensor creation
                seq = list(buf)
                
                inp = torch.tensor(seq[:-1], dtype=torch.long)
                tgt = torch.tensor(seq[1:], dtype=torch.long)
                yield inp, tgt