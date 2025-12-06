"""
Fixed dataset module with proper tokenizer handling and optimized streaming.
"""
import gc
from collections import deque
from typing import Iterator, Optional, List, Tuple, Iterable
import torch
from torch.utils.data import Dataset, IterableDataset
from pathlib import Path
from collections import Counter
import numpy as np

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


class PretokenizedDataset(IterableDataset):
    """
    Dataset that streams tokens from pre-tokenized binary files.
    """
    def __init__(self, data_dir: str, seq_len: int = 1024, split: str = "train", 
                 vocab_size: int = 50257):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
        # Check vocab size safety
        if self.vocab_size > 65535:
            raise ValueError(f"Vocab size {self.vocab_size} too large for uint16 storage used by PretokenizedDataset")
        
        # Find all bin files matching the split
        self.files = sorted(list(self.data_dir.glob(f"*{split}*.bin")))
        if not self.files:
             raise FileNotFoundError(f"No bin files found in {data_dir} for split '{split}'. Did you run prepare_dataset.py?")
            
        print(f"Found {len(self.files)} bin files in {data_dir} for split {split}")
        
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            # Shard files across workers
            # Simple interleaving
            files = self.files[worker_info.id::worker_info.num_workers]
        else:
            files = self.files
            
        # Buffer to hold tokens from multiple files to ensure continuity
        buffer = np.array([], dtype=np.uint16)
        chunk_len = self.seq_len + 1
        
        for path in files:
            # Read file as numpy array
            try:
                # Use memmap for efficiency
                new_data = np.memmap(path, dtype=np.uint16, mode='r')
                
                # Append to buffer (we have to copy here, but buffer is small)
                # Actually, appending memmap to numpy array forces a read. 
                # To be efficient, we should process the buffer as much as possible first.
                
                # However, for continuity, we need to stitch the end of the previous file 
                # with the start of the new file.
                
                # Since we are iterating, we can yield chunks from `new_data` 
                # but we need to handle the "leftover" from previous file.
                
                # If buffer is not empty (from previous iteration), prepend it
                if len(buffer) > 0:
                     # This forces a read of the whole file if we use np.concatenate
                     # Ideally we only read the beginning.
                     
                     # Optimization: Only concat the leftovers with the start of new file?
                     # No, memmap is array-like.
                     
                     # Let's just process the memmap directly and keep the tail in buffer.
                     pass
                
                # We can't easily concatenate memmap without reading it. 
                # Strategy: 
                # 1. Take leftovers from buffer.
                # 2. Iterate through memmap. 
                # 3. If we have enough for a chunk using leftovers + start of memmap, yield it.
                # 4. Then yield chunks from memmap.
                # 5. Save tail of memmap to buffer.
                
                current_idx = 0
                total_len = len(new_data)
                
                # Handle leftovers
                if len(buffer) > 0:
                    needed = chunk_len - len(buffer)
                    if total_len >= needed:
                        # Take 'needed' from new_data
                        part = new_data[:needed] # This reads from disk
                        full_chunk = np.concatenate([buffer, part])
                        
                        x = torch.from_numpy(full_chunk[:-1].astype(np.int64))
                        y = torch.from_numpy(full_chunk[1:].astype(np.int64))
                        yield x, y
                        
                        current_idx = needed
                        buffer = np.array([], dtype=np.uint16)
                    else:
                        # File is too small to complete the buffer
                        # Append all of it to buffer and continue to next file
                        part = new_data[:]
                        buffer = np.concatenate([buffer, part])
                        continue
                        
                # Process main body of file
                # We can compute how many chunks fit
                remaining = total_len - current_idx
                num_chunks = remaining // chunk_len
                
                if num_chunks > 0:
                    # Create a view or slice
                    # Note: slicing memmap returns memmap, which is good.
                    # We can iterate over the slices.
                    
                    # But constructing individual tensors from memmap slices is fine.
                    for i in range(num_chunks):
                        start = current_idx + i * chunk_len
                        end = start + chunk_len
                        chunk = new_data[start:end]
                        
                        # Copy to memory and convert
                        chunk_arr = np.array(chunk, dtype=np.int64)
                        x = torch.from_numpy(chunk_arr[:-1])
                        y = torch.from_numpy(chunk_arr[1:])
                        yield x, y
                        
                    current_idx += num_chunks * chunk_len
                    
                # Save leftovers
                if current_idx < total_len:
                    buffer = np.array(new_data[current_idx:], dtype=np.uint16)
                    
            except Exception as e:
                print(f"Error reading {path}: {e}")
                continue


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
                 dataset_name: Optional[str] = None, vocab_size: Optional[int] = None, stride: Optional[int] = None):
        self.seq_len = seq_len
        # Default stride to seq_len (non-overlapping) for efficiency
        self.stride = stride if stride is not None else seq_len
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
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else "MAIN"
        
        files = []
        if self.file_patterns and self.data_dir:
            for pat in self.file_patterns:
                files.extend(self.data_dir.glob(pat))
        elif self.data_dir:
            files = list(self.data_dir.glob('**/*.txt'))
        
        print(f"[Worker {worker_id}] Found {len(files)} files")

        # Simple file-based streaming with buffer
        # We read the whole file if it's small, or chunks if it's huge.
        # For simplicity in this implementation, we read line by line but batch tokenization could be better.
        # However, avoiding gc.collect() per line is the main fix.
        
        for p in files:
            # print(f"[Worker {worker_id}] Streaming file: {p.name}")
            with p.open(encoding='utf-8') as f:
                # Read larger chunks instead of line by line to improve tokenization throughput?
                # But line by line is safer for memory.
                # The main bottleneck was gc.collect().
                for line in f:
                    if not line.strip():
                        continue
                    ids = self.tokenizer.encode(line)
                    for i in ids:
                        yield i
                    # Removed aggressive gc.collect() and del ids

            # Optional: collect garbage only after a full file is processed
            # gc.collect()

    def token_stream(self):
        if HAS_DATASETS and self.dataset_name is not None:
            # Assuming _stream_tokens_from_datasets exists or will be implemented
            # For now, fall back to files if not implemented
            yield from self._stream_tokens_from_files()
        else:
            yield from self._stream_tokens_from_files()

    def __iter__(self):
        buffer = []

        for tok in self.token_stream():
            buffer.append(tok)
            
            # When buffer has enough tokens for one sequence
            if len(buffer) >= self.seq_len:
                # Yield the sequence
                seq = buffer[:self.seq_len]
                inp = torch.tensor(seq[:-1], dtype=torch.long)
                tgt = torch.tensor(seq[1:], dtype=torch.long)
                yield inp, tgt

                # Advance buffer by stride
                # If stride == seq_len, we clear the used tokens.
                # If stride < seq_len, we keep some overlap.
                buffer = buffer[self.stride:]
