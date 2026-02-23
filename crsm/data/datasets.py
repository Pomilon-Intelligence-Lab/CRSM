"""
Fixed dataset module with proper tokenizer handling and optimized streaming.
"""
import gc
import json
from collections import deque
from typing import Iterator, Optional, List, Tuple, Iterable
import torch
from torch.utils.data import Dataset, IterableDataset
from pathlib import Path
from collections import Counter
import numpy as np

from .tokenizers import Tokenizer

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
            files = self.files[worker_info.id::worker_info.num_workers]
        else:
            files = self.files
            
        buffer = np.array([], dtype=np.uint16)
        chunk_len = self.seq_len + 1
        
        for path in files:
            try:
                new_data = np.memmap(path, dtype=np.uint16, mode='r')
                current_idx = 0
                total_len = len(new_data)
                
                if len(buffer) > 0:
                    needed = chunk_len - len(buffer)
                    if total_len >= needed:
                        part = new_data[:needed]
                        full_chunk = np.concatenate([buffer, part])
                        x = torch.from_numpy(full_chunk[:-1].astype(np.int64))
                        y = torch.from_numpy(full_chunk[1:].astype(np.int64))
                        yield x, y
                        current_idx = needed
                        buffer = np.array([], dtype=np.uint16)
                    else:
                        part = new_data[:]
                        buffer = np.concatenate([buffer, part])
                        continue
                        
                remaining = total_len - current_idx
                num_chunks = remaining // chunk_len
                
                if num_chunks > 0:
                    for i in range(num_chunks):
                        start = current_idx + i * chunk_len
                        end = start + chunk_len
                        chunk = new_data[start:end]
                        chunk_arr = np.array(chunk, dtype=np.int64)
                        x = torch.from_numpy(chunk_arr[:-1])
                        y = torch.from_numpy(chunk_arr[1:])
                        yield x, y
                    current_idx += num_chunks * chunk_len
                    
                if current_idx < total_len:
                    buffer = np.array(new_data[current_idx:], dtype=np.uint16)
            except Exception as e:
                print(f"Error reading {path}: {e}")
                continue


class StreamingTextDataset(IterableDataset):
    def __init__(self, data_dir: Optional[str] = None, file_patterns: Optional[Iterable[str]] = None,
                 seq_len: int = 128, tokenizer: Optional[object] = None, hf_tokenizer_name: Optional[str] = None,
                 dataset_name: Optional[str] = None, vocab_size: Optional[int] = None, stride: Optional[int] = None):
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len
        self.tokenizer = tokenizer
        
        if hf_tokenizer_name is not None:
            self.tokenizer = Tokenizer(hf_name=hf_tokenizer_name)
        elif self.tokenizer is None:
            self.tokenizer = Tokenizer(vocab_size=vocab_size, prepopulate=False)

        self.data_dir = Path(data_dir) if data_dir else None
        self.file_patterns = list(file_patterns) if file_patterns is not None else None
        self.dataset_name = dataset_name

    def _stream_tokens_from_files(self) -> Iterable[int]:
        files = []
        if self.file_patterns and self.data_dir:
            for pat in self.file_patterns:
                files.extend(self.data_dir.glob(pat))
        elif self.data_dir:
            files = list(self.data_dir.glob('**/*.txt'))
        
        for p in files:
            with p.open(encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    ids = self.tokenizer.encode(line)
                    for i in ids: yield i

    def token_stream(self):
        yield from self._stream_tokens_from_files()

    def __iter__(self):
        buffer = []
        for tok in self.token_stream():
            buffer.append(tok)
            if len(buffer) >= self.seq_len:
                seq = buffer[:self.seq_len]
                inp = torch.tensor(seq[:-1], dtype=torch.long)
                tgt = torch.tensor(seq[1:], dtype=torch.long)
                yield inp, tgt
                buffer = buffer[self.stride:]


class ARCDataset(Dataset):
    """
    Dataset for ARC-AGI tasks. 
    Encodes grids into token sequences.
    
    Tokens:
    0-9: Colors
    10: [EXAMPLE_START]
    11: [INPUT_START]
    12: [OUTPUT_START]
    13: [ROW_END]
    14: [GRID_END]
    15: [TEST_START]
    """
    def __init__(self, data_path: str = None, samples: List[dict] = None, seq_len: int = 512):
        self.seq_len = seq_len
        self.data = []
        
        if data_path:
            p = Path(data_path)
            if p.is_dir():
                files = list(p.glob("*.json"))
                for f in files:
                    with f.open('r') as jf:
                        self.data.append(json.load(jf))
            elif p.is_file():
                with p.open('r') as jf:
                    self.data.append(json.load(jf))
        elif samples:
            self.data = samples

    def encode_grid(self, grid: List[List[int]]) -> List[int]:
        tokens = []
        rows = len(grid)
        cols = len(grid[0]) if rows > 0 else 0
        
        # Dimension Headers: 
        # 16-45 (Rows 1-30)
        # 46-75 (Cols 1-30)
        tokens.append(min(45, 15 + rows))
        tokens.append(min(75, 45 + cols))
        
        for row in grid:
            tokens.extend(row)
            tokens.append(13) # [ROW_END]
        tokens.append(14) # [GRID_END]
        return tokens

    def encode_task(self, task: dict) -> Tuple[List[int], List[int]]:
        tokens = []
        # Encode training examples
        for ex in task.get('train', []):
            tokens.append(10) # [EXAMPLE_START]
            tokens.append(11) # [INPUT_START]
            tokens.extend(self.encode_grid(ex['input']))
            tokens.append(12) # [OUTPUT_START]
            tokens.extend(self.encode_grid(ex['output']))
            
        # Encode test input
        test_examples = task.get('test', [])
        if not test_examples:
            return tokens, []
            
        test_ex = test_examples[0]
        tokens.append(15) # [TEST_START]
        tokens.append(11) # [INPUT_START]
        tokens.extend(self.encode_grid(test_ex['input']))
        tokens.append(12) # [OUTPUT_START]
        
        # Target part (for training)
        if 'output' in test_ex:
            target_tokens = self.encode_grid(test_ex['output'])
        else:
            target_tokens = []
            
        return tokens, target_tokens

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        task = self.data[idx]
        input_tokens, target_tokens = self.encode_task(task)
        full_seq = input_tokens + target_tokens
        
        if len(full_seq) > self.seq_len:
            full_seq = full_seq[:self.seq_len]
        
        x = torch.zeros(self.seq_len, dtype=torch.long)
        y = torch.zeros(self.seq_len, dtype=torch.long)
        
        actual_len = len(full_seq)
        if actual_len > 1:
            x[:actual_len-1] = torch.tensor(full_seq[:-1])
            y[:actual_len-1] = torch.tensor(full_seq[1:])
            
        return x, y, len(input_tokens)