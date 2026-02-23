
import unittest
import torch
import numpy as np
from pathlib import Path
import shutil
import tempfile
from crsm.data.datasets import PretokenizedDataset

class TestPretokenizedDatasetContinuity(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.seq_len = 5
        self.chunk_len = self.seq_len + 1 # 6
        
        # Create two files.
        # File 1: 0, 1, 2, 3 (4 tokens) - too small for one sequence
        # File 2: 4, 5, 6, 7, 8, 9, 10, 11 (8 tokens)
        
        # Combined: 0..11 (12 tokens)
        # Sequence 1: 0,1,2,3,4,5 (input 0..4, target 1..5)
        # Sequence 2: 6,7,8,9,10,11 (input 6..10, target 7..11)
        
        # Wait, simple streaming uses chunks of size seq_len + 1.
        # So we need 6 tokens per yield.
        
        self.split = "train"
        
        f1 = self.test_dir / f"data_{self.split}_001.bin"
        tokens1 = np.arange(4, dtype=np.uint16)
        with open(f1, "wb") as f:
            f.write(tokens1.tobytes())
            
        f2 = self.test_dir / f"data_{self.split}_002.bin"
        tokens2 = np.arange(4, 12, dtype=np.uint16)
        with open(f2, "wb") as f:
            f.write(tokens2.tobytes())
            
    def tearDown(self):
        shutil.rmtree(self.test_dir)
        
    def test_continuity(self):
        ds = PretokenizedDataset(self.test_dir, seq_len=self.seq_len, split=self.split)
        iterator = iter(ds)
        
        # First batch should come from merging File 1 and start of File 2
        # Data: 0,1,2,3 + 4,5
        x1, y1 = next(iterator)
        print(f"Batch 1: {x1.tolist()}")
        self.assertEqual(x1.tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(y1.tolist(), [1, 2, 3, 4, 5])
        
        # Second batch should come from File 2
        # Remaining in File 2: 6,7,8,9,10,11 (6 tokens left)
        # This is exactly one chunk.
        
        # But wait, my logic was:
        # 1. needed = 6 - 4 = 2.
        # 2. Take 2 from File 2 (4,5). Yield.
        # 3. current_idx becomes 2.
        # 4. Total len of File 2 is 8. Remaining = 6.
        # 5. num_chunks = 1.
        # 6. Yield next chunk (from idx 2 to 8 -> 6,7,8,9,10,11).
        
        try:
            x2, y2 = next(iterator)
            print(f"Batch 2: {x2.tolist()}")
            self.assertEqual(x2.tolist(), [6, 7, 8, 9, 10])
            self.assertEqual(y2.tolist(), [7, 8, 9, 10, 11])
        except StopIteration:
            self.fail("Should have yielded a second batch")
            
        # Should be empty now
        with self.assertRaises(StopIteration):
            next(iterator)

if __name__ == "__main__":
    unittest.main()
