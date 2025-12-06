
import unittest
import torch
import numpy as np
from pathlib import Path
import shutil
import tempfile
from crsm.dataset import PretokenizedDataset

class TestPretokenizedDataset(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.seq_len = 10
        
        # Create a dummy bin file
        self.tokens = np.arange(100, dtype=np.uint16)
        self.split = "train"
        self.file_path = self.test_dir / f"data_{self.split}_001.bin"
        
        with open(self.file_path, "wb") as f:
            f.write(self.tokens.tobytes())
            
    def tearDown(self):
        shutil.rmtree(self.test_dir)
        
    def test_loading(self):
        ds = PretokenizedDataset(self.test_dir, seq_len=self.seq_len, split=self.split)
        
        iterator = iter(ds)
        
        # First batch: 0..10 -> input 0..9, target 1..10
        x, y = next(iterator)
        
        self.assertEqual(len(x), self.seq_len)
        self.assertEqual(len(y), self.seq_len)
        
        self.assertTrue(torch.all(x == torch.arange(0, 10, dtype=torch.long)))
        self.assertTrue(torch.all(y == torch.arange(1, 11, dtype=torch.long)))
        
    def test_split_filtering(self):
        # Create a val file
        val_path = self.test_dir / "data_val_001.bin"
        with open(val_path, "wb") as f:
            f.write(self.tokens.tobytes())
            
        ds_train = PretokenizedDataset(self.test_dir, split="train")
        self.assertEqual(len(ds_train.files), 1)
        self.assertIn("train", ds_train.files[0].name)
        
        ds_val = PretokenizedDataset(self.test_dir, split="val")
        self.assertEqual(len(ds_val.files), 1)
        self.assertIn("val", ds_val.files[0].name)

if __name__ == "__main__":
    unittest.main()
