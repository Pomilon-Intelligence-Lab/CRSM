import tempfile
from pathlib import Path
from crsm.dataset import StreamingTextDataset


def test_streaming_from_files():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        # create a tiny corpus
        (p / 'a.txt').write_text('hello world this is a test')
        (p / 'b.txt').write_text('another line of text for streaming')

        ds = StreamingTextDataset(data_dir=td, seq_len=4)
        it = iter(ds)
        # collect a few samples
        samples = []
        for _ in range(3):
            inp, tgt = next(it)
            assert inp.shape[0] == 3
            assert tgt.shape[0] == 3
            samples.append((inp, tgt))
        assert len(samples) == 3
