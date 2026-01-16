import json
from pathlib import Path
from typing import Optional
from .tokenizer import Tokenizer

def traces_to_token_shards(traces_path: str, out_dir: str, shard_size: int = 1000, hf_tokenizer_name: Optional[str] = None):
    p = Path(traces_path)
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    tok = Tokenizer(hf_name=hf_tokenizer_name) if hf_tokenizer_name else Tokenizer()

    entries = []
    for line in p.open('r', encoding='utf-8'):
        obj = json.loads(line)
        prompt = obj.get('prompt', '')
        trace = obj.get('trace', '')
        input_text = (prompt + '\n' + trace).strip()
        input_ids = tok.encode(input_text)
        target_ids = tok.encode(trace)
        entries.append({'input_ids': input_ids, 'target_ids': target_ids})

    for i in range(0, len(entries), shard_size):
        shard = entries[i:i+shard_size]
        shard_path = outp / f'shard_{i//shard_size:05d}.jsonl'
        with shard_path.open('w', encoding='utf-8') as f:
            for e in shard:
                f.write(json.dumps(e, ensure_ascii=False) + '\n')

    return str(outp)