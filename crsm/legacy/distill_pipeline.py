"""Large-scale distillation pipeline: sharded JSONL writers with basic filtering and optional embeddings.

This tool reads prompts, runs teacher provider generation (parallel), filters traces,
optionally computes embeddings using a local HF model (if available), and writes
sharded JSONL output to an output directory.
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

from .distill import get_provider


def _compute_embedding_local(text: str, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        inputs = tok(text, return_tensors='pt', truncation=True)
        with torch.no_grad():
            out = model(**inputs)
        # mean pooling
        emb = out.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy().tolist()
        return emb
    except Exception:
        # fallback: small hashing-based pseudo-embedding
        return [float(abs(hash(text)) % 10000) / 10000.0]


def generate_sharded(prompts: List[str], out_dir: str, provider_kind: str = 'gemini',
                     shard_size: int = 1000, workers: int = 8, prompt_erasure: bool = False,
                     add_embeddings: bool = False, embed_model: Optional[str] = None):
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    provider = get_provider(provider_kind)

    def _gen(prompt):
        try:
            trace = provider.generate(prompt)
        except Exception as e:
            trace = f'__ERROR__:{e}'
        if prompt_erasure:
            trace = trace.replace(prompt, '')
        out = {'prompt': prompt, 'trace': trace, 'erased': prompt_erasure}
        if add_embeddings:
            out['embedding'] = _compute_embedding_local(trace if not prompt_erasure else trace)
        return out

    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_gen, p): p for p in prompts}
        for fut in as_completed(futures):
            results.append(fut.result())

    # write sharded files
    for i in range(0, len(results), shard_size):
        shard = results[i:i+shard_size]
        shard_path = outp / f'shard_{i//shard_size:05d}.jsonl'
        with shard_path.open('w', encoding='utf-8') as f:
            for r in shard:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompts-file', type=str, required=True)
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--provider', type=str, default='gemini')
    parser.add_argument('--shard-size', type=int, default=1000)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--prompt-erasure', action='store_true')
    parser.add_argument('--add-embeddings', action='store_true')
    parser.add_argument('--embed-model', type=str, default=None)
    args = parser.parse_args()

    prompts = [l.strip() for l in Path(args.prompts_file).read_text(encoding='utf-8').splitlines() if l.strip()]
    generate_sharded(prompts, args.out_dir, provider_kind=args.provider, shard_size=args.shard_size,
                     workers=args.workers, prompt_erasure=args.prompt_erasure, add_embeddings=args.add_embeddings,
                     embed_model=args.embed_model)


if __name__ == '__main__':
    main()
