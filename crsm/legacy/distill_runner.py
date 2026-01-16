"""High-level distillation runner to generate teacher traces from prompts.

Usage:
    python -m crsm.distill_runner --prompts-file prompts.txt --out traces.jsonl --provider gemini

This script supports simple parallel generation using ThreadPoolExecutor to speed up IO-bound teacher API calls.
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

from .distill import get_provider


def read_prompts(path: str) -> List[str]:
    p = Path(path)
    return [l.strip() for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]


def generate_batch(prompts: List[str], provider_kind: str, out_path: str, prompt_erasure: bool = False, workers: int = 4):
    prov = get_provider(provider_kind)
    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(prov.generate, p): p for p in prompts}
        for fut in as_completed(futures):
            p = futures[fut]
            try:
                trace = fut.result()
            except Exception as e:
                trace = f'__ERROR__:{e}'
            if prompt_erasure:
                erased = trace.replace(p, '')
                entry = {'prompt': p, 'trace': erased, 'erased': True}
            else:
                entry = {'prompt': p, 'trace': trace, 'erased': False}
            results.append(entry)

    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open('w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompts-file', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--provider', type=str, default='gemini')
    parser.add_argument('--prompt-erasure', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    prompts = read_prompts(args.prompts_file)
    generate_batch(prompts, args.provider, args.out, prompt_erasure=args.prompt_erasure, workers=args.workers)


if __name__ == '__main__':
    main()
