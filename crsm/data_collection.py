# crsm/data_collect.py
import argparse
from pathlib import Path
from .distill_runner import generate_batch
from .latent import traces_to_token_shards

def collect_and_tokenize(prompts_file: str, out_traces: str, out_shards_dir: str, provider: str = 'gemini',
                         workers: int = 4, shard_size: int = 1000, hf_tokenizer_name: str | None = None,
                         prompt_erasure: bool = False):
    # generate traces (JSONL)
    generate_batch([l.strip() for l in Path(prompts_file).read_text(encoding='utf-8').splitlines() if l.strip()],
                   provider_kind=provider, out_path=out_traces, prompt_erasure=prompt_erasure, workers=workers)
    # convert to token shards
    traces_to_token_shards(out_traces, out_shards_dir, shard_size=shard_size, hf_tokenizer_name=hf_tokenizer_name)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompts-file', required=True)
    parser.add_argument('--out-traces', required=True)
    parser.add_argument('--out-shards-dir', required=True)
    parser.add_argument('--provider', default='gemini')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--shard-size', type=int, default=1000)
    parser.add_argument('--hf-tokenizer-name', type=str, default=None)
    parser.add_argument('--prompt-erasure', action='store_true')
    args = parser.parse_args()
    collect_and_tokenize(args.prompts_file, args.out_traces, args.out_shards_dir, provider=args.provider,
                         workers=args.workers, shard_size=args.shard_size, hf_tokenizer_name=args.hf_tokenizer_name,
                         prompt_erasure=args.prompt_erasure)

if __name__ == '__main__':
    main()