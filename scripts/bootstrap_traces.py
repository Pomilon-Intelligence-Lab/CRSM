"""Generate pseudo-reasoning traces from existing text."""
from pathlib import Path
import random

input_file = "data/text_corpus/wikitext103.txt"
output_file = "data/train_traces.jsonl"

print("Generating reasoning traces from text...")

traces = []
with open(input_file) as f:
    lines = [l.strip() for l in f if len(l.strip()) > 100]

# Sample 10K lines and format as traces
samples = random.sample(lines, min(10000, len(lines)))

import json
with open(output_file, "w") as out:
    for text in samples:
        # Simple heuristic: treat as reasoning trace
        trace = {
            "prompt": text[:50],  # First 50 chars as prompt
            "trace": text,
            "erased": False
        }
        out.write(json.dumps(trace) + "\n")

print(f"✓ Generated {len(samples)} traces to {output_file}")