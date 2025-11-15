from datasets import load_dataset
from pathlib import Path

print("Downloading WikiText-103...")
dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

output_dir = Path("data/text_corpus")
output_dir.mkdir(parents=True, exist_ok=True)

# Save to single file
output_file = output_dir / "wikitext103.txt"
with open(output_file, "w", encoding="utf-8") as f:
    for item in dataset:
        text = item['text'].strip()
        if len(text) > 50:  # Filter very short lines
            f.write(text + "\n")

print(f"✓ Downloaded {len(dataset)} articles to {output_file}")
print(f"  File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
