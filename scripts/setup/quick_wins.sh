#!/bin/bash
set -e

echo "============================================================"
echo "CRSM Quick Wins - Immediate Improvements"
echo "============================================================"

# 1. Download better training data
echo -e "\n[1/4] Downloading WikiText dataset..."
python << 'EOF'
from datasets import load_dataset
from pathlib import Path

Path("data/text_corpus").mkdir(parents=True, exist_ok=True)

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
with open("data/text_corpus/wikitext.txt", "w") as f:
    for item in dataset:
        if item['text'].strip() and len(item['text']) > 50:
            f.write(item['text'] + "\n")

print(f"✓ Downloaded {len(dataset)} articles")
EOF

# 2. Build larger vocabulary
echo -e "\n[2/4] Building larger vocabulary (5000 tokens)..."
python scripts/build_vocab.py \
    --corpus-dir data/text_corpus \
    --vocab-size 5000 \
    --output experiments/improved/vocab.json

# 3. Train with better hyperparameters
echo -e "\n[3/4] Training improved model..."
python -m crsm.cli train \
    --epochs 50 \
    --batch-size 32 \
    --vocab-size 5000 \
    --seq-len 128 \
    --lr 0.0005 \
    --data-dir data/text_corpus \
    --checkpoint-dir experiments/improved/model \
    --grad-accum 2 \
    --no-value-loss

# 4. Evaluate
echo -e "\n[4/4] Evaluating model..."
python scripts/evaluate.py \
    --model-path experiments/improved/model/crsm_epoch50.pt \
    --vocab-path experiments/improved/vocab.json \
    --config-path configs/small.json \
    --test-data data/text_corpus \
    --output experiments/improved/evaluation.json

echo -e "\n============================================================"
echo "✓ QUICK WINS COMPLETE"
echo "============================================================"
echo "Results:"
cat experiments/improved/evaluation.json
echo ""
echo "Next: Try generation with improved model"
echo "  python scripts/generate_crsm.py \\"
echo "    --model-path experiments/improved/model/crsm_epoch50.pt \\"
echo "    --vocab-path experiments/improved/vocab.json \\"
echo "    --prompt 'Your prompt' \\"
echo "    --temperature 0.7"