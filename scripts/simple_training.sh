#!/bin/bash
# Complete CRSM Training Pipeline
set -e

echo "============================================================"
echo "CRSM Complete Training Pipeline (Updated)"
echo "============================================================"

# Configuration
VOCAB_SIZE=1000
D_MODEL=128
NUM_LAYERS=2
D_FFN=512
SEQ_LEN=32
OUTPUT_DIR="experiments/complete_crsm"
CORPUS_DIR="data/text_corpus"

# Verify corpus exists
if [ ! -d "$CORPUS_DIR" ] || [ -z "$(ls -A $CORPUS_DIR/*.txt 2>/dev/null)" ]; then
    echo "Error: Training corpus not found at $CORPUS_DIR"
    echo "Please create training data first."
    exit 1
fi

# Step 1: Build vocabulary
echo -e "\n[Step 1/6] Building vocabulary..."
python scripts/build_vocab.py \
    --corpus-dir "$CORPUS_DIR" \
    --vocab-size $VOCAB_SIZE \
    --output "$OUTPUT_DIR/vocab.json" \
    --test-text "The continuous reasoning state model"

if [ $? -ne 0 ]; then
    echo "Error: Vocabulary building failed"
    exit 1
fi

# Step 2: Train backbone
echo -e "\n[Step 2/6] Training backbone..."
python -m crsm.cli train \
    --epochs 30 \
    --batch-size 16 \
    --vocab-size $VOCAB_SIZE \
    --seq-len $SEQ_LEN \
    --lr 0.001 \
    --data-dir "$CORPUS_DIR" \
    --checkpoint-dir "$OUTPUT_DIR/backbone" \
    --no-value-loss \
    --seed 42

if [ $? -ne 0 ]; then
    echo "Error: Backbone training failed"
    exit 1
fi

# Verify backbone checkpoint exists
BACKBONE_CKPT="$OUTPUT_DIR/backbone/crsm_epoch30.pt"
if [ ! -f "$BACKBONE_CKPT" ]; then
    echo "Error: Backbone checkpoint not found at $BACKBONE_CKPT"
    exit 1
fi

echo "✓ Backbone training complete"

# Step 3: Distill dynamics (if traces available)
if [ -f "data/train_traces.jsonl" ]; then
    echo -e "\n[Step 3/6] Distilling dynamics..."
    python scripts/distill_dynamics.py \
        --model-path "$BACKBONE_CKPT" \
        --output-path "$OUTPUT_DIR/dynamics.pt" \
        --traces-path "data/train_traces.jsonl" \
        --num-samples 5000 \
        --epochs 10 \
        --lr 0.001 \
        --d-model $D_MODEL \
        --vocab-size $VOCAB_SIZE \
        --num-layers $NUM_LAYERS
    
    if [ $? -ne 0 ]; then
        echo "Error: Dynamics distillation failed"
        exit 1
    fi
    
    # Test dynamics quality
    echo -e "\n  Testing dynamics quality..."
    python scripts/test_self_modification.py \
        --checkpoint "$BACKBONE_CKPT" \
        --dynamics-path "$OUTPUT_DIR/dynamics.pt" \
        --iterations 3
    
    echo "✓ Dynamics distillation complete"
else
    echo -e "\n[Step 3/6] Skipping dynamics (no traces found)"
    echo "  Create data/train_traces.jsonl for dynamics training"
fi

# Step 4: Fine-tune with value head
echo -e "\n[Step 4/6] Fine-tuning with value head..."
python -m crsm.cli train \
    --epochs 40 \
    --batch-size 16 \
    --vocab-size $VOCAB_SIZE \
    --seq-len $SEQ_LEN \
    --lr 0.0005 \
    --data-dir "$CORPUS_DIR" \
    --resume "$BACKBONE_CKPT" \
    --checkpoint-dir "$OUTPUT_DIR/final" \
    --seed 42

if [ $? -ne 0 ]; then
    echo "Error: Fine-tuning failed"
    exit 1
fi

FINAL_CKPT="$OUTPUT_DIR/final/crsm_epoch40.pt"
if [ ! -f "$FINAL_CKPT" ]; then
    echo "Error: Final checkpoint not found"
    exit 1
fi

echo "✓ Fine-tuning complete"

# Step 5: Test generation (sampling)
echo -e "\n[Step 5/6] Testing generation (sampling)..."
python scripts/generate_crsm.py \
    --model-path "$FINAL_CKPT" \
    --vocab-path "$OUTPUT_DIR/vocab.json" \
    --config-path "configs/small.json" \
    --prompt "The continuous reasoning state model" \
    --max-length 20 \
    --temperature 0.8

# Step 6: Test with MCTS (if dynamics available)
if [ -f "$OUTPUT_DIR/dynamics.pt" ]; then
    echo -e "\n[Step 6/6] Testing generation (MCTS with dynamics)..."
    python scripts/generate_crsm.py \
        --model-path "$FINAL_CKPT" \
        --vocab-path "$OUTPUT_DIR/vocab.json" \
        --dynamics-path "$OUTPUT_DIR/dynamics.pt" \
        --config-path "configs/small.json" \
        --prompt "The model training" \
        --max-length 15 \
        --use-mcts \
        --show-metrics
    
    # Run comprehensive self-modification test
    echo -e "\n  Running self-modification verification..."
    python scripts/test_self_modification.py \
        --checkpoint "$FINAL_CKPT" \
        --dynamics-path "$OUTPUT_DIR/dynamics.pt" \
        --iterations 5
else
    echo -e "\n[Step 6/6] Skipping MCTS test (no dynamics)"
fi

echo -e "\n============================================================"
echo "✓ COMPLETE PIPELINE FINISHED"
echo "============================================================"
echo "Artifacts:"
echo "  - Vocabulary: $OUTPUT_DIR/vocab.json"
echo "  - Backbone: $BACKBONE_CKPT"
if [ -f "$OUTPUT_DIR/dynamics.pt" ]; then
    echo "  - Dynamics: $OUTPUT_DIR/dynamics.pt"
fi
echo "  - Final Model: $FINAL_CKPT"
echo ""
echo "Quick test commands:"
echo ""
echo "# Generate with sampling:"
echo "python scripts/generate_crsm.py \\"
echo "  --model-path $FINAL_CKPT \\"
echo "  --vocab-path $OUTPUT_DIR/vocab.json \\"
echo "  --config-path configs/small.json \\"
echo "  --prompt 'Your prompt here' \\"
echo "  --temperature 0.8"
echo ""
if [ -f "$OUTPUT_DIR/dynamics.pt" ]; then
    echo "# Generate with MCTS (self-modification):"
    echo "python scripts/generate_crsm.py \\"
    echo "  --model-path $FINAL_CKPT \\"
    echo "  --vocab-path $OUTPUT_DIR/vocab.json \\"
    echo "  --dynamics-path $OUTPUT_DIR/dynamics.pt \\"
    echo "  --config-path configs/small.json \\"
    echo "  --prompt 'Your prompt here' \\"
    echo "  --use-mcts --show-metrics"
fi