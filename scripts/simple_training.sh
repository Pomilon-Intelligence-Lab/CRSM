#!/bin/bash
# Complete CRSM Training Pipeline
# This script trains a full CRSM model from scratch

set -e

echo "============================================================"
echo "CRSM Complete Training Pipeline"
echo "============================================================"

# Configuration
VOCAB_SIZE=1000
D_MODEL=128
NUM_LAYERS=2
D_FFN=512
SEQ_LEN=32
OUTPUT_DIR="experiments/complete_crsm"
CORPUS_DIR="data/text_corpus"

# Step 1: Ensure training corpus exists
if [ ! -f "$CORPUS_DIR/training_text.txt" ]; then
    echo -e "\n[Step 1/6] Creating training corpus..."
    mkdir -p "$CORPUS_DIR"
    
    cat > "$CORPUS_DIR/training_text.txt" << 'EOF'
The continuous reasoning state model combines state space models with Monte Carlo tree search.
State space models like Mamba enable efficient sequence processing with linear complexity.
Monte Carlo tree search provides strategic planning through simulated lookahead and exploration.
The model maintains a continuous latent state that evolves during generation.
Asynchronous deliberation allows deep reasoning without sacrificing response latency.
Value head predictions guide the search toward high-quality generation outcomes.
The architecture integrates backbone processing with deliberation modules seamlessly.
Training on real text data helps the model learn meaningful linguistic patterns.
Token embeddings capture semantic relationships between words in the vocabulary.
The system can generate coherent continuations by predicting likely next tokens.
Gradient accumulation enables training larger models on limited hardware resources.
Mixed precision training accelerates computation while maintaining numerical stability.
Checkpoint saving allows resuming training from intermediate states safely.
The model learns to balance exploitation of known patterns with exploration of alternatives.
Planning ahead through tree search improves generation quality significantly.
The latent dynamics model predicts how hidden states evolve over time.
Value estimates help evaluate the quality of different generation trajectories.
The tokenizer maps text to integer sequences for neural network processing.
Decoding converts model outputs back into human-readable text strings.
The training loop iterates over batches updating parameters via backpropagation.
Loss functions measure prediction accuracy guiding the optimization process.
Regularization techniques prevent overfitting to the training data distribution.
The model generalizes by learning underlying patterns rather than memorizing examples.
Inference time deliberation enhances output quality through strategic reasoning.
The system balances computational efficiency with generation quality objectives.
Neural language models predict probability distributions over vocabulary tokens.
Attention mechanisms help models focus on relevant context information.
State space models offer an alternative to attention for sequence modeling.
The Mamba architecture achieves strong performance with linear time complexity.
CRSM extends Mamba by adding deliberative reasoning capabilities effectively.
EOF
    
    echo "✓ Created training corpus"
else
    echo -e "\n[Step 1/6] Using existing training corpus"
fi

# Step 2: Build vocabulary
echo -e "\n[Step 2/6] Building vocabulary..."
python scripts/build_vocab.py \
    --corpus-dir "$CORPUS_DIR" \
    --vocab-size $VOCAB_SIZE \
    --output "$OUTPUT_DIR/vocab.json" \
    --test-text "The continuous reasoning state model"

# Step 3: Train backbone (language modeling only)
echo -e "\n[Step 3/6] Training backbone..."
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

echo "✓ Backbone training complete"

# Step 4: Distill dynamics (if traces available)
if [ -f "data/train_traces.jsonl" ]; then
    echo -e "\n[Step 4/6] Distilling dynamics..."
    python scripts/distill_dynamics.py \
        --model-path "$OUTPUT_DIR/backbone/crsm_epoch30.pt" \
        --output-path "$OUTPUT_DIR/dynamics.pt" \
        --traces-path "data/train_traces.jsonl" \
        --num-samples 5000 \
        --epochs 10 \
        --lr 0.001 \
        --d-model $D_MODEL \
        --vocab-size $VOCAB_SIZE \
        --num-layers $NUM_LAYERS
    
    echo "✓ Dynamics distillation complete"
else
    echo -e "\n[Step 4/6] Skipping dynamics (no traces found)"
    echo "  To train dynamics, create data/train_traces.jsonl with reasoning traces"
fi

# Step 5: Fine-tune with value head
echo -e "\n[Step 5/6] Fine-tuning with value head..."
python -m crsm.cli train \
    --epochs 40 \
    --batch-size 16 \
    --vocab-size $VOCAB_SIZE \
    --seq-len $SEQ_LEN \
    --lr 0.0005 \
    --data-dir "$CORPUS_DIR" \
    --resume "$OUTPUT_DIR/backbone/crsm_epoch30.pt" \
    --checkpoint-dir "$OUTPUT_DIR/final" \
    --seed 42

echo "✓ Fine-tuning complete"

# Step 6: Test generation
echo -e "\n[Step 6/6] Testing generation..."

# Test with sampling (fast)
echo -e "\nTest 1: Temperature Sampling"
python scripts/generate_crsm.py \
    --model-path "$OUTPUT_DIR/final/crsm_epoch40.pt" \
    --vocab-path "$OUTPUT_DIR/vocab.json" \
    --config-path "configs/small.json" \
    --prompt "The continuous reasoning state model" \
    --max-length 20 \
    --temperature 0.8

# Test with MCTS (if dynamics available)
if [ -f "$OUTPUT_DIR/dynamics.pt" ]; then
    echo -e "\nTest 2: MCTS with Dynamics"
    python scripts/generate_crsm.py \
        --model-path "$OUTPUT_DIR/final/crsm_epoch40.pt" \
        --vocab-path "$OUTPUT_DIR/vocab.json" \
        --dynamics-path "$OUTPUT_DIR/dynamics.pt" \
        --config-path "configs/small.json" \
        --prompt "The model training" \
        --max-length 15 \
        --use-mcts
fi

echo -e "\n============================================================"
echo "✓ COMPLETE PIPELINE FINISHED"
echo "============================================================"
echo "Artifacts:"
echo "  - Vocabulary: $OUTPUT_DIR/vocab.json"
echo "  - Backbone: $OUTPUT_DIR/backbone/crsm_epoch30.pt"
if [ -f "$OUTPUT_DIR/dynamics.pt" ]; then
    echo "  - Dynamics: $OUTPUT_DIR/dynamics.pt"
fi
echo "  - Final Model: $OUTPUT_DIR/final/crsm_epoch40.pt"
echo ""
echo "To generate text:"
echo "  python scripts/generate_crsm.py \\"
echo "    --model-path $OUTPUT_DIR/final/crsm_epoch40.pt \\"
echo "    --vocab-path $OUTPUT_DIR/vocab.json \\"
if [ -f "$OUTPUT_DIR/dynamics.pt" ]; then
    echo "    --dynamics-path $OUTPUT_DIR/dynamics.pt \\"
fi
echo "    --config-path configs/small.json \\"
echo "    --prompt 'Your prompt here'"