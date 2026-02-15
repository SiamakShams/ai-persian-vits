#!/bin/bash
# Train VITS on Persian corpus

set -e

ITERATIONS=${1:-100000}

echo "🎤 Training VITS on Persian Corpus"
echo "==================================="
echo "Iterations: $ITERATIONS"
echo "This will take 3-5 days on RTX 5070 Ti"
echo ""

if [ ! -d "../datasets/processed" ]; then
    echo "❌ Processed datasets not found"
    echo "Run: python3 preprocessing/preprocess_datasets.py --output_dir datasets/processed"
    exit 1
fi

echo "Starting training..."
python3 train_vits.py \
    --data_path ../datasets/processed \
    --checkpoint_path ../checkpoints \
    --iterations "$ITERATIONS" \
    --batch_size 32 \
    --learning_rate 2e-4

echo "✅ Training complete!"
echo "Model saved to: checkpoints/vits_persian_final.pth"
