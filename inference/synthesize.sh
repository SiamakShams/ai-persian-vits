#!/bin/bash
# Synthesize speech in cloned voice

set -e

if [ $# -lt 3 ]; then
    echo "Usage: bash synthesize.sh <text> <speaker_name> <output.wav>"
    echo "Example: bash synthesize.sh 'سلام دنیا' john output.wav"
    exit 1
fi

TEXT="$1"
SPEAKER="$2"
OUTPUT="$3"

echo "🎤 Synthesizing: $TEXT"
echo "Speaker: $SPEAKER"
echo ""

VOICE_MODEL="../checkpoints/voice_clones/${SPEAKER}.pth"

if [ ! -f "$VOICE_MODEL" ]; then
    echo "❌ Voice model not found: $VOICE_MODEL"
    echo "Fine-tune first: bash finetuning/finetune_voice.sh voice_samples/${SPEAKER}.wav $SPEAKER"
    exit 1
fi

python3 synthesize.py \
    --text "$TEXT" \
    --speaker_model "$VOICE_MODEL" \
    --output_path "../$OUTPUT"

echo "✅ Synthesis complete!"
echo "Output: $OUTPUT"
