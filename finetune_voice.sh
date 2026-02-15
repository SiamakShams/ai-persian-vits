#!/bin/bash
# Fine-tune VITS on voice sample for cloning

set -e

if [ $# -lt 2 ]; then
    echo "Usage: bash finetune_voice.sh <voice_sample.wav> <speaker_name>"
    echo "Example: bash finetune_voice.sh voice_samples/john.wav john"
    exit 1
fi

VOICE_SAMPLE="$1"
SPEAKER_NAME="$2"

echo "🎤 Fine-tuning Voice Clone"
echo "============================="
echo "Voice: $VOICE_SAMPLE"
echo "Speaker: $SPEAKER_NAME"
echo "This will take 1-2 hours"
echo ""

if [ ! -f "checkpoints/vits_persian_final.pth" ]; then
    echo "❌ Base model not found"
    echo "Run Phase 1 first: bash train_vits.sh 100000"
    exit 1
fi

if [ ! -f "$VOICE_SAMPLE" ]; then
    echo "❌ Voice sample not found: $VOICE_SAMPLE"
    exit 1
fi

echo "Starting fine-tuning..."
python3 finetuning/finetune_voice.py \
    --base_model checkpoints/vits_persian_final.pth \
    --voice_sample "$VOICE_SAMPLE" \
    --speaker_name "$SPEAKER_NAME" \
    --output_path "checkpoints/voice_clones/${SPEAKER_NAME}.pth" \
    --iterations 10000 \
    --learning_rate 5e-5

echo "✅ Fine-tuning complete!"
echo "Voice model saved to: checkpoints/voice_clones/${SPEAKER_NAME}.pth"
echo ""
echo "Next: bash synthesize.sh 'متن فارسی' $SPEAKER_NAME output.wav"
