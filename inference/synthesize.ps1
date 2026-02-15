# synthesize.ps1 - Synthesize speech in cloned voice

$ErrorActionPreference = "Stop"

if ($args.Count -lt 3) {
    Write-Host "Usage: .\synthesize.ps1 <text> <speaker_name> <output.wav>" -ForegroundColor Yellow
    Write-Host "Example: .\synthesize.ps1 'سلام دنیا' john output.wav"
    exit 1
}

$text = $args[0]
$speaker = $args[1]
$output = $args[2]

Write-Host "🎤 Synthesizing: $text" -ForegroundColor Cyan
Write-Host "Speaker: $speaker" -ForegroundColor Yellow
Write-Host ""

$voiceModel = "../checkpoints/voice_clones/$speaker.pth"

if (-not (Test-Path $voiceModel)) {
    Write-Host "❌ Voice model not found: $voiceModel" -ForegroundColor Red
    Write-Host "Fine-tune first: .\finetuning\finetune_voice.ps1 voice_samples/$speaker.wav $speaker"
    exit 1
}

python synthesize.py `
    --text $text `
    --speaker_model $voiceModel `
    --output_path "../$output"

Write-Host "✅ Synthesis complete!" -ForegroundColor Green
Write-Host "Output: $output"
