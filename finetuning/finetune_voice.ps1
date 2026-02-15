# finetune_voice.ps1 - Fine-tune VITS on voice sample for cloning

$ErrorActionPreference = "Stop"

if ($args.Count -lt 2) {
    Write-Host "Usage: .\finetune_voice.ps1 <voice_sample.wav> <speaker_name>" -ForegroundColor Yellow
    Write-Host "Example: .\finetune_voice.ps1 voice_samples/john.wav john"
    exit 1
}

$voiceSample = $args[0]
$speakerName = $args[1]

Write-Host "🎤 Fine-tuning Voice Clone" -ForegroundColor Cyan
Write-Host "=============================" -ForegroundColor Cyan
Write-Host "Voice: $voiceSample" -ForegroundColor Yellow
Write-Host "Speaker: $speakerName" -ForegroundColor Yellow
Write-Host "This will take 1-2 hours"
Write-Host ""

if (-not (Test-Path "../checkpoints/vits_persian_final.pth")) {
    Write-Host "❌ Base model not found" -ForegroundColor Red
    Write-Host "Run Phase 1 first: .\training\train_vits.ps1 100000"
    exit 1
}

if (-not (Test-Path "../$voiceSample")) {
    Write-Host "❌ Voice sample not found: $voiceSample" -ForegroundColor Red
    exit 1
}

Write-Host "Starting fine-tuning..." -ForegroundColor Yellow
python finetune_voice.py `
    --base_model ../checkpoints/vits_persian_final.pth `
    --voice_sample "../$voiceSample" `
    --speaker_name $speakerName `
    --output_path "../checkpoints/voice_clones/$speakerName.pth" `
    --iterations 10000 `
    --learning_rate 5e-5

Write-Host "✅ Fine-tuning complete!" -ForegroundColor Green
Write-Host "Voice model saved to: checkpoints/voice_clones/$speakerName.pth"
Write-Host ""
Write-Host "Next: .\inference\synthesize.ps1 'متن فارسی' $speakerName output.wav"
