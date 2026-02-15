# train_vits.ps1 - Train VITS on Persian corpus

$ErrorActionPreference = "Stop"

$iterations = if ($args.Count -gt 0) { $args[0] } else { 100000 }

Write-Host "🎤 Training VITS on Persian Corpus" -ForegroundColor Cyan
Write-Host "===================================" -ForegroundColor Cyan
Write-Host "Iterations: $iterations" -ForegroundColor Yellow
Write-Host "This will take 3-5 days on RTX 5070 Ti"
Write-Host ""

if (-not (Test-Path "../datasets/processed")) {
    Write-Host "❌ Processed datasets not found" -ForegroundColor Red
    Write-Host "Run: python preprocessing/preprocess_datasets.py --output_dir datasets/processed"
    exit 1
}

Write-Host "Starting training..." -ForegroundColor Yellow
python train_vits.py `
    --data_path ../datasets/processed `
    --checkpoint_path ../checkpoints `
    --iterations $iterations `
    --batch_size 32 `
    --learning_rate 2e-4

Write-Host "✅ Training complete!" -ForegroundColor Green
Write-Host "Model saved to: checkpoints/vits_persian_final.pth"
