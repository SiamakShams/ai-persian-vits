#!/usr/bin/env powershell
# AI Persian VITS - Quick Production Environment Launcher

param(
    [Parameter(Mandatory=$false, ValueFromRemainingArguments=$true)]
    [string[]]$Command
)

Write-Host "Activating vits-env..." -ForegroundColor Cyan

# Activate conda environment
conda activate vits-env

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Could not activate vits-env" -ForegroundColor Red
    Write-Host "Run setup-conda.ps1 first to create the environment" -ForegroundColor Yellow
    exit 1
}

Write-Host "✓ Environment activated" -ForegroundColor Green
Write-Host ""

if ($Command.Count -eq 0) {
    Write-Host "Available commands:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  verify         - Run setup verification" -ForegroundColor White
    Write-Host "  train          - Start training pipeline" -ForegroundColor White
    Write-Host "  preprocess     - Preprocess datasets" -ForegroundColor White
    Write-Host "  infer          - Run inference (text-to-speech)" -ForegroundColor White
    Write-Host "  finetune       - Fine-tune existing model" -ForegroundColor White
    Write-Host "  tensorboard    - Launch TensorBoard monitoring" -ForegroundColor White
    Write-Host "  shell          - Open interactive Python shell" -ForegroundColor White
    Write-Host "  jupyter        - Launch Jupyter notebook" -ForegroundColor White
    Write-Host ""
    Write-Host "Usage: .\activate-env.ps1 <command>" -ForegroundColor Cyan
    Write-Host "Example: .\activate-env.ps1 verify" -ForegroundColor Cyan
}
else {
    $cmd = $Command[0].ToLower()
    
    switch ($cmd) {
        "verify" {
            Write-Host "Running verification..." -ForegroundColor Cyan
            python verify_setup.py
        }
        "train" {
            Write-Host "Starting training..." -ForegroundColor Cyan
            Set-Location training
            python train_vits.py --config vits_config.json
        }
        "preprocess" {
            Write-Host "Starting preprocessing..." -ForegroundColor Cyan
            Set-Location preprocessing
            python preprocess_datasets.py --input_dir ../datasets/raw --output_dir ../datasets/processed
        }
        "infer" {
            Write-Host "Starting inference shell..." -ForegroundColor Cyan
            Set-Location inference
            python -i synthesize.py
        }
        "finetune" {
            Write-Host "Starting fine-tuning..." -ForegroundColor Cyan
            Set-Location finetuning
            python finetune_voice.py --config finetune_config.json
        }
        "tensorboard" {
            Write-Host "Launching TensorBoard..." -ForegroundColor Cyan
            Write-Host "Open browser to http://localhost:6006" -ForegroundColor Green
            tensorboard --logdir=outputs/logs
        }
        "shell" {
            Write-Host "Starting Python shell..." -ForegroundColor Cyan
            python
        }
        "jupyter" {
            Write-Host "Launching Jupyter..." -ForegroundColor Cyan
            Write-Host "Access at http://localhost:8888" -ForegroundColor Green
            jupyter notebook --notebook-dir=./
        }
        default {
            Write-Host "Unknown command: $cmd" -ForegroundColor Red
            Write-Host "Run '\activate-env.ps1' without arguments for help" -ForegroundColor Yellow
        }
    }
}
