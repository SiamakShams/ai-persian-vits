#!/usr/bin/env powershell
# AI Persian VITS - Production Setup with Conda
# Usage: powershell -ExecutionPolicy Bypass -File setup-conda.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "AI Persian VITS - Conda Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if conda is installed
Write-Host "Checking Conda installation..." -ForegroundColor Yellow
try {
    $condaVersion = conda --version 2>$null
    Write-Host "✓ $condaVersion" -ForegroundColor Green
}
catch {
    Write-Host "✗ Conda not found. Please install Miniconda or Conda first." -ForegroundColor Red
    Write-Host "Download from: https://docs.conda.io/projects/miniconda/en/latest/" -ForegroundColor Yellow
    exit 1
}

Write-Host ""

# Check if vits-env already exists
Write-Host "Checking for existing 'vits-env'..." -ForegroundColor Yellow
$envExists = conda env list | Select-String "vits-env"

if ($envExists) {
    Write-Host "Environment 'vits-env' already exists" -ForegroundColor Green
    $remove = Read-Host "Remove and recreate? (y/n)"
    if ($remove.ToLower() -eq 'y') {
        Write-Host "Removing old environment..." -ForegroundColor Yellow
        conda remove -n vits-env --all -y
    }
    else {
        Write-Host "Skipping environment creation" -ForegroundColor Green
    }
}

Write-Host ""

# Create environment from environment.yml or manual setup
$useYaml = Read-Host "Use environment.yml? (y/n) [default: y]"
if ($useYaml.ToLower() -ne 'n') {
    Write-Host "Creating environment from environment.yml..." -ForegroundColor Yellow
    
    if (Test-Path "environment.yml") {
        conda env create -f environment.yml
    }
    else {
        Write-Host "✗ environment.yml not found in current directory" -ForegroundColor Red
        Write-Host "Using manual setup instead..." -ForegroundColor Yellow
        $useYaml = 'n'
    }
}

if ($useYaml.ToLower() -eq 'n') {
    Write-Host "Creating environment manually..." -ForegroundColor Yellow
    
    # Create base environment
    Write-Host "Step 1: Creating environment with Python 3.11..." -ForegroundColor Cyan
    conda create -n vits-env python=3.11 -y
    
    # Activate and install PyTorch
    Write-Host "Step 2: Installing PyTorch (CPU)..." -ForegroundColor Cyan
    conda activate vits-env
    pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cpu --quiet
    
    # Install other dependencies
    Write-Host "Step 3: Installing Python dependencies..." -ForegroundColor Cyan
    pip install librosa soundfile numpy scipy matplotlib tensorboard pyyaml tqdm jupyter --quiet
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Verifying Setup..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Verify installation
conda activate vits-env
python verify_setup.py

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setup Complete! ✓" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Activate environment: conda activate vits-env" -ForegroundColor White
Write-Host "2. Check documentation: Read PRODUCTION_SETUP.md" -ForegroundColor White
Write-Host "3. Prepare datasets in: datasets/raw/" -ForegroundColor White
Write-Host "4. Run preprocessing: cd preprocessing && python preprocess_datasets.py" -ForegroundColor White
Write-Host "5. Train model: cd training && python train_vits.py" -ForegroundColor White
Write-Host ""
