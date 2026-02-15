# AI Persian VITS - Voice Synthesis & Cloning

Persian text-to-speech using VITS with two-phase training and voice cloning.

**Goal:** Train a Persian-optimized VITS model, then fine-tune it on individual voices for high-quality synthesis.

---

## 🎯 What Is This Project?

**VITS** is a neural vocoder that generates speech from text. This implementation:
- Trains on multiple Persian datasets to learn Persian phonetics
- Fine-tunes on individual voices (10-60 seconds) for voice cloning
- Produces natural, expressive Persian speech

**Two-Phase Approach:**
1. **Phase 1 (One-time)**: Train base VITS model on Persian corpus → `vits_persian_final.pth`
2. **Phase 2 (Per-voice)**: Fine-tune base model on individual speaker samples → Voice clones

---

## 💻 System Configuration

**Current Setup:**
- **Environment**: Conda (vits-env)
- **Python**: 3.11
- **PyTorch**: 2.10.0+cpu
- **Device**: CPU (20 cores)
- **Status**: ✅ Production Ready

**To Upgrade to GPU** (RTX 5070 Ti):
```powershell
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## 📂 Project Structure & File Purposes

```
ai-persian-vits/
│
├── 📋 DOCUMENTATION
│   ├── README.md                 ← You are here (source of truth)
│   ├── GETTINGSTARTED.md        ← Quick steps to get running
│   ├── PRODUCTION_SETUP.md      ← Detailed setup guide
│   ├── QUICK_START.md           ← Quick reference card
│   ├── STATUS_REPORT.md         ← Deployment checklist
│   ├── TROUBLESHOOTING.md       ← Known issues & solutions
│   ├── CUDA_SETUP.md            ← GPU configuration
│   └── COPILOT.md               ← AI assistant notes
│
├── 🔧 SETUP & CONFIGURATION
│   ├── setup-conda.ps1          ← First-time environment setup
│   ├── activate-env.ps1         ← Daily environment launcher
│   ├── environment.yml          ← Conda environment export
│   ├── verify_setup.py          ← Verify all systems ready
│   ├── requirements.txt          ← Pip dependencies (reference)
│   └── setup.sh                  ← Legacy bash setup
│
├── ⚙️ PREPROCESSING (Convert raw data → training datasets)
│   ├── preprocess_datasets.py   ← Main preprocessing script
│   ├── audio_processor.py       ← Audio normalization & features
│   ├── text_processor.py        ← Persian text normalization
│   └── README.md (in folder)    ← Dataset format docs
│
├── 🚂 TRAINING (Phase 1: Train base VITS model)
│   ├── train_vits.py            ← Main training script
│   ├── train_vits.ps1           ← PowerShell trainer
│   ├── train_vits.sh            ← Bash trainer
│   ├── vits_config.json         ← Model architecture & hyperparams
│   └── README.md (in folder)    ← Training guide
│
├── 🎤 FINETUNING (Phase 2: Clone individual voices)
│   ├── finetune_voice.py        ← Fine-tuning script
│   ├── finetune_voice.ps1       ← PowerShell fine-tuner
│   ├── finetune_voice.sh        ← Bash fine-tuner
│   ├── finetune_config.json     ← Fine-tuning hyperparams
│   └── README.md (in folder)    ← Fine-tuning guide
│
├── 🔊 INFERENCE (Generate speech from text)
│   ├── synthesize.py            ← Main inference script
│   ├── synthesize.ps1           ← PowerShell synthesizer
│   ├── synthesize.sh            ← Bash synthesizer
│   ├── voice_encoder.py         ← Extract speaker embeddings
│   └── README.md (in folder)    ← Inference guide
│
├── 📦 UTILITIES (Helper functions)
│   ├── audio_utils.py           ← Audio I/O & processing
│   ├── text_utils.py            ← Text processing & cleanup
│   ├── file_utils.py            ← File operations
│   └── __init__.py              ← Module initialization
│
├── 📂 DATA DIRECTORIES
│   ├── datasets/
│   │   ├── raw/                 ← Place source datasets here
│   │   │   ├── GPTInformal-Persian/
│   │   │   ├── Mana-TTS/
│   │   │   ├── ParsiGoo/
│   │   │   └── Quran-Persian/
│   │   └── processed/           ← Created by preprocessing
│   │       ├── train.txt        ← Training data list
│   │       ├── val.txt          ← Validation data list
│   │       ├── metadata.txt     ← Audio metadata
│   │       └── summary.json     ← Dataset statistics
│   │
│   ├── checkpoints/             ← Saved models
│   │   └── vits_persian_final.pth  ← Base model (Phase 1 output)
│   │
│   └── outputs/                 ← Training/inference outputs
│       ├── logs/                ← TensorBoard event files
│       ├── checkpoints/         ← Intermediate checkpoints
│       └── inference/           ← Generated audio samples
│
└── 🐳 CONTAINERIZATION
    └── Dockerfile              ← Docker container definition
```

---

## 🚀 Quick Start

### 1️⃣ First-Time Setup (5 minutes)

```powershell
# Navigate to project
cd d:\Development\ai-persian-vits

# Create Conda environment with all dependencies
.\setup-conda.ps1

# Verify everything works
conda activate vits-env
python verify_setup.py
```

✅ Output should show: CPU cores available, all packages installed.

### 2️⃣ Prepare Data

Download Persian TTS datasets and place in `datasets/raw/`:
- [GPTInformal-Persian](https://huggingface.co/datasets/sinch/GPTInformal-Persian)
- [Mana-TTS](https://huggingface.co/datasets/persiannlp/mana-tts)
- [ParsiGoo](https://www.kaggle.com/datasets/matinkashefi/parsigoo-dataset)
- [Quran-Persian](https://github.com/persiannlp/quran)

Or use **dummy data** (for testing):
```powershell
conda activate vits-env
cd preprocessing
python preprocess_datasets.py --generate_dummy
```

### 3️⃣ Preprocess Datasets (Phase 0)

Convert raw datasets to training format:
```powershell
conda activate vits-env
cd preprocessing
python preprocess_datasets.py --input_dir ../datasets/raw --output_dir ../datasets/processed
```

Creates:
- `train.txt` / `val.txt` - File lists with text & audio paths
- `metadata.txt` - Speaker info & statistics
- `summary.json` - Dataset overview

**Expect:** 10-30 minutes depending on data size.

### 4️⃣ Train Base Model (Phase 1) - One Time Only

```powershell
conda activate vits-env
cd training
python train_vits.py --config vits_config.json --epochs 100
```

**Expect:**
- **CPU**: 2-8 hours per epoch (100+ epochs = weeks of training)
- **GPU**: 30 minutes per epoch (after CUDA install)
- **Output**: `checkpoints/vits_persian_final.pth`

**Monitor training:**
```powershell
# In another terminal
conda activate vits-env
tensorboard --logdir=outputs/logs
# Visit http://localhost:6006
```

### 5️⃣ Fine-tune on Individual Voice (Phase 2)

Once you have a trained base model:

```powershell
conda activate vits-env
cd finetuning
python finetune_voice.py --config finetune_config.json --num_epochs 50
```

**Expect:** 30-60 minutes per voice on CPU.

### 6️⃣ Generate Speech (Inference)

```powershell
conda activate vits-env
cd inference
python synthesize.py \
    --text "سلام دنیا" \
    --model_path ../checkpoints/vits_persian_final.pth \
    --output_path output.wav
```

**Expect:** < 1 second to generate speech.

---

## ⏱️ Timeline Expectations

| Stage | Duration | Frequency |
|-------|----------|-----------|
| Setup | 5 minutes | Once |
| Preprocessing | 10-30 min | Once per dataset |
| Phase 1 (Training) | 2-8 hrs/epoch | Once (100+ epochs) |
| Phase 2 (Fine-tune) | 30-60 min | Once per voice |
| Inference | < 1 sec | Per sentence |

---

## 🎮 Using Quick Launcher

Instead of manual activation, use:

```powershell
.\activate-env.ps1 verify       # Check setup
.\activate-env.ps1 train        # Start training
.\activate-env.ps1 preprocess   # Preprocess data
.\activate-env.ps1 infer        # Run inference
.\activate-env.ps1 finetune     # Fine-tune voice
.\activate-env.ps1 tensorboard  # Monitor training
.\activate-env.ps1 jupyter      # Launch notebook
```

---

## 📋 Configuration Files

### `training/vits_config.json`
Controls:
- Model architecture (layers, hidden dims)
- Learning rate, batch size
- Number of epochs
- Audio preprocessing parameters

### `finetuning/finetune_config.json`
Controls:
- Fine-tuning learning rate (usually lower than training)
- Number of fine-tuning epochs
- Freeze which layers

---

## 🔄 Workflow Summary

```
Raw Audio Data
    ↓
[Preprocessing] → Normalized audio + text pairs
    ↓
[Phase 1: Training] → Base VITS model (vits_persian_final.pth)
    ↓
[Phase 2: Fine-tuning] → Voice-specific model
    ↓
[Inference] → Persian speech synthesis
```

---

## 📚 Full Documentation Files

| File | When to Read |
|------|--------------|
| **GETTINGSTARTED.md** | First thing - quick steps to run everything now |
| **README.md** | Now - full understanding of project (this file) |
| **PRODUCTION_SETUP.md** | Deep setup details, configuration options |
| **QUICK_START.md** | Quick reference during development |
| **STATUS_REPORT.md** | Deployment checklist, verification results |
| **TROUBLESHOOTING.md** | When something breaks |
| **CUDA_SETUP.md** | Only if upgrading to GPU |

---

## 🐛 Common Issues

| Issue | Solution |
|-------|----------|
| PyTorch not found | Run `.\setup-conda.ps1` again |
| Out of memory | Reduce batch_size in config JSON |
| Out of disk space | Free up space or use smaller dataset |
| CUDA errors | Ignore if using CPU; skip CUDA_SETUP.md |
| Audio not loading | Check dataset format in preprocessing README |

See `TROUBLESHOOTING.md` for more.

---

## 🔗 Key Commands Reference

```powershell
# Setup
.\setup-conda.ps1                                    # One-time setup
conda activate vits-env                             # Activate environment
python verify_setup.py                              # Verify all systems

# Quick Launcher
.\activate-env.ps1 <command>                        # See options above

# Manual Commands
cd preprocessing; python preprocess_datasets.py     # Preprocess data
cd training; python train_vits.py                   # Train base model
cd finetuning; python finetune_voice.py            # Fine-tune voice
cd inference; python synthesize.py                  # Generate speech
tensorboard --logdir=outputs/logs                   # Monitor training
```

---

## 🎯 One Year From Now

If you're returning to this project:

1. **Remember the goal**: Train Persian VITS → Fine-tune on voices → Synthesis
2. **Check status**: Run `python verify_setup.py` to ensure environment is ready
3. **Review structure**: This README explains every folder and file
4. **Start with**: `GETTINGSTARTED.md` for immediate steps
5. **Existing models**: Check `checkpoints/` for previously trained models
6. **Outputs**: Check `outputs/` for previous training logs and generated samples

---

## 📞 Quick Help

- **Initial setup problems?** → See `GETTINGSTARTED.md`
- **Want full details?** → See `PRODUCTION_SETUP.md`
- **Quick reference?** → See `QUICK_START.md`
- **Something broke?** → See `TROUBLESHOOTING.md`
- **Need GPU?** → See `CUDA_SETUP.md`

---

**Version**: 2.0 (Production Ready)  
**Last Updated**: February 2026  
**Device Mode**: CPU (GPU Ready)  
**Status**: ✅ Ready for Training
