# Getting Started - AI Persian VITS

### System configuration
| Component | Details |
| :--- | :--- |
| **Environment** | Conda (`vits-env`) |
| **Python** | 3.11.14 |
| **PyTorch** | 2.10.0+cpu |
| **Device** | CPU (20 cores) |
| **Device Mode** | CPU Training |
| **Status** | ✅ Production Ready |


## 1. First Time Setup (One-Time)

```powershell
# Run setup script
.\setup-conda.ps1

# Verify everything works
conda activate vits-env
python verify_setup.py
```

Takes ~5 minutes. Creates `vits-env` Conda environment with all dependencies.

---

## 2. Activate Environment (Every Time)

```powershell
# Option A: Quick launcher
.\activate-env.ps1 <command>

# Option B: Manual activation
conda activate vits-env
```

**Available commands:**
- `verify` - Check setup
- `train` - Start training
- `preprocess` - Process datasets
- `infer` - Run inference
- `finetune` - Fine-tune model
- `tensorboard` - Monitor training
- `jupyter` - Launch notebooks

---

## 3. Prepare Data

Place Persian audio datasets in `datasets/raw/`:
```
datasets/raw/
├── GPTInformal-Persian/
├── Mana-TTS/
└── Quran-Persian/
```

---

## 4. Preprocess Datasets

```powershell
conda activate vits-env
cd preprocessing
python preprocess_datasets.py --input_dir ../datasets/raw --output_dir ../datasets/processed
```

---

## 5. Train Base Model (Phase 1)

```powershell
conda activate vits-env
cd training

# Incremental training (recommended):
python train_vits.py --data_path ../datasets/processed --checkpoint_path ../checkpoints --iterations 50000 --batch_size 32 --learning_rate 2e-4
python train_vits.py --data_path ../datasets/processed --checkpoint_path ../checkpoints --iterations 100000 --batch_size 32 --learning_rate 2e-4 --resume
python train_vits.py --data_path ../datasets/processed --checkpoint_path ../checkpoints --iterations 200000 --batch_size 32 --learning_rate 2e-4 --resume
python train_vits.py --data_path ../datasets/processed --checkpoint_path ../checkpoints --iterations 500000 --batch_size 32 --learning_rate 2e-4 --resume
```

**Expect:** 
- CPU: 5-10 hours per 1000 steps
- Monitor: `tensorboard --logdir=../outputs/logs`

---

## 6. Fine-tune on Voice Sample (Phase 2)

```powershell
conda activate vits-env
cd finetuning
python finetune_voice.py --config finetune_config.json --num_epochs 50
```

---

## 7. Generate Speech (Inference)

```powershell
conda activate vits-env
cd inference
python synthesize.py --text "سلام دنیا" --model_path ../checkpoints/vits_persian_final.pth --output_path output.wav
```

---

## 🎯 What to Expect

- **Setup**: 5 minutes
- **Preprocessing**: 10-30 minutes (depends on data)
- **Training (CPU)**: 2-5 hours per checkpoint
- **Fine-tuning**: 30-60 minutes
- **Inference**: < 1 second per sentence

---

## 📖 Need More Info?

- **Full details**: See `README.md`
- **Troubleshooting**: See `TROUBLESHOOTING.md`
- **Production setup**: See `PRODUCTION_SETUP.md`
- **Quick reference**: See `QUICK_START.md`


# When you return in 1 year

## When you first start:
```shell
1. Read GETTINGSTARTED.md (5 minutes)
2. Follow the 7 exact steps
3. Refer to README.md for details as needed
```

```shell
1. Run: python verify_setup.py
2. Read: README.md (refreshes memory on goal & structure)
3. Continue from: GETTINGSTARTED.md step 2
```

``shell
- GETTINGSTARTED.md - what to do
- QUICK_START.md - command reference
- README.md - when you need to understand why
```