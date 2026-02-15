# Troubleshooting

## CUDA Errors

**Error: "CUDA out of memory"**
- Edit `training/vits_config.json`: reduce `batch_size` to 16
- Enable `use_fp16: true` and `gradient_checkpointing: true`

**Error: "CUDA illegal memory access"**
- Update drivers: `sudo apt-get install -y --only-upgrade nvidia-driver-560`
- Update PyTorch: `pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu121`

**Error: "CuBLAS failed"**
- Install cuBLAS: `pip install nvidia-cublas-cu12==12.1.3.1`

## Dataset Errors

**Error: "Dataset not found"**
- Place datasets in `datasets/raw/`
- Run preprocessing: `python3 preprocessing/preprocess_datasets.py --output_dir datasets/processed`

## Voice Cloning

**Poor voice quality:**
- Use 30-60 second sample (not 10 seconds)
- Ensure clear speech, minimal background noise
- Increase fine-tuning iterations: 10000 → 20000

**Training too slow:**
- Check GPU usage: `nvidia-smi`
- Check CUDA version: `nvcc --version` (must be 12.1+)
