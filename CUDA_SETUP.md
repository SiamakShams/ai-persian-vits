```markdown name=CUDA_SETUP.md
# CUDA 12.x Setup for RTX 5070 Ti

**CRITICAL:** RTX 5070 Ti does NOT work with CUDA 11.8. You MUST use CUDA 12.1+

## Check Current Setup

```bash
nvidia-smi
# Should show: CUDA Version: 12.1 or higher

nvcc --version
# Should show: release 12.1 or higher
