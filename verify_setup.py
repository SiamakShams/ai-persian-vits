#!/usr/bin/env python3
"""Verify setup and dependencies for AI Persian VITS (CPU & GPU modes)"""

import sys
import subprocess

def check_pytorch():
    """Check PyTorch installation and available hardware"""
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✓ GPU Detected: {torch.cuda.get_device_name(0)}")
            print(f"✓ CUDA in PyTorch: {torch.version.cuda}")
            memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✓ GPU Memory: {memory:.1f}GB")
            device_mode = "GPU"
        else:
            print("✓ CPU Mode Active (PyTorch CPU build)")
            device_mode = "CPU"
            import multiprocessing
            print(f"✓ CPU Cores: {multiprocessing.cpu_count()}")
        
        return True, device_mode
    except ImportError:
        print("❌ PyTorch not installed")
        return False, None

def check_dependencies():
    """Check key dependencies"""
    deps = []
    try:
        import librosa
        deps.append(f"librosa {librosa.__version__}")
        import soundfile
        deps.append(f"soundfile {soundfile.__version__}")
        import numpy
        deps.append(f"numpy {numpy.__version__}")
        import scipy
        deps.append(f"scipy {scipy.__version__}")
        import matplotlib
        deps.append(f"matplotlib {matplotlib.__version__}")
        import tensorboard
        deps.append(f"tensorboard")
        import yaml
        deps.append(f"pyyaml")
        import tqdm
        deps.append(f"tqdm")
        
        print("✓ Audio libraries installed:")
        for dep in deps:
            print(f"  - {dep}")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False

def check_project_structure():
    """Check project directories and files"""
    import os
    required_dirs = [
        "training",
        "preprocessing", 
        "inference",
        "finetuning",
        "utils",
        "datasets",
        "checkpoints",
        "outputs"
    ]
    
    missing = []
    for dir_name in required_dirs:
        if not os.path.isdir(dir_name):
            missing.append(dir_name)
    
    if missing:
        print(f"❌ Missing directories: {', '.join(missing)}")
        return False
    else:
        print("✓ Project structure complete")
        return True

def main():
    print("=" * 60)
    print("AI Persian VITS - Production Setup Verification")
    print("=" * 60)
    print()
    
    # Check PyTorch
    print("Checking PyTorch...")
    pytorch_ok, device_mode = check_pytorch()
    print()
    
    # Check Dependencies
    print("Checking Dependencies...")
    deps_ok = check_dependencies()
    print()
    
    # Check Project Structure
    print("Checking Project Structure...")
    structure_ok = check_project_structure()
    print()
    
    # Summary
    print("=" * 60)
    print(f"🖥️  Device Mode: {device_mode}")
    print()
    
    if all([pytorch_ok, deps_ok, structure_ok]):
        print("✅ All checks passed! Ready for production.")
        if device_mode == "CPU":
            print("📌 Running in CPU mode. Training will be slower than GPU.")
            print("   To enable GPU: Install CUDA 12.1+ and reinstall PyTorch")
        return 0
    else:
        print("❌ Some checks failed. Please review errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
