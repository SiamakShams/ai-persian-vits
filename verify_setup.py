#!/usr/bin/env python3
"""Verify CUDA 12.x and all dependencies for AI Persian VITS"""

import sys
import subprocess

def check_cuda():
    """Check CUDA 12.1+"""
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if 'release 12.1' in result.stdout or 'release 12.2' in result.stdout:
            print("✓ CUDA 12.1+ installed")
            return True
        else:
            print("❌ CUDA 12.1+ required (RTX 5070 Ti only works with CUDA 12.1+)")
            return False
    except FileNotFoundError:
        print("❌ CUDA toolkit not found. Install from CUDA_SETUP.md")
        return False

def check_pytorch():
    """Check PyTorch with CUDA 12.1"""
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        
        if not torch.cuda.is_available():
            print("❌ CUDA not available in PyTorch")
            return False
        
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA in PyTorch: {torch.version.cuda}")
        
        if torch.version.cuda < "12.1":
            print("❌ PyTorch CUDA 12.1+ required")
            return False
        
        memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ GPU Memory: {memory:.1f}GB")
        
        return True
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def check_dependencies():
    """Check key dependencies"""
    try:
        import librosa
        import soundfile
        import numpy
        import scipy
        print("✓ Audio libraries installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False

def main():
    print("=" * 50)
    print("AI Persian VITS - Setup Verification")
    print("=" * 50)
    print()
    
    checks = [
        ("CUDA 12.1+", check_cuda),
        ("PyTorch CUDA", check_pytorch),
        ("Dependencies", check_dependencies),
    ]
    
    results = []
    for name, check in checks:
        print(f"Checking {name}...")
        results.append(check())
        print()
    
    print("=" * 50)
    if all(results):
        print("✅ All checks passed! Ready for training.")
        return 0
    else:
        print("❌ Some checks failed. See CUDA_SETUP.md")
        return 1

if __name__ == "__main__":
    sys.exit(main())
