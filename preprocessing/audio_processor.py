#!/usr/bin/env python3
"""Audio processing utilities for VITS training"""

import sys
from pathlib import Path

# Import from utils
sys.path.append(str(Path(__file__).parent.parent))
from utils.audio_utils import (
    load_wav,
    save_wav,
    extract_mel_spectrogram,
    normalize_volume,
    trim_silence,
    resample,
    preprocess_audio
)

# Re-export for convenience
__all__ = [
    'load_wav',
    'save_wav',
    'extract_mel_spectrogram',
    'normalize_volume',
    'trim_silence',
    'resample',
    'preprocess_audio'
]

if __name__ == "__main__":
    print("Audio processor ready")
    print("Available functions:")
    for func in __all__:
        print(f"  - {func}")
