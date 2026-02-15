#!/usr/bin/env python3
"""Audio utility functions"""

import numpy as np
import librosa
import soundfile as sf
from scipy import signal

def load_wav(path, sr=22050):
    """
    Load audio file
    
    Args:
        path: Path to audio file
        sr: Target sample rate
        
    Returns:
        audio: Audio as numpy array, normalized to [-1, 1]
    """
    audio, orig_sr = librosa.load(path, sr=sr, mono=True)
    return audio

def save_wav(path, audio, sr=22050):
    """
    Save audio to file
    
    Args:
        path: Output path
        audio: Audio as numpy array
        sr: Sample rate
    """
    sf.write(path, audio, sr, subtype='PCM_16')

def trim_silence(audio, top_db=40, frame_length=2048, hop_length=512):
    """
    Trim silence from audio
    
    Args:
        audio: Audio as numpy array
        top_db: Threshold in decibels below reference to consider as silence
        frame_length: Frame length for energy calculation
        hop_length: Hop length for energy calculation
        
    Returns:
        trimmed_audio: Audio with silence removed
    """
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db, 
                                     frame_length=frame_length,
                                     hop_length=hop_length)
    return trimmed

def normalize_volume(audio, target_level=-20.0):
    """
    Normalize audio volume to target dBFS
    
    Args:
        audio: Audio as numpy array
        target_level: Target volume in dBFS
        
    Returns:
        normalized_audio: Volume-normalized audio
    """
    # Calculate current RMS in dBFS
    rms = np.sqrt(np.mean(audio**2))
    if rms == 0:
        return audio
    
    current_db = 20 * np.log10(rms)
    gain = target_level - current_db
    
    # Apply gain
    normalized = audio * (10 ** (gain / 20))
    
    # Prevent clipping
    max_val = np.abs(normalized).max()
    if max_val > 1.0:
        normalized = normalized / max_val * 0.95
    
    return normalized

def resample(audio, orig_sr, target_sr):
    """
    Resample audio to target sample rate
    
    Args:
        audio: Audio as numpy array
        orig_sr: Original sample rate
        target_sr: Target sample rate
        
    Returns:
        resampled_audio: Resampled audio
    """
    if orig_sr == target_sr:
        return audio
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)

def extract_mel_spectrogram(audio, sr=22050, n_fft=1024, hop_length=256, 
                           n_mels=80, fmin=0.0, fmax=8000.0):
    """
    Extract mel spectrogram from audio
    
    Args:
        audio: Audio as numpy array
        sr: Sample rate
        n_fft: FFT window size
        hop_length: Hop length for STFT
        n_mels: Number of mel bands
        fmin: Minimum frequency
        fmax: Maximum frequency (None = sr/2)
        
    Returns:
        mel_spec: Mel spectrogram (n_mels, time) in dB scale
    """
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax if fmax else sr / 2,
        power=2.0
    )
    
    # Convert to dB scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db

def mel_spectrogram_to_audio(mel_spec_db, sr=22050, n_fft=1024, hop_length=256, 
                            n_iter=32, fmin=0.0, fmax=8000.0):
    """
    Convert mel spectrogram back to audio using Griffin-Lim algorithm
    
    Args:
        mel_spec_db: Mel spectrogram in dB scale
        sr: Sample rate
        n_fft: FFT window size
        hop_length: Hop length for STFT
        n_iter: Number of Griffin-Lim iterations
        fmin: Minimum frequency
        fmax: Maximum frequency
        
    Returns:
        audio: Reconstructed audio waveform
    """
    # Convert from dB back to power scale
    mel_spec = librosa.db_to_power(mel_spec_db)
    
    # Inverse mel to linear spectrogram
    linear_spec = librosa.feature.inverse.mel_to_stft(
        mel_spec,
        sr=sr,
        n_fft=n_fft,
        fmin=fmin,
        fmax=fmax if fmax else sr / 2
    )
    
    # Griffin-Lim reconstruction
    audio = librosa.griffinlim(
        linear_spec,
        n_iter=n_iter,
        hop_length=hop_length,
        win_length=n_fft
    )
    
    return audio

def preprocess_audio(audio_path, sr=22050, trim_silence_flag=True, 
                    normalize=True, top_db=40):
    """
    Complete audio preprocessing pipeline
    
    Args:
        audio_path: Path to audio file
        sr: Target sample rate
        trim_silence_flag: Whether to trim silence
        normalize: Whether to normalize volume
        top_db: Threshold for silence trimming
        
    Returns:
        audio: Preprocessed audio
        sr: Sample rate
    """
    # Load audio
    audio = load_wav(audio_path, sr=sr)
    
    # Trim silence
    if trim_silence_flag:
        audio = trim_silence(audio, top_db=top_db)
    
    # Normalize volume
    if normalize:
        audio = normalize_volume(audio)
    
    return audio, sr
