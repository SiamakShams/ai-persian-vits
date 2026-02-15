#!/usr/bin/env python3
"""Dataset loader for VITS training"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.audio_utils import load_wav, extract_mel_spectrogram, trim_silence, normalize_volume
from utils.text_utils import clean_text, text_to_sequence, get_symbol_to_id


class PersianTTSDataset(Dataset):
    """
    Dataset for Persian TTS training
    Loads audio files and text from metadata
    """
    def __init__(self, metadata_file, data_root=None, sample_rate=22050, 
                 n_fft=1024, hop_length=256, n_mels=80, 
                 max_wav_value=32768.0, trim_silence_flag=True):
        """
        Args:
            metadata_file: Path to metadata file (format: audio_path|text)
            data_root: Root directory for audio files (if relative paths in metadata)
            sample_rate: Target sample rate
            n_fft: FFT size
            hop_length: Hop length for STFT
            n_mels: Number of mel bands
            max_wav_value: Maximum wav value for normalization
            trim_silence_flag: Whether to trim silence from audio
        """
        self.metadata_file = Path(metadata_file)
        self.data_root = Path(data_root) if data_root else self.metadata_file.parent.parent
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.max_wav_value = max_wav_value
        self.trim_silence_flag = trim_silence_flag
        
        # Get symbol mapping
        self.symbol_to_id = get_symbol_to_id()
        
        # Load metadata
        self.samples = []
        self._load_metadata()
        
        print(f"  Loaded {len(self.samples)} samples from {metadata_file}")
        
    def _load_metadata(self):
        """Load metadata from file"""
        if not self.metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_file}")
        
        with open(self.metadata_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('|')
                if len(parts) < 2:
                    print(f"  Warning: Skipping invalid line {line_num}: {line}")
                    continue
                
                audio_path = parts[0]
                text = parts[1]
                
                # Handle relative and absolute paths
                audio_path = Path(audio_path)
                if not audio_path.is_absolute():
                    audio_path = self.data_root / audio_path
                
                # Skip existence check during initialization for performance
                # (will be checked during actual loading in __getitem__)
                self.samples.append({
                    'audio_path': audio_path,
                    'text': text.strip()
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a single sample
        
        Returns:
            Dictionary with:
                - text: [text_len] - text sequence (token IDs)
                - text_len: scalar - text length
                - mel: [n_mels, time] - mel spectrogram
                - mel_len: scalar - mel length
        """
        sample = self.samples[idx]
        
        # Load and process audio
        try:
            audio_path = sample['audio_path']
            audio = load_wav(str(audio_path), sr=self.sample_rate)
            
            # Trim silence if requested
            if self.trim_silence_flag:
                audio = trim_silence(audio, top_db=40)
            
            # Normalize volume
            audio = normalize_volume(audio, target_level=-20.0)
            
            # Extract mel spectrogram
            mel = extract_mel_spectrogram(
                audio,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels
            )
            
            # Normalize mel spectrogram
            mel = torch.FloatTensor(mel)
            
        except Exception as e:
            print(f"  Error loading audio {sample['audio_path']}: {e}")
            # Return a small dummy tensor to avoid breaking the batch
            mel = torch.zeros(self.n_mels, 10)
        
        # Process text
        try:
            text = sample['text']
            text_seq = text_to_sequence(text, self.symbol_to_id)
            text_tensor = torch.LongTensor(text_seq)
        except Exception as e:
            print(f"  Error processing text '{sample['text']}': {e}")
            # Return a minimal valid sequence
            text_tensor = torch.LongTensor([0])  # pad token
        
        return {
            'text': text_tensor,
            'text_len': len(text_tensor),
            'mel': mel,
            'mel_len': mel.shape[1]
        }


class PersianTTSCollate:
    """Collate function for batching variable-length samples"""
    
    def __init__(self, n_mels=80):
        self.n_mels = n_mels
    
    def __call__(self, batch):
        """
        Collate a batch of samples
        
        Args:
            batch: List of samples from dataset
            
        Returns:
            Dictionary with:
                - text: [batch_size, max_text_len] - padded text sequences
                - text_lengths: [batch_size] - text lengths
                - mel: [batch_size, n_mels, max_mel_len] - padded mel spectrograms
                - mel_lengths: [batch_size] - mel lengths
        """
        # Get lengths
        text_lengths = torch.LongTensor([x['text_len'] for x in batch])
        mel_lengths = torch.LongTensor([x['mel_len'] for x in batch])
        
        # Calculate max lengths
        max_text_len = text_lengths.max().item()
        max_mel_len = mel_lengths.max().item()
        
        # Initialize padded tensors
        batch_size = len(batch)
        text_padded = torch.LongTensor(batch_size, max_text_len).zero_()
        mel_padded = torch.FloatTensor(batch_size, self.n_mels, max_mel_len).zero_()
        
        # Fill in the data
        for i, sample in enumerate(batch):
            text = sample['text']
            mel = sample['mel']
            
            text_padded[i, :text.size(0)] = text
            mel_padded[i, :, :mel.size(1)] = mel
        
        return {
            'text': text_padded,
            'text_lengths': text_lengths,
            'mel': mel_padded,
            'mel_lengths': mel_lengths
        }


def get_data_loaders(train_file, val_file, batch_size=16, num_workers=4,
                     sample_rate=22050, n_mels=80):
    """
    Create train and validation data loaders
    
    Args:
        train_file: Path to training metadata
        val_file: Path to validation metadata
        batch_size: Batch size
        num_workers: Number of data loading workers
        sample_rate: Audio sample rate
        n_mels: Number of mel bands
        
    Returns:
        train_loader, val_loader: DataLoader objects
    """
    # Create datasets
    train_dataset = PersianTTSDataset(
        train_file,
        sample_rate=sample_rate,
        n_mels=n_mels,
        trim_silence_flag=True
    )
    
    val_dataset = PersianTTSDataset(
        val_file,
        sample_rate=sample_rate,
        n_mels=n_mels,
        trim_silence_flag=True
    )
    
    # Create collate function
    collate_fn = PersianTTSCollate(n_mels=n_mels)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader
