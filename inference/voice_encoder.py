#!/usr/bin/env python3
"""Voice encoder for speaker embedding extraction (optional component)"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.audio_utils import load_wav, extract_mel_spectrogram


class VoiceEncoder(nn.Module):
    """
    Speaker encoder network for voice cloning
    
    Note: This is an optional component for multi-speaker VITS.
    Basic VITS training doesn't require this.
    """
    
    def __init__(self, mel_n_channels=80, model_hidden_size=256, model_embedding_size=256):
        super(VoiceEncoder, self).__init__()
        self.mel_n_channels = mel_n_channels
        self.model_hidden_size = model_hidden_size
        self.model_embedding_size = model_embedding_size
        
        # LSTM layers for temporal modeling
        self.lstm = nn.LSTM(
            input_size=mel_n_channels,
            hidden_size=model_hidden_size,
            num_layers=3,
            batch_first=True,
            bidirectional=False
        )
        
        # Linear layer for embedding projection
        self.linear = nn.Linear(model_hidden_size, model_embedding_size)
        
        # Activation
        self.relu = nn.ReLU()
        
    def forward(self, mels):
        """
        Extract speaker embedding from mel spectrogram
        
        Args:
            mels: Mel spectrogram tensor [batch, mel_channels, time]
            
        Returns:
            embeddings: Speaker embedding tensor [batch, embedding_size]
        """
        # Transpose to [batch, time, mel_channels]
        mels = mels.transpose(1, 2)
        
        # LSTM processing
        _, (hidden, _) = self.lstm(mels)
        
        # Use last hidden state
        embeds_raw = hidden[-1]
        
        # Project to embedding space
        embeds = self.linear(embeds_raw)
        embeds = self.relu(embeds)
        
        # L2 normalization
        embeds = embeds / torch.norm(embeds, dim=1, keepdim=True)
        
        return embeds
    
    def embed_utterance(self, wav_path, device='cpu'):
        """
        Extract speaker embedding from audio file
        
        Args:
            wav_path: Path to audio file
            device: Device to run on ('cpu' or 'cuda')
            
        Returns:
            embedding: Speaker embedding vector
        """
        # Load audio
        audio = load_wav(wav_path, sr=22050)
        
        # Extract mel spectrogram
        mel = extract_mel_spectrogram(audio, sr=22050, n_mels=self.mel_n_channels)
        
        # Convert to tensor
        mel_tensor = torch.FloatTensor(mel).unsqueeze(0).to(device)
        
        # Run forward pass
        self.eval()
        with torch.no_grad():
            embedding = self.forward(mel_tensor)
        
        return embedding.squeeze(0).cpu().numpy()


if __name__ == "__main__":
    print("Voice encoder ready")
    print("Note: This is an optional component for multi-speaker VITS")
    print("Basic single-speaker VITS training doesn't require this module")

def load_voice_encoder(checkpoint_path):
    """Load pretrained voice encoder"""
    # TODO: Implement checkpoint loading
    pass

if __name__ == "__main__":
    # Test voice encoder
    print("Voice encoder ready")
    encoder = VoiceEncoder()
    print(f"Encoder parameters: {sum(p.numel() for p in encoder.parameters())}")
