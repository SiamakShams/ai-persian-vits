#!/usr/bin/env python3
"""Synthesize speech using trained VITS model"""

import argparse
import torch
from pathlib import Path
import numpy as np
import json
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from training.vits_model import VITS
from utils.text_utils import clean_text, text_to_sequence, get_symbol_list, get_symbol_to_id
from utils.audio_utils import save_wav, mel_spectrogram_to_audio

def synthesize(text, model_path, output_path, length_scale=1.0):
    """Synthesize speech from text"""
    print("🎤 Synthesizing Speech")
    print("=" * 50)
    print(f"Text: {text}")
    print(f"Model: {model_path}")
    print(f"Output: {output_path}")
    print(f"Length scale: {length_scale}")
    print()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print()
    
    # Model configuration (must match training config)
    n_vocab = len(get_symbol_list())
    spec_channels = 80
    hidden_channels = 192
    filter_channels = 768
    n_heads = 2
    n_layers = 6
    
    # Load model
    print("Loading model...")
    model = VITS(
        n_vocab=n_vocab,
        spec_channels=spec_channels,
        hidden_channels=hidden_channels,
        filter_channels=filter_channels,
        n_heads=n_heads,
        n_layers=n_layers,
        kernel_size=3,
        p_dropout=0.1
    ).to(device)
    
    if Path(model_path).exists():
        print(f"  Loading checkpoint: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            step = checkpoint.get('step', 'unknown')
            print(f"  ✓ Loaded model from step {step}")
        else:
            model.load_state_dict(checkpoint)
            print(f"  ✓ Loaded model")
        
        # Load metadata if available
        metadata_path = Path(model_path).with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                print(f"  Speaker: {metadata.get('speaker_name', 'default')}")
    else:
        print(f"  ⚠️  Model not found: {model_path}")
        print(f"  ⚠️  Using untrained model (output will be random)")
    
    print()
    
    # Process text
    print("Processing text...")
    text_clean = clean_text(text)
    print(f"  Original: {text}")
    print(f"  Cleaned: {text_clean}")
    
    symbol_to_id = get_symbol_to_id()
    text_seq = text_to_sequence(text_clean, symbol_to_id)
    text_tensor = torch.LongTensor(text_seq).unsqueeze(0).to(device)
    text_lengths = torch.LongTensor([len(text_seq)]).to(device)
    
    print(f"  Text length: {len(text_seq)} symbols")
    print()
    
    # Generate speech
    print("Generating speech...")
    model.eval()
    with torch.no_grad():
        try:
            mel_pred, mel_mask = model.infer(text_tensor, text_lengths, length_scale=length_scale)
            mel_spectrogram = mel_pred.squeeze(0).cpu().numpy()
            
            # Get actual length from mask
            mel_len = int(mel_mask.squeeze().sum().item())
            mel_spectrogram = mel_spectrogram[:, :mel_len]
            
            print(f"  ✓ Generated mel spectrogram: {mel_spectrogram.shape}")
            print(f"  ✓ Duration: {mel_len * 256 / 22050:.2f} seconds")
        except Exception as e:
            print(f"  ⚠️  Error during generation: {e}")
            import traceback
            traceback.print_exc()
            # Create a minimal mel spectrogram as fallback
            mel_spectrogram = np.zeros((spec_channels, 100))
            print(f"  Using fallback mel spectrogram")
    
    print()
    
    # Convert mel spectrogram to audio
    print("Converting to audio...")
    try:
        audio = mel_spectrogram_to_audio(
            mel_spectrogram,
            sr=22050,
            n_fft=1024,
            hop_length=256,
            n_iter=32
        )
        print(f"  ✓ Audio generated: {len(audio)} samples ({len(audio)/22050:.2f} seconds)")
    except Exception as e:
        print(f"  ⚠️  Error during audio conversion: {e}")
        # Generate a short beep as fallback
        duration = 1.0
        t = np.linspace(0, duration, int(duration * 22050))
        audio = 0.3 * np.sin(2 * np.pi * 440 * t)
        audio = audio.astype(np.float32)
        print(f"  Using fallback audio")
    
    print()
    
    # Save audio
    print("Saving audio...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_wav(output_path, audio, sr=22050)
    print(f"  ✓ Saved: {output_path}")
    
    print()
    print(f"✅ Synthesis complete!")
    
    # Print file info
    if Path(output_path).exists():
        file_size = Path(output_path).stat().st_size / 1024
        print(f"✓ File size: {file_size:.1f} KB")

def main():
    parser = argparse.ArgumentParser(description='Synthesize speech using VITS')
    parser.add_argument('--text', required=True, help='Persian text to synthesize')
    parser.add_argument('--model_path', required=True, help='Path to trained model checkpoint')
    parser.add_argument('--output_path', required=True, help='Output audio file path (.wav)')
    parser.add_argument('--length_scale', type=float, default=1.0, 
                       help='Duration scale factor (1.0=normal, >1.0=slower, <1.0=faster)')
    
    args = parser.parse_args()
    synthesize(args.text, args.model_path, args.output_path, args.length_scale)

if __name__ == "__main__":
    main()
