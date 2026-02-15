#!/usr/bin/env python3
"""Fine-tune VITS on voice sample for speaker adaptation"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import json
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from training.vits_model import VITS
from utils.text_utils import get_symbol_list, clean_text, text_to_sequence, get_symbol_to_id
from utils.audio_utils import load_wav, extract_mel_spectrogram, trim_silence, normalize_volume


def load_voice_sample(audio_path, sample_rate=22050, n_mels=80):
    """Load and process voice sample"""
    try:
        # Load audio
        audio = load_wav(audio_path, sr=sample_rate)
        
        # Trim silence
        audio = trim_silence(audio, top_db=40)
        
        # Normalize
        audio = normalize_volume(audio, target_level=-20.0)
        
        # Extract mel spectrogram
        mel = extract_mel_spectrogram(
            audio,
            sr=sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=n_mels
        )
        
        return torch.FloatTensor(mel)
    except Exception as e:
        print(f"  ⚠️  Error loading voice sample: {e}")
        return None


def finetune_voice(base_model, voice_sample, transcript, speaker_name, output_path, 
                   iterations, learning_rate):
    """Fine-tune VITS on voice sample for speaker adaptation"""
    print(f"🎤 Fine-tuning Voice Model")
    print(f"=" * 50)
    print(f"Base model: {base_model}")
    print(f"Voice sample: {voice_sample}")
    print(f"Transcript: {transcript}")
    print(f"Speaker: {speaker_name}")
    print(f"Iterations: {iterations}")
    print(f"Learning rate: {learning_rate}")
    print()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Model configuration
    n_vocab = len(get_symbol_list())
    spec_channels = 80
    hidden_channels = 192
    filter_channels = 768
    n_heads = 2
    n_layers = 6
    
    # Load base model
    print("Loading base model...")
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
    
    if Path(base_model).exists():
        print(f"  Loading: {base_model}")
        checkpoint = torch.load(base_model, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"  ✓ Loaded checkpoint from step {checkpoint.get('step', 'unknown')}")
        else:
            model.load_state_dict(checkpoint)
        print("  ✓ Base model loaded")
    else:
        print("  ⚠️  Base model not found, starting from scratch")
    print()
    
    # Freeze text encoder for fine-tuning (transfer learning)
    print("Freezing text encoder...")
    for param in model.text_encoder.parameters():
        param.requires_grad = False
    print("  ✓ Text encoder frozen (preserving phonetic knowledge)")
    print()
    
    # Setup optimizer (only for unfrozen parameters)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(trainable_params, lr=learning_rate, betas=(0.8, 0.99), eps=1e-9)
    
    # Load voice sample
    print("Processing voice sample...")
    if not Path(voice_sample).exists():
        print(f"  ❌ Voice sample not found: {voice_sample}")
        return
    
    mel_target = load_voice_sample(voice_sample, sample_rate=22050, n_mels=spec_channels)
    if mel_target is None:
        print(f"  ❌ Failed to load voice sample")
        return
    
    mel_target = mel_target.to(device)
    mel_length = torch.LongTensor([mel_target.shape[1]]).to(device)
    print(f"  ✓ Voice features extracted: {mel_target.shape}")
    print()
    
    # Process transcript
    print("Processing transcript...")
    if not transcript:
        print(f"  ⚠️  No transcript provided, using generic text")
        transcript = "این یک نمونه متن فارسی است"
    
    text_clean = clean_text(transcript)
    symbol_to_id = get_symbol_to_id()
    text_seq = text_to_sequence(text_clean, symbol_to_id)
    text_tensor = torch.LongTensor(text_seq).unsqueeze(0).to(device)
    text_length = torch.LongTensor([len(text_seq)]).to(device)
    
    print(f"  Original: {transcript}")
    print(f"  Cleaned: {text_clean}")
    print(f"  Sequence length: {len(text_seq)}")
    print()
    
    # Fine-tuning loop
    print("Starting fine-tuning...")
    print()
    
    model.train()
    best_loss = float('inf')
    losses = []
    
    mel_target_expanded = mel_target.unsqueeze(0)  # [1, n_mels, time]
    
    for step in tqdm(range(iterations), desc="Fine-tuning"):
        try:
            # Forward pass
            outputs = model(text_tensor, text_length, mel_target_expanded, mel_length)
            
            # Calculate loss (weighted combination)
            loss_kl = outputs['loss_kl']
            loss_dur = outputs['loss_dur']
            loss = loss_kl + loss_dur
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            # Track losses
            loss_val = loss.item()
            losses.append(loss_val)
            
            if loss_val < best_loss:
                best_loss = loss_val
            
            # Log progress
            if (step + 1) % 100 == 0:
                avg_loss = sum(losses[-100:]) / min(100, len(losses))
                tqdm.write(f"Step {step+1}/{iterations} - Loss: {loss_val:.4f} - "
                          f"Avg: {avg_loss:.4f} - Best: {best_loss:.4f}")
        
        except Exception as e:
            print(f"\n⚠️  Error at step {step}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print()
    
    # Save fine-tuned model
    print("Saving fine-tuned model...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'speaker_name': speaker_name,
        'voice_sample': str(voice_sample),
        'transcript': transcript,
        'best_loss': best_loss,
        'iterations': iterations,
    }, output_path)
    
    print(f"  ✓ Model saved: {output_path}")
    
    # Save metadata
    metadata = {
        'speaker_name': speaker_name,
        'voice_sample': str(voice_sample),
        'transcript': transcript,
        'iterations': iterations,
        'final_loss': losses[-1] if losses else None,
        'best_loss': best_loss,
        'avg_loss_last_100': sum(losses[-100:]) / min(100, len(losses)) if losses else None,
    }
    
    metadata_path = Path(output_path).with_suffix('.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ Metadata saved: {metadata_path}")
    print()
    print(f"✅ Fine-tuning complete!")
    print(f"✓ Best loss: {best_loss:.4f}")
    if losses:
        print(f"✓ Final loss: {losses[-1]:.4f}")
        print(f"✓ Avg loss (last 100): {sum(losses[-100:]) / min(100, len(losses)):.4f}")

def main():
    parser = argparse.ArgumentParser(description='Fine-tune VITS for speaker adaptation')
    parser.add_argument('--base_model', required=True, help='Path to base trained model')
    parser.add_argument('--voice_sample', required=True, help='Path to voice sample audio (.wav)')
    parser.add_argument('--transcript', default='', help='Transcript of voice sample')
    parser.add_argument('--speaker_name', required=True, help='Name of target speaker')
    parser.add_argument('--output_path', required=True, help='Path to save fine-tuned model')
    parser.add_argument('--iterations', type=int, default=5000, help='Number of fine-tuning iterations')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate for fine-tuning')
    
    args = parser.parse_args()
    finetune_voice(args.base_model, args.voice_sample, args.transcript, args.speaker_name,
                  args.output_path, args.iterations, args.learning_rate)

if __name__ == "__main__":
    main()
