#!/usr/bin/env python3
"""Train VITS on Persian corpus"""

import argparse
import os
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from training.vits_model import VITS
from training.dataset import get_data_loaders
from utils.text_utils import get_symbol_list

def train_vits(data_path, checkpoint_path, iterations, batch_size, learning_rate, resume=False):
    """Train VITS on Persian corpus"""
    print(f"🎤 Training VITS Model")
    print(f"=" * 50)
    print(f"Data path: {data_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Iterations: {iterations}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Resume: {resume}")
    print()
    
    # Create checkpoint directory
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    print()
    
    # Model configuration
    n_vocab = len(get_symbol_list())
    spec_channels = 80
    hidden_channels = 192
    filter_channels = 768
    n_heads = 2
    n_layers = 6
    
    print(f"Model configuration:")
    print(f"  Vocabulary size: {n_vocab}")
    print(f"  Mel channels: {spec_channels}")
    print(f"  Hidden channels: {hidden_channels}")
    print(f"  Filter channels: {filter_channels}")
    print(f"  Attention heads: {n_heads}")
    print(f"  Encoder layers: {n_layers}")
    print()
    
    # Create model
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
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    print()
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.8, 0.99), eps=1e-9)
    
    # Resume from checkpoint if requested
    start_iteration = 0
    if resume:
        checkpoints = sorted(Path(checkpoint_path).glob('checkpoint_*.pth'))
        if checkpoints:
            latest_checkpoint = checkpoints[-1]
            print(f"📂 Loading checkpoint: {latest_checkpoint}")
            checkpoint = torch.load(latest_checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_iteration = checkpoint['step']
            print(f"✓ Resumed from iteration {start_iteration}")
            print()
        else:
            print(f"⚠️  No checkpoints found, starting from scratch")
            print()
    
    # Create dataset and dataloader
    data_path = Path(data_path)
    train_file = data_path / "train.txt"
    val_file = data_path / "val.txt"
    
    if not train_file.exists():
        print(f"❌ Training file not found: {train_file}")
        print(f"Run preprocessing first: python preprocessing/preprocess_datasets.py")
        return
    
    print("Loading datasets...")
    train_loader, val_loader = get_data_loaders(
        train_file, val_file,
        batch_size=batch_size,
        num_workers=0,  # Set to 0 for Windows compatibility
        sample_rate=22050,
        n_mels=spec_channels
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print()
    
    # Tensorboard
    log_dir = Path(checkpoint_path).parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    print(f"TensorBoard logs: {log_dir}")
    print(f"  View with: tensorboard --logdir={log_dir}")
    print()
    
    # Training loop
    model.train()
    global_step = start_iteration
    epoch = 0
    
    print(f"Starting training from step {global_step}...")
    print()
    
    while global_step < iterations:
        epoch += 1
        epoch_losses = {
            'total': 0.0,
            'kl': 0.0,
            'dur': 0.0
        }
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            if global_step >= iterations:
                break
            
            # Move batch to device
            text = batch['text'].to(device)
            text_lengths = batch['text_lengths'].to(device)
            mel = batch['mel'].to(device)
            mel_lengths = batch['mel_lengths'].to(device)
            
            # Forward pass
            try:
                outputs = model(text, text_lengths, mel, mel_lengths)
                
                # Calculate losses
                loss_kl = outputs['loss_kl']
                loss_dur = outputs['loss_dur']
                loss_total = loss_kl + loss_dur
                
                # Backward pass
                optimizer.zero_grad()
                loss_total.backward()
                
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                
                optimizer.step()
                
                # Track losses
                epoch_losses['total'] += loss_total.item()
                epoch_losses['kl'] += loss_kl.item()
                epoch_losses['dur'] += loss_dur.item()
                
                global_step += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss_total.item():.4f}',
                    'kl': f'{loss_kl.item():.4f}',
                    'dur': f'{loss_dur.item():.4f}',
                    'step': global_step
                })
                
                # Log to tensorboard
                if global_step % 10 == 0:
                    writer.add_scalar('train/loss_total', loss_total.item(), global_step)
                    writer.add_scalar('train/loss_kl', loss_kl.item(), global_step)
                    writer.add_scalar('train/loss_dur', loss_dur.item(), global_step)
                    writer.add_scalar('train/grad_norm', grad_norm.item(), global_step)
                
                # Save checkpoint periodically
                if global_step % 1000 == 0:
                    checkpoint_file = Path(checkpoint_path) / f"checkpoint_{global_step}.pth"
                    torch.save({
                        'step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss_total.item(),
                    }, checkpoint_file)
                    print(f"\n✓ Saved checkpoint: {checkpoint_file}")
                
                # Validation
                if global_step % 5000 == 0:
                    print(f"\n📊 Running validation...")
                    val_loss = validate(model, val_loader, device)
                    writer.add_scalar('val/loss_total', val_loss, global_step)
                    print(f"Validation loss: {val_loss:.4f}\n")
                    model.train()
                
            except Exception as e:
                print(f"\n⚠️  Error in training step {global_step}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Epoch summary
        if len(train_loader) > 0:
            avg_loss = epoch_losses['total'] / len(train_loader)
            avg_kl = epoch_losses['kl'] / len(train_loader)
            avg_dur = epoch_losses['dur'] / len(train_loader)
            print(f"\nEpoch {epoch} - Avg Loss: {avg_loss:.4f} (KL: {avg_kl:.4f}, Dur: {avg_dur:.4f})")
    
    # Save final model
    final_model_path = Path(checkpoint_path) / "vits_persian_final.pth"
    torch.save({
        'step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_model_path)
    
    print()
    print(f"✅ Training complete!")
    print(f"Final model saved to: {final_model_path}")
    print(f"Total steps: {global_step}")
    
    writer.close()

def validate(model, val_loader, device):
    """Run validation"""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            try:
                text = batch['text'].to(device)
                text_lengths = batch['text_lengths'].to(device)
                mel = batch['mel'].to(device)
                mel_lengths = batch['mel_lengths'].to(device)
                
                outputs = model(text, text_lengths, mel, mel_lengths)
                loss = outputs['loss_kl'] + outputs['loss_dur']
                
                total_loss += loss.item()
                n_batches += 1
            except Exception as e:
                print(f"  Warning: Validation batch failed: {e}")
                continue
    
    return total_loss / max(n_batches, 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--checkpoint_path', required=True)
    parser.add_argument('--iterations', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint')
    
    args = parser.parse_args()
    train_vits(args.data_path, args.checkpoint_path, args.iterations, 
              args.batch_size, args.learning_rate, args.resume)

if __name__ == "__main__":
    main()
