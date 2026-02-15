#!/usr/bin/env python3
"""Rebuild metadata files from existing processed audio and parquet sources"""

import argparse
from pathlib import Path
import json
import pandas as pd
from tqdm import tqdm

def rebuild_metadata(processed_dir, raw_dir):
    """Rebuild metadata from already-processed audio files"""
    print("🔧 Rebuilding Metadata from Existing Audio Files")
    print("=" * 60)
    print()
    
    processed_dir = Path(processed_dir)
    raw_dir = Path(raw_dir)
    
    all_samples = []
    
    # Process each dataset
    datasets = ['GPTInformal-Persian', 'Mana-TTS', 'Quran-Persian']
    
    for dataset_name in datasets:
        audio_dir = processed_dir / dataset_name / "audio"
        if not audio_dir.exists():
            print(f"  ⚠ {dataset_name} audio directory not found, skipping")
            continue
        
        print(f"  📂 Processing {dataset_name}...")
        
        # Get all audio files
        audio_files = sorted(list(audio_dir.glob(f"{dataset_name}_*.wav")))
        if not audio_files:
            print(f"    ⚠ No audio files found in {audio_dir}")
            continue
        
        print(f"    Found {len(audio_files)} audio files")
        
        # Load corresponding parquet files to get transcripts
        raw_dataset_path = raw_dir / dataset_name / "dataset"
        if not raw_dataset_path.exists():
            raw_dataset_path = raw_dir / dataset_name
        
        parquet_files = sorted(list(raw_dataset_path.glob('dataset_part_*.parquet')))
        if not parquet_files:
            print(f"    ⚠ No parquet files found in {raw_dataset_path}")
            continue
        
        # Load all transcripts into a list
        transcripts = []
        print(f"    Loading transcripts from {len(parquet_files)} parquet files...")
        
        for idx, parquet_file in enumerate(parquet_files):
            try:
                # Only read the text column to save memory and time
                df = pd.read_parquet(parquet_file, columns=None)
                
                # Detect text column
                text_col = None
                for col in df.columns:
                    if col.lower() in ['transcript', 'text', 'sentence']:
                        text_col = col
                        break
                
                if text_col:
                    for text in df[text_col]:
                        if isinstance(text, str) and len(text.strip()) > 0:
                            transcripts.append(text.strip())
                
                # Print progress every 10 files for large datasets
                if (idx + 1) % 10 == 0:
                    print(f"      Progress: {idx + 1}/{len(parquet_files)} files ({len(transcripts)} transcripts)")
                    
            except KeyboardInterrupt:
                print(f"    ⚠ Interrupted at file {idx + 1}/{len(parquet_files)}")
                print(f"    ✓ Using {len(transcripts)} transcripts loaded so far")
                break
            except Exception as e:
                print(f"    ⚠ Error reading {parquet_file.name}: {e}")
                continue
        
        print(f"    Loaded {len(transcripts)} transcripts from {min(idx + 1, len(parquet_files))} parquet files")
        
        # Match audio files with transcripts
        matched_count = min(len(audio_files), len(transcripts))
        
        for idx in range(matched_count):
            audio_file = audio_files[idx]
            text = transcripts[idx]
            
            # Create relative path from project root
            rel_path = audio_file.relative_to(processed_dir.parent)
            
            all_samples.append({
                'audio': str(rel_path).replace('\\', '/'),
                'text': text,
                'dataset': dataset_name
            })
        
        if matched_count < len(audio_files):
            print(f"    ⚠ Only matched {matched_count}/{len(audio_files)} audio files (not enough transcripts)")
        else:
            print(f"    ✓ Matched {matched_count} audio files with transcripts")
    
    print()
    print(f"📊 Total samples: {len(all_samples)}")
    print()
    
    # Split into train/val (90/10)
    split_idx = int(len(all_samples) * 0.9)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]
    
    print(f"  Training samples: {len(train_samples)}")
    print(f"  Validation samples: {len(val_samples)}")
    print()
    
    # Save metadata files
    print("💾 Saving metadata files...")
    
    # Save train metadata
    train_file = processed_dir / "train.txt"
    with open(train_file, 'w', encoding='utf-8') as f:
        for sample in train_samples:
            f.write(f"{sample['audio']}|{sample['text']}\n")
    print(f"  ✓ Saved: {train_file}")
    
    # Save validation metadata
    val_file = processed_dir / "val.txt"
    with open(val_file, 'w', encoding='utf-8') as f:
        for sample in val_samples:
            f.write(f"{sample['audio']}|{sample['text']}\n")
    print(f"  ✓ Saved: {val_file}")
    
    # Save metadata.txt (for compatibility)
    metadata_file = processed_dir / "metadata.txt"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            f.write(f"{sample['audio']}|{sample['text']}\n")
    print(f"  ✓ Saved: {metadata_file}")
    
    # Save JSON summary
    summary = {
        'total_samples': len(all_samples),
        'train_samples': len(train_samples),
        'val_samples': len(val_samples),
        'datasets_processed': datasets
    }
    
    summary_file = processed_dir / "summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Saved: {summary_file}")
    
    print()
    print("✅ Metadata rebuild complete!")
    print(f"Output directory: {processed_dir}")
    print()

def main():
    parser = argparse.ArgumentParser(description='Rebuild metadata from existing audio files')
    parser.add_argument('--processed_dir', default='../datasets/processed', 
                       help='Directory with processed audio files')
    parser.add_argument('--raw_dir', default='../datasets/raw', 
                       help='Directory with raw parquet files')
    
    args = parser.parse_args()
    rebuild_metadata(args.processed_dir, args.raw_dir)

if __name__ == "__main__":
    main()
