#!/usr/bin/env python3
"""Preprocess Persian TTS datasets from Parquet format"""

import argparse
from pathlib import Path
import json
import numpy as np
import soundfile as sf
from tqdm import tqdm

def process_parquet_dataset(dataset_path, dataset_name, output_dir, output_samples):
    """Process a parquet-based dataset"""
    try:
        import pandas as pd
    except ImportError:
        print(f"  ❌ pandas required for parquet support. Install: pip install pandas pyarrow")
        return 0
    
    if not dataset_path or not Path(dataset_path).exists():
        print(f"  ⚠ {dataset_name} not found, skipping")
        return 0
    
    print(f"  📂 Processing {dataset_name}...")
    dataset_path = Path(dataset_path)
    
    # Find all parquet files
    parquet_files = sorted(list(dataset_path.glob('*/dataset_part_*.parquet'))) + \
                    sorted(list(dataset_path.glob('dataset_part_*.parquet')))
    
    if not parquet_files:
        print(f"    ⚠ No parquet files found in {dataset_path}")
        return 0
    
    print(f"    Found {len(parquet_files)} parquet files")
    
    samples_count = 0
    audio_output_dir = output_dir / dataset_name / "audio"
    audio_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each parquet file
    for parquet_file in parquet_files:
        try:
            df = pd.read_parquet(parquet_file)
            
            # Detect columns (different datasets use different names)
            audio_col = 'audio' if 'audio' in df.columns else None
            text_col = None
            for col in df.columns:
                if col.lower() in ['transcript', 'text', 'sentence']:
                    text_col = col
                    break
            
            if not audio_col or not text_col:
                print(f"    ⚠ Skipping {parquet_file.name}: missing audio or text columns")
                continue
            
            sr_col = 'samplerate' if 'samplerate' in df.columns else 'sample_rate'
            
            # Process each row
            for idx, row in df.iterrows():
                try:
                    audio = row[audio_col]
                    text = row[text_col]
                    sr = int(row[sr_col]) if sr_col in df.columns else 22050
                    
                    # Validate
                    if not isinstance(audio, np.ndarray) or len(text.strip()) == 0:
                        continue
                    
                    # Save audio as WAV
                    audio_file = audio_output_dir / f"{dataset_name}_{samples_count:05d}.wav"
                    sf.write(str(audio_file), audio, sr)
                    
                    # Create sample entry
                    output_samples.append({
                        'audio': str(audio_file.relative_to(output_dir.parent)),
                        'text': text.strip(),
                        'dataset': dataset_name,
                        'samplerate': sr
                    })
                    
                    samples_count += 1
                    
                except Exception as e:
                    continue
            
            print(f"    ✓ Processed {parquet_file.name}: {samples_count} samples")
            
        except Exception as e:
            print(f"    ⚠ Error processing {parquet_file.name}: {e}")
            continue
    
    print(f"    Total from {dataset_name}: {samples_count} samples")
    return samples_count

def preprocess_datasets(dpt_path=None, mana_path=None, parisgoo_path=None, 
                       quran_path=None, output_dir=None, input_dir=None, generate_dummy=False):
    """Preprocess all datasets"""
    print("🎤 Preprocessing Persian Datasets (Parquet Format)")
    print("=" * 60)
    print()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_samples = []
    total_count = 0
    
    # If input_dir provided, auto-discover datasets
    if input_dir:
        input_dir = Path(input_dir)
        datasets = []
        dataset_paths = {
            'GPTInformal-Persian': input_dir / 'GPTInformal-Persian',
            'Mana-TTS': input_dir / 'Mana-TTS',
            'Quran-Persian': input_dir / 'Quran-Persian'
        }
        
        for name, path in dataset_paths.items():
            if path.exists():
                datasets.append((path, name))
    else:
        # Process each dataset with individual paths (legacy)
        datasets = [
            (dpt_path, "DPT-InformalPersian"),
            (mana_path, "Mana-TTS"),
            (parisgoo_path, "ParisGoo"),
            (quran_path, "Quran-Persian")
        ]
    
    for path, name in datasets:
        if path:
            count = process_parquet_dataset(path, name, output_dir, all_samples)
            total_count += count
    
    # If no datasets found or generate_dummy flag, create dummy data
    if total_count == 0 or generate_dummy:
        print("\n  ⚠ Creating dummy data for testing...")
        for i in range(100):
            all_samples.append({
                'audio': f'dummy_audio_{i}.wav',
                'text': f'این یک نمونه متن فارسی است شماره {i}',
                'dataset': 'dummy'
            })
        total_count = 100
    
    print()
    print(f"📊 Total samples collected: {total_count}")
    print()
    
    # Split into train/val (90/10)
    split_idx = int(len(all_samples) * 0.9)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]
    
    print(f"  Training samples: {len(train_samples)}")
    print(f"  Validation samples: {len(val_samples)}")
    print()
    
    # Save metadata files
    print("💾 Saving metadata...")
    
    # Save train metadata
    train_file = output_dir / "train.txt"
    with open(train_file, 'w', encoding='utf-8') as f:
        for sample in train_samples:
            f.write(f"{sample['audio']}|{sample['text']}\n")
    print(f"  ✓ Saved: {train_file}")
    
    # Save validation metadata
    val_file = output_dir / "val.txt"
    with open(val_file, 'w', encoding='utf-8') as f:
        for sample in val_samples:
            f.write(f"{sample['audio']}|{sample['text']}\n")
    print(f"  ✓ Saved: {val_file}")
    
    # Save metadata.txt (for compatibility)
    metadata_file = output_dir / "metadata.txt"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            f.write(f"{sample['audio']}|{sample['text']}\n")
    print(f"  ✓ Saved: {metadata_file}")
    
    # Save JSON summary
    summary = {
        'total_samples': total_count,
        'train_samples': len(train_samples),
        'val_samples': len(val_samples),
        'datasets_processed': [name for path, name in datasets if path]
    }
    
    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Saved: {summary_file}")
    
    print()
    print("✅ Preprocessing complete!")
    print(f"Output directory: {output_dir}")
    print()
    print("Next step:")
    print(f"  Run training: python training/train_vits.py --data_path {output_dir} --checkpoint_path checkpoints")

def main():
    parser = argparse.ArgumentParser(description='Preprocess Persian TTS datasets from Parquet format')
    parser.add_argument('--input_dir', default=None, help='Input directory containing datasets (auto-discovers)', type=str)
    parser.add_argument('--output_dir', required=True, help='Output directory for processed data')
    parser.add_argument('--generate_dummy', action='store_true', help='Generate dummy data for testing')
    parser.add_argument('--dpt_path', default=None, help='Path to DPT-InformalPersian dataset (legacy)')
    parser.add_argument('--mana_path', default=None, help='Path to Mana-TTS dataset (legacy)')
    parser.add_argument('--parisgoo_path', default=None, help='Path to ParisGoo dataset (legacy)')
    parser.add_argument('--quran_path', default=None, help='Path to QuranPersian dataset (legacy)')
    
    args = parser.parse_args()
    preprocess_datasets(args.dpt_path, args.mana_path, args.parisgoo_path, 
                       args.quran_path, args.output_dir, args.input_dir, args.generate_dummy)

if __name__ == "__main__":
    main()
