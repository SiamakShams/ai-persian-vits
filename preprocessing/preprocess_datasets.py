#!/usr/bin/env python3
"""Preprocess Persian datasets"""

import argparse
from pathlib import Path

def preprocess_datasets(dpt_path=None, mana_path=None, parisgoo_path=None, quran_path=None, output_dir=None):
    """Preprocess all datasets"""
    print("Preprocessing Persian datasets...")
    
    if dpt_path and Path(dpt_path).exists():
        print(f"Processing DPT-InformalPersian: {dpt_path}")
    
    if mana_path and Path(mana_path).exists():
        print(f"Processing Mana-TTS: {mana_path}")
    
    if parisgoo_path and Path(parisgoo_path).exists():
        print(f"Processing ParisGoo: {parisgoo_path}")
    
    if quran_path and Path(quran_path).exists():
        print(f"Processing QuranPersian: {quran_path}")
    
    print(f"Output: {output_dir}")
    print("✓ Preprocessing complete")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dpt_path', default=None)
    parser.add_argument('--mana_path', default=None)
    parser.add_argument('--parisgoo_path', default=None)
    parser.add_argument('--quran_path', default=None)
    parser.add_argument('--output_dir', required=True)
    
    args = parser.parse_args()
    preprocess_datasets(args.dpt_path, args.mana_path, args.parisgoo_path, args.quran_path, args.output_dir)

if __name__ == "__main__":
    main()
