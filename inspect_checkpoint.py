#!/usr/bin/env python3
"""Inspect checkpoint structure"""

import torch
from pathlib import Path

checkpoint_path = "checkpoints/vits_persian_final.pth"

print(f"Loading checkpoint: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print(f"\nCheckpoint type: {type(checkpoint)}")
print(f"\nTop-level keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'tensor/model'}")

if isinstance(checkpoint, dict):
    if 'model_state_dict' in checkpoint:
        print(f"\nModel state dict keys (first 20):")
        keys = list(checkpoint['model_state_dict'].keys())
        for key in keys[:20]:
            print(f"  {key}")
        if len(keys) > 20:
            print(f"  ... and {len(keys) - 20} more")
            
        print(f"\nOther checkpoint info:")
        for key in checkpoint.keys():
            if key != 'model_state_dict':
                print(f"  {key}: {checkpoint[key]}")
    else:
        print(f"\nState dict keys (first 20):")
        keys = list(checkpoint.keys())
        for key in keys[:20]:
            print(f"  {key}")
        if len(keys) > 20:
            print(f"  ... and {len(keys) - 20} more")
