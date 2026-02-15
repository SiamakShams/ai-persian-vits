#!/usr/bin/env python3
"""Test if the processed data can be loaded properly"""

from pathlib import Path
from training.dataset import get_data_loaders

def test_data_loading():
    """Test loading the preprocessed data"""
    print("🧪 Testing Data Loading")
    print("=" * 60)
    print()
    
    data_path = Path("datasets/processed")
    train_file = data_path / "train.txt"
    val_file = data_path / "val.txt"
    
    if not train_file.exists():
        print(f"❌ Training file not found: {train_file}")
        return False
    
    if not val_file.exists():
        print(f"❌ Validation file not found: {val_file}")
        return False
    
    print(f"✓ Found train file: {train_file}")
    print(f"✓ Found val file: {val_file}")
    print()
    
    try:
        print("Loading datasets...")
        train_loader, val_loader = get_data_loaders(
            train_file, val_file,
            batch_size=4,
            num_workers=0,
            sample_rate=22050,
            n_mels=80
        )
        
        print(f"✓ Train batches: {len(train_loader)}")
        print(f"✓ Val batches: {len(val_loader)}")
        print()
        
        # Try to load one batch
        print("Testing batch loading...")
        for batch in train_loader:
            print(f"✓ Successfully loaded batch with {len(batch)} items")
            print(f"  Batch keys: {list(batch.keys()) if isinstance(batch, dict) else 'tensor batch'}")
            break
        
        print()
        print("✅ Data loading test PASSED!")
        print()
        return True
        
    except Exception as e:
        print(f"❌ Data loading test FAILED!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    success = test_data_loading()
    sys.exit(0 if success else 1)
