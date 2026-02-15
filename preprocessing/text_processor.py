#!/usr/bin/env python3
"""Text processing utilities for Persian language"""

import sys
from pathlib import Path

# Import from utils
sys.path.append(str(Path(__file__).parent.parent))
from utils.text_utils import (
    normalize_persian_text,
    clean_text,
    text_to_sequence,
    sequence_to_text,
    expand_abbreviations,
    normalize_numbers,
    phonemize,
    get_symbol_list,
    get_symbol_to_id,
    get_id_to_symbol
)

# Re-export for convenience
__all__ = [
    'normalize_persian_text',
    'clean_text',
    'text_to_sequence',
    'sequence_to_text',
    'expand_abbreviations',
    'normalize_numbers',
    'phonemize',
    'get_symbol_list',
    'get_symbol_to_id',
    'get_id_to_symbol'
]

if __name__ == "__main__":
    print("Text processor ready")
    print("Available functions:")
    for func in __all__:
        print(f"  - {func}")
    
    # Test
    sample_text = "سلام دنیا"
    print(f"\nTest: {sample_text}")
    cleaned = clean_text(sample_text)
    print(f"Cleaned: {cleaned}")
    seq = text_to_sequence(cleaned)
    print(f"Sequence: {seq}")
