#!/usr/bin/env python3
"""Text utility functions for Persian TTS"""

import re
import unicodedata

# Persian alphabet and common symbols
PERSIAN_CHARS = 'آابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی'
PERSIAN_DIACRITICS = '\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652'  # Tanwin, Fatha, Damma, Kasra, Shadda, Sukun
VALID_SYMBOLS = list(PERSIAN_CHARS) + [' ', '.', '،', '؛', '؟', '!', '-', ':', '(', ')']

# Persian phoneme-like character set (simplified)
PERSIAN_PHONEMES = list('آابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی .')

# Arabic to Persian character mapping
ARABIC_TO_PERSIAN = {
    'ك': 'ک',
    'ي': 'ی',
    'ى': 'ی',
    'ة': 'ه',
    '٠': '۰',
    '١': '۱',
    '٢': '۲',
    '٣': '۳',
    '٤': '۴',
    '٥': '۵',
    '٦': '۶',
    '٧': '۷',
    '٨': '۸',
    '٩': '۹',
}

# English digits to Persian
ENGLISH_TO_PERSIAN_DIGITS = {
    '0': '۰',
    '1': '۱',
    '2': '۲',
    '3': '۳',
    '4': '۴',
    '5': '۵',
    '6': '۶',
    '7': '۷',
    '8': '۸',
    '9': '۹',
}

# Persian number words (simplified)
PERSIAN_NUMBERS = {
    '0': 'صفر',
    '1': 'یک',
    '2': 'دو',
    '3': 'سه',
    '4': 'چهار',
    '5': 'پنج',
    '6': 'شش',
    '7': 'هفت',
    '8': 'هشت',
    '9': 'نه',
    '10': 'ده',
}

def normalize_persian_text(text):
    """
    Normalize Persian text
    
    Args:
        text: Input Persian text
        
    Returns:
        normalized_text: Normalized text
    """
    # Convert to NFKC normalization form
    text = unicodedata.normalize('NFKC', text)
    
    # Replace Arabic characters with Persian equivalents
    for arabic, persian in ARABIC_TO_PERSIAN.items():
        text = text.replace(arabic, persian)
    
    # Remove diacritics (optional for TTS)
    for diacritic in PERSIAN_DIACRITICS:
        text = text.replace(diacritic, '')
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text

def clean_text(text):
    """
    Clean text for TTS processing
    
    Args:
        text: Input text
        
    Returns:
        cleaned_text: Cleaned text
    """
    # Normalize Persian text
    text = normalize_persian_text(text)
    
    # Remove control characters
    text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C')
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Ensure proper spacing around punctuation
    text = re.sub(r'\s+([.,،؛؟!])', r'\1', text)
    text = re.sub(r'([.,،؛؟!])(?=[^\s])', r'\1 ', text)
    
    return text

def text_to_sequence(text, symbol_to_id=None):
    """
    Convert text to sequence of symbol IDs
    
    Args:
        text: Input text
        symbol_to_id: Dictionary mapping symbols to IDs. If None, uses default
        
    Returns:
        sequence: List of symbol IDs
    """
    # Clean text first
    text = clean_text(text)
    
    # Create default symbol mapping if not provided
    if symbol_to_id is None:
        symbol_to_id = {s: i for i, s in enumerate(VALID_SYMBOLS)}
        symbol_to_id['<pad>'] = len(symbol_to_id)
        symbol_to_id['<unk>'] = len(symbol_to_id)
    
    # Convert text to sequence
    sequence = []
    for char in text:
        if char in symbol_to_id:
            sequence.append(symbol_to_id[char])
        else:
            # Use unknown token for unseen characters
            sequence.append(symbol_to_id.get('<unk>', 0))
    
    return sequence

def sequence_to_text(sequence, id_to_symbol=None):
    """
    Convert sequence of symbol IDs back to text
    
    Args:
        sequence: List of symbol IDs
        id_to_symbol: Dictionary mapping IDs to symbols. If None, uses default
        
    Returns:
        text: Output text
    """
    # Create default symbol mapping if not provided
    if id_to_symbol is None:
        symbols = VALID_SYMBOLS + ['<pad>', '<unk>']
        id_to_symbol = {i: s for i, s in enumerate(symbols)}
    
    # Convert sequence to text
    chars = [id_to_symbol.get(idx, '') for idx in sequence]
    text = ''.join(chars)
    
    return text

def expand_abbreviations(text):
    """
    Expand common Persian abbreviations
    
    Args:
        text: Input text
        
    Returns:
        expanded_text: Text with expanded abbreviations
    """
    abbreviations = {
        'ص': 'صفحه',
        'ج': 'جلد',
        'م': 'متر',
        'کم': 'کیلومتر',
        'کگ': 'کیلوگرم',
    }
    
    for abbr, full in abbreviations.items():
        text = re.sub(r'\b' + abbr + r'\b', full, text)
    
    return text

def normalize_numbers(text):
    """
    Convert numbers to Persian words
    
    Args:
        text: Input text with numbers
        
    Returns:
        text_with_words: Text with numbers as words
    """
    def replace_number(match):
        num_str = match.group(0)
        
        # Handle single digits
        if len(num_str) == 1 and num_str in PERSIAN_NUMBERS:
            return PERSIAN_NUMBERS[num_str]
        
        # For multi-digit numbers, convert each digit separately
        # (Simplified - proper implementation would handle teens, hundreds, etc.)
        words = []
        for digit in num_str:
            if digit in PERSIAN_NUMBERS:
                words.append(PERSIAN_NUMBERS[digit])
        
        return ' '.join(words) if words else num_str
    
    # Replace English digits
    text = re.sub(r'\d+', replace_number, text)
    
    # Replace Persian digits
    for persian_digit, word in PERSIAN_NUMBERS.items():
        persian_char = ENGLISH_TO_PERSIAN_DIGITS.get(persian_digit, persian_digit)
        # Simple single-digit replacement
        text = text.replace(persian_char + ' ', word + ' ')
    
    return text

def phonemize(text):
    """
    Convert text to phonemes (simplified for Persian)
    
    Args:
        text: Input Persian text
        
    Returns:
        phonemes: Phoneme sequence (simplified character-based)
    """
    # Clean and normalize text
    text = clean_text(text)
    
    # For Persian, we use a character-based approach as a simplified phonemization
    # A full phonemizer would require linguistic rules or a trained model
    phonemes = []
    for char in text:
        if char in PERSIAN_PHONEMES:
            phonemes.append(char)
        elif char in ' .,!?؟،؛':
            phonemes.append(char)
    
    return ''.join(phonemes)

def get_symbol_list():
    """
    Get the list of valid symbols for the model
    
    Returns:
        symbols: List of symbols
    """
    return VALID_SYMBOLS + ['<pad>', '<unk>']

def get_symbol_to_id():
    """
    Get symbol to ID mapping
    
    Returns:
        symbol_to_id: Dictionary mapping symbols to IDs
    """
    symbols = get_symbol_list()
    return {s: i for i, s in enumerate(symbols)}

def get_id_to_symbol():
    """
    Get ID to symbol mapping
    
    Returns:
        id_to_symbol: Dictionary mapping IDs to symbols
    """
    symbols = get_symbol_list()
    return {i: s for i, s in enumerate(symbols)}
