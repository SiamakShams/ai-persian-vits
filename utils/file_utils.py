#!/usr/bin/env python3
"""File and path utility functions"""

import os
import json
from pathlib import Path

def ensure_dir(directory):
    """
    Create directory if it doesn't exist
    
    Args:
        directory: Path to directory
    """
    Path(directory).mkdir(parents=True, exist_ok=True)

def load_json(path):
    """
    Load JSON file
    
    Args:
        path: Path to JSON file
        
    Returns:
        data: Loaded JSON data
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, path, indent=2):
    """
    Save data to JSON file
    
    Args:
        data: Data to save
        path: Output path
        indent: JSON indentation
    """
    ensure_dir(Path(path).parent)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

def load_filepaths_and_text(filename):
    """
    Load file paths and transcripts from metadata file
    
    Args:
        filename: Path to metadata file
        
    Returns:
        data: List of (filepath, text) tuples
    """
    with open(filename, 'r', encoding='utf-8') as f:
        filepaths_and_text = []
        for line in f:
            parts = line.strip().split('|')
            if len(parts) >= 2:
                filepaths_and_text.append((parts[0], parts[1]))
        return filepaths_and_text

def get_files_by_extension(directory, extension):
    """
    Get all files with given extension in directory
    
    Args:
        directory: Directory to search
        extension: File extension (e.g., '.wav', '.txt')
        
    Returns:
        files: List of file paths
    """
    return list(Path(directory).rglob(f'*{extension}'))

def get_file_size_mb(path):
    """
    Get file size in megabytes
    
    Args:
        path: Path to file
        
    Returns:
        size_mb: File size in MB
    """
    return os.path.getsize(path) / (1024 * 1024)

def ensure_checkpoint_dir(checkpoint_path):
    """
    Ensure checkpoint directory exists and return latest checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint directory
        
    Returns:
        latest_checkpoint: Path to latest checkpoint or None
    """
    ensure_dir(checkpoint_path)
    checkpoints = sorted(Path(checkpoint_path).glob('*.pth'))
    return str(checkpoints[-1]) if checkpoints else None
