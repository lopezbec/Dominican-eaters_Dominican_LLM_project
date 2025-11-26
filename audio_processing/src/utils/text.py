"""Text processing utilities for Dominican content."""

import re
from typing import Optional
import pandas as pd


def clean_lyrics(text: str) -> str:
    """
    Clean lyrics text from Genius annotations.
    
    Args:
        text: Raw lyrics text from Genius
        
    Returns:
        Cleaned lyrics text
    """
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    text = re.sub(r'\d+\s+Contributors?.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Translations?.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(English|Español|Deutsch|Français|Italiano|Português).*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'Embed\s*$', '', text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r'^.*Lyrics\s*$', '', text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if len(line) > 0 and not line.isdigit():
            cleaned_lines.append(line)
    
    text = '\n'.join(cleaned_lines)
    
    return text.strip()


def normalize_text(text: str, preserve_newlines: bool = False) -> str:
    """
    Normalize text for comparison.
    
    Args:
        text: Input text
        preserve_newlines: If True, preserve line breaks during normalization
        
    Returns:
        Normalized text (lowercase, no punctuation, normalized whitespace)
    """
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    
    if preserve_newlines:
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
    else:
        text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def clean_reference_text(text: str) -> str:
    """
    Clean reference text by removing formatting artifacts.
    
    Args:
        text: Raw reference text
        
    Returns:
        Cleaned text with normalized whitespace
    """
    text = clean_lyrics(text)
    text = text.replace('\r\n', ' ')
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    text = text.replace('\t', ' ')
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def sanitize_filename(name: str, max_length: int = 100) -> str:
    """
    Sanitize a string for use as filename.
    
    Args:
        name: Input string
        max_length: Maximum filename length
        
    Returns:
        Sanitized filename
    """
    name = re.sub(r'[^\w\s-]', '', str(name))
    name = re.sub(r'[-\s]+', '_', name)
    return name[:max_length]
