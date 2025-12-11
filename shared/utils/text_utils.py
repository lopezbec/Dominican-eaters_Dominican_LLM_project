"""Text normalization utilities for consistent string comparison."""

import unicodedata
import re
import pandas as pd
from typing import List, Optional


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison by removing accents and special characters.
    
    Single Responsibility: Text normalization only.
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized lowercase text with accents removed
    """
    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )
    text = ' '.join(text.lower().split())
    return text


def extract_significant_words(text: str, min_length: int = 2) -> List[str]:
    """
    Extract significant words from text, filtering common stop words.
    
    Single Responsibility: Extract meaningful keywords.
    
    Args:
        text: Text to analyze
        min_length: Minimum word length to consider
        
    Returns:
        List of significant words
    """
    common_words = {
        'el', 'la', 'los', 'las', 'un', 'una', 'de', 'del', 'y', 'o', 
        'en', 'a', 'para', 'con', 'por', 'su', 'al', 'lo', 'que'
    }
    
    normalized = normalize_text(text)
    words = [
        word for word in normalized.split() 
        if word not in common_words and len(word) > min_length
    ]
    return words


def extract_last_name(full_name: str) -> str:
    """
    Extract last name from full name.
    
    Single Responsibility: Parse author names.
    
    Args:
        full_name: Full author name
        
    Returns:
        Last name (most distinctive part)
    """
    normalized = normalize_text(full_name)
    parts = normalized.split()
    return parts[-1] if parts else normalized


def calculate_text_similarity(text1: str, text2: str, threshold: float = 0.5) -> bool:
    """
    Calculate if two texts are similar based on shared significant words.
    
    Single Responsibility: Text similarity comparison.
    
    Args:
        text1: First text
        text2: Second text
        threshold: Minimum proportion of matching words (0.0 to 1.0)
        
    Returns:
        True if texts are similar, False otherwise
    """
    words1 = set(extract_significant_words(text1))
    words2 = set(extract_significant_words(text2))
    
    if not words1:
        return normalize_text(text1) in normalize_text(text2)
    
    matching_words = len(words1.intersection(words2))
    return matching_words >= len(words1) * threshold


def clean_lyrics_text(text: str) -> str:
    """
    Clean lyrics text by removing metadata tags and extra whitespace.
    
    Single Responsibility: Lyrics-specific cleaning.
    
    Args:
        text: Raw lyrics text
        
    Returns:
        Cleaned lyrics text
    """
    text = re.sub(r'[\(\[].*?[\)\]]', '', text)
    lines = [line for line in text.splitlines() if line.strip()]
    return '\n'.join(lines)


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


def normalize_text_advanced(text: str, preserve_newlines: bool = False) -> str:
    """
    Normalize text for comparison with advanced options.
    
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
