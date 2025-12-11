"""Common file I/O utilities for reading and writing data."""

import os
from pathlib import Path
from typing import List, Optional, Callable, TypeVar

T = TypeVar('T')


def ensure_directory_exists(directory: str) -> bool:
    """
    Ensure a directory exists, creating it if necessary.
    
    Single Responsibility: Directory creation.
    
    Args:
        directory: Path to directory
        
    Returns:
        True if directory exists or was created successfully
    """
    try:
        os.makedirs(directory, exist_ok=True)
        return True
    except Exception as e:
        print(f"Error creating directory {directory}: {e}")
        return False


def load_items_from_text_file(
    filename: str,
    parser: Callable[[int, str], Optional[T]]
) -> Optional[List[T]]:
    """
    Load items from a text file using a custom parser function.
    
    Single Responsibility: Generic text file loading with custom parsing.
    
    Args:
        filename: Path to text file
        parser: Function that takes (line_number, line_text) and returns parsed item
        
    Returns:
        List of parsed items or None if file not found
    """
    try:
        if not os.path.exists(filename):
            return None
        
        items = []
        with open(filename, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f, 1):
                line = line.strip()
                
                if not line or line.startswith('#'):
                    continue
                
                item = parser(idx, line)
                if item:
                    items.append(item)
        
        return items if items else None
        
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None


def save_text_file(filename: str, content: str) -> bool:
    """
    Save text content to a file.
    
    Single Responsibility: Write text to file.
    
    Args:
        filename: Output filename
        content: Text content to write
        
    Returns:
        True if successful, False otherwise
    """
    try:
        directory = os.path.dirname(filename)
        if directory:
            ensure_directory_exists(directory)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return True
        
    except Exception as e:
        print(f"Error writing to {filename}: {e}")
        return False


def get_safe_filename(text: str, max_length: int = 50) -> str:
    """
    Convert text to a safe filename by removing invalid characters.
    
    Single Responsibility: Filename sanitization.
    
    Args:
        text: Text to convert
        max_length: Maximum filename length
        
    Returns:
        Safe filename string
    """
    import re
    safe = re.sub(r'[^\w\s-]', '', text)
    safe = re.sub(r'[-\s]+', '_', safe)
    return safe[:max_length]


def find_files_by_pattern(directory: str, pattern: str) -> List[Path]:
    """
    Find all files in a directory matching a pattern.
    
    Single Responsibility: File discovery.
    
    Args:
        directory: Directory to search
        pattern: Glob pattern (e.g., "*.json", "*.txt")
        
    Returns:
        List of Path objects matching pattern
    """
    try:
        dir_path = Path(directory)
        if not dir_path.exists():
            return []
        return list(dir_path.glob(pattern))
    except Exception as e:
        print(f"Error searching directory {directory}: {e}")
        return []
