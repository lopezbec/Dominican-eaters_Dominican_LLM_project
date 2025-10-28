"""Configuration management."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """
    Centralized configuration.
    """
    
    # YouTube search settings
    SEARCH_TIMEOUT: int = 30
    VIDEOS_PER_SEARCH: int = 3
    
    # File paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    BOOKS_FILE: str = "books_list.txt"
    OUTPUT_FILE: str = "dominican_audiobooks.xlsx"
    OUTPUT_CSV: str = "dominican_audiobooks.csv"
    
    # Processing settings
    SLEEP_BETWEEN_SEARCHES: int = 2  # Seconds to wait between searches
    
    @classmethod
    def validate(cls) -> bool:
        """
        Validate configuration.
        
        Returns:
            bool: Always True for this project (no API keys needed)
        """
        return True


config = Config()
