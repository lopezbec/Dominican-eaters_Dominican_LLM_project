"""Configuration management using environment variables."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """
    Centralized configuration using class variables.
    """
    
    GENIUS_ACCESS_TOKEN: str = os.getenv("GENIUS_ACCESS_TOKEN", "")
    GENIUS_BASE_URL: str = "https://api.genius.com"
    
    API_TIMEOUT: int = 20
    SCRAPING_TIMEOUT: int = 20
    
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    SEARCHES_FILE: str = "searches.txt"
    OUTPUT_FILE: str = "dominican_songs.xlsx"
    
    RESULTS_PER_PAGE: int = 1
    
    @classmethod
    def validate(cls) -> bool:
        """
        Validate that required configuration is present.
        
        Returns:
            bool: True if config is valid, False otherwise
        """
        if not cls.GENIUS_ACCESS_TOKEN:
            print(" Error: GENIUS_ACCESS_TOKEN not found in .env file")
            return False
        
        return True


config = Config()
