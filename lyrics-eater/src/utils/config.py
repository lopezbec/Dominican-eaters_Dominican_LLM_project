"""Configuration management using environment variables."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from shared.utils.base_config import BaseConfig

load_dotenv()


class Config(BaseConfig):
    
    def __init__(self):
        super().__init__(
            module_name="lyrics-eater",
            input_file="searches.txt",
            API_TIMEOUT=20,
            SCRAPING_TIMEOUT=20,
            RESULTS_PER_PAGE=1
        )
        
        self.SEARCHES_FILE = self.INPUT_FILE
        self.GENIUS_ACCESS_TOKEN = os.getenv("GENIUS_ACCESS_TOKEN", "")
        self.GENIUS_BASE_URL = "https://api.genius.com"
        self.API_TIMEOUT = 20
        self.SCRAPING_TIMEOUT = 20
        self.RESULTS_PER_PAGE = 1
    
    def validate(self) -> bool:
        if not self.GENIUS_ACCESS_TOKEN:
            print(" Error: GENIUS_ACCESS_TOKEN not found in .env file")
            return False
        
        return True


config = Config()
