"""Configuration management using environment variables."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from shared.utils.module_config import ModuleConfig

load_dotenv()

_base_config = ModuleConfig(module_name="songs", input_file="searches.txt")


class Config:
    
    PROJECT_ROOT: Path = _base_config.PROJECT_ROOT
    SEARCHES_FILE: str = _base_config.INPUT_FILE
    OUTPUT_FILE: str = _base_config.OUTPUT_FILE
    
    GENIUS_ACCESS_TOKEN: str = os.getenv("GENIUS_ACCESS_TOKEN", "")
    GENIUS_BASE_URL: str = "https://api.genius.com"
    
    API_TIMEOUT: int = 20
    SCRAPING_TIMEOUT: int = 20
    
    RESULTS_PER_PAGE: int = 1
    
    @classmethod
    def validate(cls) -> bool:
        if not cls.GENIUS_ACCESS_TOKEN:
            print(" Error: GENIUS_ACCESS_TOKEN not found in .env file")
            return False
        
        return True


config = Config()
