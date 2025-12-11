"""Configuration management."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from shared.utils.module_config import ModuleConfig

load_dotenv()

_base_config = ModuleConfig(module_name="audiobooks", input_file="books_list.txt")


class Config:
    
    PROJECT_ROOT: Path = _base_config.PROJECT_ROOT
    BOOKS_FILE: str = _base_config.INPUT_FILE
    OUTPUT_FILE: str = _base_config.OUTPUT_FILE
    OUTPUT_CSV: str = _base_config.OUTPUT_CSV
    
    SEARCH_TIMEOUT: int = 30
    VIDEOS_PER_SEARCH: int = 3
    
    SLEEP_BETWEEN_SEARCHES: int = 2
    
    @classmethod
    def validate(cls) -> bool:
        return True


config = Config()
