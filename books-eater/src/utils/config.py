"""Configuration management."""

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
            module_name="audiobooks",
            input_file="books_list.txt",
            SEARCH_TIMEOUT=30,
            VIDEOS_PER_SEARCH=3,
            SLEEP_BETWEEN_SEARCHES=2
        )
        
        self.BOOKS_FILE = self.INPUT_FILE
        self.SEARCH_TIMEOUT = 30
        self.VIDEOS_PER_SEARCH = 3
        self.SLEEP_BETWEEN_SEARCHES = 2


config = Config()

