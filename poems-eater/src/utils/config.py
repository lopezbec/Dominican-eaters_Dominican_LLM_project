"""Configuration settings for Poems-Eater."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from shared.utils.base_config import BaseConfig


class Config(BaseConfig):
    
    def __init__(self):
        super().__init__(
            module_name="poems",
            input_file="poems_list.txt",
            VIDEOS_PER_SEARCH=3,
            MIN_VIDEO_DURATION=30,
            MAX_VIDEO_DURATION=1200
        )
        
        self.POEMS_FILE = self.INPUT_FILE
        self.VIDEOS_PER_SEARCH = 3
        self.MIN_VIDEO_DURATION = 30
        self.MAX_VIDEO_DURATION = 1200


config = Config()
