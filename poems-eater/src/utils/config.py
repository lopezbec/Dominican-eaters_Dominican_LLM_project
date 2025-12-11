"""Configuration settings for Poems-Eater."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from shared.utils.module_config import ModuleConfig

_base_config = ModuleConfig(module_name="poems", input_file="poems_list.txt")

OUTPUT_FILE = _base_config.OUTPUT_FILE
OUTPUT_CSV = _base_config.OUTPUT_CSV

POEMS_FILE = _base_config.INPUT_FILE

VIDEOS_PER_SEARCH = 3

MIN_VIDEO_DURATION = 30
MAX_VIDEO_DURATION = 1200
