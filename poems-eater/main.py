#!/usr/bin/env python3

"""
Poems Eater - Dominican Poetry Recitation Finder
Searches for Dominican poetry recitations on YouTube
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
from pathlib import Path
from typing import List
from shared.base.base_runner import BaseEaterRunner
from shared.clients.youtube_client import YouTubeClient
from shared.clients.adapters import YouTubeSearcherAdapter
from src.services import PoemService
from src.utils import config
from src.utils.dominican_poems import get_poems_as_objects
from src.models.poem import Poem
from shared.utils.base_config import BaseConfig

logger = logging.getLogger(__name__)


class PoemsEaterRunner(BaseEaterRunner):
    def __init__(self):
        super().__init__(
            module_name="poems",
            title="Poems Eater - Buscador de Recitaciones de Poemas Dominicanos",
        )

    def load_config(self) -> BaseConfig:
        return config

    def get_client(self) -> YouTubeSearcherAdapter:
        youtube_client = YouTubeClient(videos_per_search=self.config.VIDEOS_PER_SEARCH)
        logger.info("Cliente de YouTube inicializado (sin límites de API!)\n")
        return YouTubeSearcherAdapter(youtube_client)

    def get_service(self, client: YouTubeSearcherAdapter) -> PoemService:
        return PoemService(client)

    def load_input_data(self) -> List[Poem]:
        """
        Load poem data with smart resume support.
        
        Tries to load from existing Excel file first (contains URLs already found).
        Falls back to predefined dataset if Excel doesn't exist.
        
        This allows the scraper to skip poems that already have YouTube URLs,
        dramatically reducing runtime on subsequent runs.
        
        Returns:
            List of Poem objects
        """
        excel_path = Path(self.config.OUTPUT_FILE)
        
        # Try loading from Excel first (performance optimization)
        if excel_path.exists():
            logger.info(f"📂 Found existing data at {excel_path}")
            poems = self.load_from_excel(excel_path)
            if poems:
                logger.info(f"✓ Loaded {len(poems)} poems from Excel (will skip already found)")
                return poems
        
        # Fallback to predefined dataset if no Excel exists
        logger.info("🆕 No existing data found, using predefined dataset")
        return self.load_with_fallback(
            file_path=self.config.POEMS_FILE,
            fallback_provider=get_poems_as_objects,
            file_description=f"poemas desde '{self.config.POEMS_FILE}'",
            fallback_description="dataset predefinido de poesía dominicana",
        )

    def save_results(self, poems: List[Poem]) -> bool:
        return self.save_dual_format(poems)

    def _get_model_class(self) -> type:
        from src.models.poem import Poem

        return Poem

    def _get_output_file(self) -> str:
        return self.config.OUTPUT_FILE

    def _get_output_csv(self) -> str:
        return self.config.OUTPUT_CSV


def main() -> None:
    """Main entry point for application."""
    from shared.base.base_runner import BaseEaterRunner

    BaseEaterRunner.create_and_run(PoemsEaterRunner)


if __name__ == "__main__":
    main()
