#!/usr/bin/env python3

"""
Poems Eater - Dominican Poetry Recitation Finder
Searches for Dominican poetry recitations on YouTube
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from typing import Any, List
from shared.base.base_runner import BaseEaterRunner
from shared.clients.youtube_client import YouTubeClient
from shared.clients.adapters import YouTubeSearcherAdapter
from shared.utils.file_handler import FileHandler
from src.services import PoemService
from src.utils import config
from src.utils.dominican_poems import get_poems_as_objects
from src.models.poem import Poem
from shared.utils.base_config import BaseConfig
from src.utils.file_handler import load_poems_from_file


class PoemsEaterRunner(BaseEaterRunner):
    
    def __init__(self):
        super().__init__(
            module_name="poems",
            title="Poems Eater - Buscador de Recitaciones de Poemas Dominicanos"
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
        from src.utils.dominican_poems import get_poems_as_objects
        return self.load_with_fallback(
            file_path=self.config.POEMS_FILE,
            fallback_provider=get_poems_as_objects,
            file_description=f"poemas desde '{self.config.POEMS_FILE}'",
            fallback_description="dataset predefinido de poesía dominicana"
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
