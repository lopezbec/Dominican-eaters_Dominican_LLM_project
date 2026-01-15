#!/usr/bin/env python3
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
from typing import Any, List, Tuple
from shared.base.base_runner import BaseEaterRunner
from shared.clients.genius_client import GeniusAPIClient
from shared.clients.youtube_client import YouTubeClient
from shared.clients.adapters import GeniusSearcherAdapter, YouTubeSearcherAdapter
from src.services import LyricsService
from src.utils import config
from shared.utils.base_config import BaseConfig
from src.models.song import Song

logger = logging.getLogger(__name__)


class LyricsEaterRunner(BaseEaterRunner):
    def __init__(self):
        super().__init__(
            module_name="lyrics", title="Lyrics Eater - Dominican Songs Processor"
        )

    def load_config(self) -> BaseConfig:
        return config

    def get_client(self) -> Tuple[GeniusSearcherAdapter, YouTubeSearcherAdapter]:
        genius_client = GeniusAPIClient(self.config.GENIUS_ACCESS_TOKEN)
        genius_adapter = GeniusSearcherAdapter(genius_client)

        youtube_client = YouTubeClient()
        youtube_adapter = YouTubeSearcherAdapter(youtube_client)

        logger.info("YouTube scraper enabled (no API limits!)\n")
        return (genius_adapter, youtube_adapter)

    def get_service(
        self, client: Tuple[GeniusSearcherAdapter, YouTubeSearcherAdapter]
    ) -> LyricsService:
        return LyricsService(client[0], client[1])

    def load_input_data(self) -> List[Any]:
        if not self.config.validate():
            logger.info(
                "\n Tip: Create a .env file with GENIUS_ACCESS_TOKEN=your_token"
            )
            return []

        from shared.utils.file_handler import FileHandler

        searches = FileHandler.load_searches(self.config.SEARCHES_FILE)

        if not searches:
            logger.info(" Error: No searches found in '{self.config.SEARCHES_FILE}'")
            logger.info(
                " Tip: Create '{self.config.SEARCHES_FILE}' with one search per line"
            )
            return []

        logger.info("Processing {len(searches)} searches")
        return searches

    def save_results(self, items: List[Any]) -> bool:
        return self.save_dual_format(items)

    def _get_model_class(self) -> type:
        return Song

    def _get_output_file(self) -> str:
        return self.config.OUTPUT_FILE

    def _get_output_csv(self) -> str:
        return self.config.OUTPUT_CSV


def main() -> None:
    from shared.base.base_runner import BaseEaterRunner

    BaseEaterRunner.create_and_run(LyricsEaterRunner)


if __name__ == "__main__":
    main()
