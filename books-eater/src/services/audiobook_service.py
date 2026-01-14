"""Business logic for processing audiobook searches."""

import logging
from typing import Tuple

from shared.services.base_service import BaseService
from shared.services.stats_formatter import ProcessingStats
from shared.interfaces.searchers import YouTubeSearcher
from shared.models.enums import ContentAvailability
from src.models.book import Book

logger = logging.getLogger(__name__)


class AudiobookService(BaseService):
    
    def __init__(self, youtube_client: YouTubeSearcher):
        super().__init__()
        self.youtube_client = youtube_client
    def process_book(self, book: Book) -> Tuple[Book, bool]:
        """
        Process a single book search and update with YouTube info.
        
        Args:
            book: Book object to search for
            
        Returns:
            Tuple of (updated Book object, success boolean)
        """
        logger.info("Searching for audiobook: %s - %s", book.titulo, book.autor)

        result = self.youtube_client.search_audiobook(book.titulo, book.autor)
        
        if result:
            # Determine if it's complete or partial
            is_partial = 'fragmento' in result['type'].lower() or 'parcial' in result['type'].lower()
            
            if is_partial:
                book.mark_as_partial(
                    url=result['url'],
                    duration=result['duration'],
                    content_type=result['type']
                )
                logger.info("Partial match found: %s (%s)", result['type'], result['duration'])
            else:
                book.mark_as_found(
                    url=result['url'],
                    duration=result['duration'],
                    content_type=result['type']
                )
                logger.info("Complete match found: %s (%s)", result['type'], result['duration'])

            return book, True
        else:
            logger.info("No match found for: %s - %s", book.titulo, book.autor)
            return book, False
    
    def _process_single(self, item: Book) -> Tuple[Book, bool]:
        return self.process_book(item)
    
    def _update_stats(self, stats: ProcessingStats, item: Book, success: bool):
        if item.disponibilidad == ContentAvailability.FOUND:
            stats.found += 1
        elif item.disponibilidad == ContentAvailability.PARTIAL:
            stats.partial += 1
        else:
            stats.not_found += 1
