from abc import ABC, abstractmethod
from typing import Any, Tuple, List, Optional
import logging

from shared.services.stats_formatter import ProcessingStats, StatisticsFormatter

logger = logging.getLogger(__name__)


class BaseService(ABC):
    def __init__(self, stats_formatter: Optional[StatisticsFormatter] = None):
        self.stats_formatter = stats_formatter or StatisticsFormatter()

    def process_multiple(
        self, items: List[Any], show_progress: bool = True
    ) -> Tuple[List[Any], ProcessingStats]:
        stats = self._init_stats(len(items))

        for idx, item in enumerate(items, 1):
            try:
                if show_progress:
                    self._show_progress(idx, stats.total, item)

                updated_item, success = self._process_single(item)
                self._update_stats(stats, updated_item, success)

            except KeyboardInterrupt:
                self._handle_interrupt(idx)
                break
            except Exception as e:
                self._handle_error(e, stats)
                continue

        return items, stats

    @abstractmethod
    def _process_single(self, item: Any) -> Tuple[Any, bool]:
        pass

    def _init_stats(self, total: int) -> ProcessingStats:
        return ProcessingStats(total=total)

    @abstractmethod
    def _update_stats(self, stats: ProcessingStats, item: Any, success: bool):
        pass

    def _show_progress(self, current: int, total: int, item: Any):
        logger.info("\n[{current}/{total}] Procesando...")

    def _handle_interrupt(self, processed_count: int):
        logger.info("\n\nProceso interrumpido por el usuario")
        logger.info("Procesados hasta ahora: {processed_count - 1}")

    def _handle_error(self, error: Exception, stats: ProcessingStats):
        logger.info("   Error inesperado: {error}")
        stats.not_found += 1

    def print_statistics(self, stats: ProcessingStats):
        output = self.stats_formatter.format(stats)
        logger.info("\n{output}\n")
