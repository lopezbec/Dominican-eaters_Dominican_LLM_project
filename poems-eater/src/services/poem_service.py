"""Business logic for processing poem recitation searches."""

import logging
from typing import Any, Tuple
from collections import Counter

from shared.services.base_service import BaseService
from shared.services.stats_formatter import ProcessingStats, DetailedStatisticsFormatter
from shared.interfaces.searchers import YouTubeSearcher
from shared.models.enums import ContentAvailability
from src.models.poem import Poem

logger = logging.getLogger(__name__)


class PoemService(BaseService):
    
    def __init__(self, youtube_client: YouTubeSearcher):
        super().__init__()
        self.youtube_client = youtube_client
        self.formatter = DetailedStatisticsFormatter()
    
    def process_poem(self, poem: Poem) -> Tuple[Poem, bool]:
        """
        Process a single poem search and update with YouTube info.
        
        Args:
            poem: Poem object to search for
            
        Returns:
            Tuple of (updated Poem object, success boolean)
        """
        logger.info("Searching for poem recitation: %s - %s (%s)", poem.titulo, poem.autor, poem.genero)
        
        result = self.youtube_client.search_poem_recitation(
            poem.titulo, 
            poem.autor,
            poem.genero
        )
        
        if result:
            is_partial = 'fragmento' in result['type'].lower() or 'parcial' in result['type'].lower()
            
            notas = f"Video: {result['title'][:100]}"
            
            if is_partial:
                poem.mark_as_partial(
                    url=result['url'],
                    duration=result['duration'],
                    content_type=result['type'],
                    recitador=result.get('recitador', 'N/A'),
                    calidad=result.get('quality', 'Aceptable'),
                    notas=notas
                )
                logger.info("Partial match found: %s (%s) - %s", result['type'], result['duration'], result.get('recitador', 'N/A'))
                return poem, True
            else:
                poem.mark_as_found(
                    url=result['url'],
                    duration=result['duration'],
                    content_type=result['type'],
                    recitador=result.get('recitador', 'N/A'),
                    calidad=result.get('quality', 'Buena'),
                    notas=notas
                )
                logger.info("Complete match found: %s (%s) - %s", result['type'], result['duration'], result.get('recitador', 'N/A'))
                return poem, True
        else:
            logger.info("No match found for: %s - %s", poem.titulo, poem.autor)
            return poem, False
    
    def _process_single(self, item: Poem) -> Tuple[Poem, bool]:
        return self.process_poem(item)
    
    def _init_stats(self, total: int) -> ProcessingStats:
        stats = ProcessingStats(total=total)
        stats.metadata: dict[str, Any] = {
            'authors': [],
            'genres': [],
            'content_types': [],
            'qualities': [],
            'total_duration': 0,
            'duration_count': 0
        }
        return stats
    
    def _update_stats(self, stats: ProcessingStats, item: Poem, success: bool):
        if item.disponibilidad == ContentAvailability.FOUND:
            stats.found += 1
            stats.metadata['authors'].append(item.autor)
            stats.metadata['genres'].append(item.genero)
            stats.metadata['content_types'].append(item.tipo_contenido)
            stats.metadata['qualities'].append(item.calidad)
            
            duration_parts = item.duracion.split(':')
            if len(duration_parts) == 2 and duration_parts[0].isdigit():
                minutes = int(duration_parts[0])
                stats.metadata['total_duration'] += minutes
                stats.metadata['duration_count'] += 1
                
        elif item.disponibilidad == ContentAvailability.PARTIAL:
            stats.partial += 1
            stats.metadata['authors'].append(item.autor)
            stats.metadata['genres'].append(item.genero)
        else:
            stats.not_found += 1
    
    def print_statistics(self, stats: ProcessingStats):
        print(f"\n{'='*70}")
        print(" RESULTADOS DE LA BÚSQUEDA:")
        print(f"{'='*70}")
        
        print(f"\n Estadísticas Generales:")
        print(f"   Total procesado: {stats.total}")
        print(f"    Encontrados: {stats.found} ({stats.found_rate:.1f}%)")
        print(f"     Parciales: {stats.partial} ({stats.partial_rate:.1f}%)")
        print(f"    No encontrados: {stats.not_found} ({stats.not_found_rate:.1f}%)")
        
        print(f"\n    Tasa de éxito: {stats.success_rate:.1f}%")
        
        if stats.metadata.get('authors'):
            author_counts = Counter(stats.metadata['authors'])
            print(f"\n Autores Más Representados:")
            for author, count in author_counts.most_common(5):
                print(f"   - {author}: {count} poemas")
        
        if stats.metadata.get('genres'):
            genre_counts = Counter(stats.metadata['genres'])
            print(f"\n Géneros Más Encontrados:")
            for genre, count in genre_counts.most_common(5):
                if genre != "N/A":
                    print(f"   - {genre}: {count} poemas")
        
        if stats.metadata.get('content_types'):
            type_counts = Counter(stats.metadata['content_types'])
            print(f"\n Tipos de Contenido:")
            for content_type, count in type_counts.most_common():
                print(f"   - {content_type}: {count} videos")
        
        if stats.metadata.get('qualities'):
            quality_counts = Counter(stats.metadata['qualities'])
            print(f"\n Distribución de Calidad:")
            quality_order = ['Excelente', 'Buena', 'Aceptable', 'Baja']
            for quality in quality_order:
                count = quality_counts.get(quality, 0)
                if count > 0:
                    print(f"   - {quality}: {count} videos")
        
        if stats.metadata.get('duration_count', 0) > 0:
            avg_duration = stats.metadata['total_duration'] / stats.metadata['duration_count']
            print(f"\n⏱  Duración Promedio: {avg_duration:.1f} minutos")
        
        print(f"\n{'='*70}\n")

