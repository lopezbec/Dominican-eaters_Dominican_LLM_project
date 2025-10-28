"""Business logic for processing poem recitation searches."""

from typing import List, Tuple, Dict
from collections import Counter

from src.clients.youtube_client import YouTubeClient
from src.models.poem import Poem


class PoemService:
    """
    Service for processing poem recitation search queries.
    """
    
    def __init__(self, youtube_client: YouTubeClient):
        """
        Initialize the service.
        
        Args:
            youtube_client: YouTube client instance
        """
        self.youtube_client = youtube_client
    
    def process_poem(self, poem: Poem) -> Tuple[Poem, bool]:
        """
        Process a single poem search and update with YouTube info.
        
        Args:
            poem: Poem object to search for
            
        Returns:
            Tuple of (updated Poem object, success boolean)
        """
        print(f"   Buscando: {poem.titulo} - {poem.autor} ({poem.genero})")
        
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
                print(f"      Parcial: {result['type']} ({result['duration']}) - {result.get('recitador', 'N/A')}")
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
                print(f"      Encontrado: {result['type']} ({result['duration']}) - {result.get('recitador', 'N/A')}")
                return poem, True
        else:
            print(f"      No encontrado")
            return poem, False
    
    def process_multiple_poems(
        self,
        poems: List[Poem],
        show_progress: bool = True
    ) -> Tuple[List[Poem], Dict[str, any]]:
        """
        Process multiple poems.
        
        Args:
            poems: List of Poem objects
            show_progress: Whether to show progress messages
            
        Returns:
            Tuple of (updated poems list, statistics dictionary)
        """
        stats = {
            'total': len(poems),
            'found': 0,
            'partial': 0,
            'not_found': 0,
            'authors': [],
            'genres': [],
            'content_types': [],
            'qualities': [],
            'total_duration': 0,
            'duration_count': 0
        }
        
        for idx, poem in enumerate(poems, 1):
            try:
                if show_progress:
                    print(f"\n[{idx}/{stats['total']}] Procesando...")
                
                updated_poem, success = self.process_poem(poem)
                
                if updated_poem.disponibilidad == "ENCONTRADO":
                    stats['found'] += 1
                    stats['authors'].append(updated_poem.autor)
                    stats['genres'].append(updated_poem.genero)
                    stats['content_types'].append(updated_poem.tipo_contenido)
                    stats['qualities'].append(updated_poem.calidad)
                    
                    duration_parts = updated_poem.duracion.split(':')
                    if len(duration_parts) == 2 and duration_parts[0].isdigit():
                        minutes = int(duration_parts[0])
                        stats['total_duration'] += minutes
                        stats['duration_count'] += 1
                        
                elif updated_poem.disponibilidad == "PARCIAL":
                    stats['partial'] += 1
                    stats['authors'].append(updated_poem.autor)
                    stats['genres'].append(updated_poem.genero)
                else:
                    stats['not_found'] += 1
                    
            except KeyboardInterrupt:
                print("\n\n  Proceso interrumpido por el usuario")
                print(f"Poemas procesados hasta ahora: {idx - 1}")
                break
            except Exception as e:
                print(f"    Error inesperado: {e}")
                stats['not_found'] += 1
                continue
        
        return poems, stats
    
    def print_statistics(self, stats: Dict[str, any]):
        """
        Print detailed search statistics.
        
        Args:
            stats: Statistics dictionary
        """
        print(f"\n{'='*70}")
        print(" RESULTADOS DE LA BÚSQUEDA:")
        print(f"{'='*70}")
        
        print(f"\n Estadísticas Generales:")
        print(f"   Total procesado: {stats['total']}")
        print(f"    Encontrados: {stats['found']} ({stats['found']/stats['total']*100:.1f}%)")
        print(f"     Parciales: {stats['partial']} ({stats['partial']/stats['total']*100:.1f}%)")
        print(f"    No encontrados: {stats['not_found']} ({stats['not_found']/stats['total']*100:.1f}%)")
        
        success_rate = (stats['found'] + stats['partial']) / stats['total'] * 100
        print(f"\n    Tasa de éxito: {success_rate:.1f}%")
        
        if stats['authors']:
            author_counts = Counter(stats['authors'])
            print(f"\n Autores Más Representados:")
            for author, count in author_counts.most_common(5):
                print(f"   - {author}: {count} poemas")
        
        if stats['genres']:
            genre_counts = Counter(stats['genres'])
            print(f"\n Géneros Más Encontrados:")
            for genre, count in genre_counts.most_common(5):
                if genre != "N/A":
                    print(f"   - {genre}: {count} poemas")
        
        if stats['content_types']:
            type_counts = Counter(stats['content_types'])
            print(f"\n Tipos de Contenido:")
            for content_type, count in type_counts.most_common():
                print(f"   - {content_type}: {count} videos")
        
        if stats['qualities']:
            quality_counts = Counter(stats['qualities'])
            print(f"\n Distribución de Calidad:")
            quality_order = ['Excelente', 'Buena', 'Aceptable', 'Baja']
            for quality in quality_order:
                count = quality_counts.get(quality, 0)
                if count > 0:
                    print(f"   - {quality}: {count} videos")
        
        if stats['duration_count'] > 0:
            avg_duration = stats['total_duration'] / stats['duration_count']
            print(f"\n⏱  Duración Promedio: {avg_duration:.1f} minutos")
        
        print(f"\n{'='*70}\n")
