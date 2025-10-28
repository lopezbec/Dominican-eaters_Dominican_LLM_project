"""Business logic for processing audiobook searches."""

from typing import List, Tuple, Dict

from src.clients.youtube_client import YouTubeClient
from src.models.book import Book


class AudiobookService:
    """
    Service for processing audiobook search queries.
    """
    
    def __init__(self, youtube_client: YouTubeClient):
        """
        Initialize the service.
        
        Args:
            youtube_client: YouTube client instance
        """
        self.youtube_client = youtube_client
    
    def process_book(self, book: Book) -> Tuple[Book, bool]:
        """
        Process a single book search and update with YouTube info.
        
        Args:
            book: Book object to search for
            
        Returns:
            Tuple of (updated Book object, success boolean)
        """
        print(f"   Buscando: {book.titulo} - {book.autor}")

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
                print(f"      Parcial encontrado: {result['type']} ({result['duration']})")
            else:
                book.mark_as_found(
                    url=result['url'],
                    duration=result['duration'],
                    content_type=result['type']
                )
                print(f"      Encontrado: {result['type']} ({result['duration']})")

            return book, True
        else:
            print(f"      No encontrado")
            return book, False
    
    def process_multiple_books(
        self,
        books: List[Book],
        show_progress: bool = True
    ) -> Tuple[List[Book], Dict[str, int]]:
        """
        Process multiple books.
        
        Args:
            books: List of Book objects
            show_progress: Whether to show progress messages
            
        Returns:
            Tuple of (updated books list, statistics dictionary)
        """
        stats = {
            'total': len(books),
            'found': 0,
            'partial': 0,
            'not_found': 0
        }
        
        for idx, book in enumerate(books, 1):
            try:
                if show_progress:
                    print(f"\n[{idx}/{stats['total']}] Procesando...")
                
                updated_book, success = self.process_book(book)
                
                if updated_book.disponibilidad == "ENCONTRADO":
                    stats['found'] += 1
                elif updated_book.disponibilidad == "PARCIAL":
                    stats['partial'] += 1
                else:
                    stats['not_found'] += 1
                    
            except KeyboardInterrupt:
                print("\n\nProceso interrumpido por el usuario")
                print(f"Libros procesados hasta ahora: {idx - 1}")
                break
            except Exception as e:
                print(f"   Error inesperado: {e}")
                stats['not_found'] += 1
                continue
        
        return books, stats
    
    def print_statistics(self, stats: Dict[str, int]):
        """
        Print search statistics.
        
        Args:
            stats: Statistics dictionary
        """
        print(f"\n{'='*60}")
        print("Resultados de la búsqueda:")
        print(f"{'='*60}")
        print(f"   Total procesado: {stats['total']}")
        print(f"   Encontrados: {stats['found']} ({stats['found']/stats['total']*100:.1f}%)")
        print(f"   Parciales: {stats['partial']} ({stats['partial']/stats['total']*100:.1f}%)")
        print(f"   No encontrados: {stats['not_found']} ({stats['not_found']/stats['total']*100:.1f}%)")

        success_rate = (stats['found'] + stats['partial']) / stats['total'] * 100
        print(f"\n   Tasa de éxito: {success_rate:.1f}%")
        print(f"{'='*60}\n")
