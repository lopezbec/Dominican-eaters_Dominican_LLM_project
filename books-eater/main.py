#!/usr/bin/env python3

"""
Books Eater - Dominican Audiobooks Finder
Searches for Dominican literature audiobooks on YouTube
"""

from src.clients import YouTubeClient
from src.services import AudiobookService
from src.utils import config, FileHandler, DOMINICAN_BOOKS
from src.utils.dominican_books import get_books_as_objects


def main() -> None:
    """Main entry point for the application."""
    print(f"\n{'='*60}")
    print("Books Eater - Buscador de Audiolibros Dominicanos")
    print(f"{'='*60}\n")
    
    # Try to load books from file first
    books = FileHandler.load_books_from_file(config.BOOKS_FILE)

    if books:
        print(f"Cargados {len(books)} libros desde '{config.BOOKS_FILE}'")
    else:
        # Use predefined dataset
        print(f"Usando dataset predefinido de literatura dominicana")
        books = get_books_as_objects()
        print(f"{len(books)} libros en el dataset")
    
    print(f"\n{'='*60}")
    print("Iniciando búsqueda en YouTube...")
    print(f"{'='*60}\n")
    
    # Initialize YouTube client (no API key needed!)
    youtube_client = YouTubeClient(videos_per_search=config.VIDEOS_PER_SEARCH)
    print("Cliente de YouTube inicializado (sin límites de API!)\n")
    
    # Initialize service
    audiobook_service = AudiobookService(youtube_client)
    
    # Process all books
    books, stats = audiobook_service.process_multiple_books(books)
    
    if books:
        print(f"\n{'='*60}")
        print("Guardando resultados...")
        print(f"{'='*60}\n")
        
        # Save to Excel
        FileHandler.save_to_excel(books, config.OUTPUT_FILE)
        
        # Optionally save to CSV
        FileHandler.save_to_csv(books, config.OUTPUT_CSV)
        
        # Print statistics
        audiobook_service.print_statistics(stats)

        print(f"Archivos generados:")
        print(f"   - {config.OUTPUT_FILE}")
        print(f"   - {config.OUTPUT_CSV}")
        
    else:
        print("\nNo se procesaron libros")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPrograma interrumpido por el usuario")
        print("¡Hasta luego!")
    except Exception as e:
        print(f"\nError fatal: {e}")
        import traceback
        traceback.print_exc()
