#!/usr/bin/env python3

"""
Poems Eater - Dominican Poetry Recitation Finder
Searches for Dominican poetry recitations on YouTube
"""

from src.clients import YouTubeClient
from src.services import PoemService
from src.utils import config, FileHandler, get_poems_as_objects


def main() -> None:
    """Main entry point for the application."""
    
    print(f"\n{'='*70}")
        print("Poems Eater - Buscador de Recitaciones de Poemas Dominicanos")
    print(f"{'='*70}\n")
    
    # Try to load poems from file first
        poems = FileHandler.load_poems_from_file(config.POEMS_FILE)
    
    if poems:
            print(f"Cargados {len(poems)} poemas desde '{config.POEMS_FILE}'")
    else:
        # Use predefined dataset
            print(f"Usando dataset predefinido de poesía dominicana")
        poems = get_poems_as_objects()
            print(f"{len(poems)} poemas en el dataset")
    
    print(f"\n{'='*70}")
    print("Iniciando búsqueda en YouTube...")
    print(f"{'='*70}\n")
    
    # Initialize YouTube client (no API key needed!)
    youtube_client = YouTubeClient(videos_per_search=config.VIDEOS_PER_SEARCH)
        print("Cliente de YouTube inicializado (sin límites de API!)\n")
    
    # Initialize service
    poem_service = PoemService(youtube_client)
    
    # Process all poems
    poems, stats = poem_service.process_multiple_poems(poems)
    
    if poems:
        print(f"\n{'='*70}")
            print("Guardando resultados...")
        print(f"{'='*70}\n")
        
        # Save to Excel
        FileHandler.save_to_excel(poems, config.OUTPUT_FILE)
        
        # Optionally save to CSV
        FileHandler.save_to_csv(poems, config.OUTPUT_CSV)
        
        # Print statistics
        poem_service.print_statistics(stats)
        
            print(f"Archivos generados:")
        print(f"   - {config.OUTPUT_FILE}")
        print(f"   - {config.OUTPUT_CSV}")
        
    else:
            print("\nNo se procesaron poemas")


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
