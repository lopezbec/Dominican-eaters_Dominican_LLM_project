#!/usr/bin/env python3

from src.clients import GeniusAPIClient, YouTubeAPIClient
from src.services import LyricsService
from src.utils import config, FileHandler


def main() -> None:

    # Get the .env and searches.txt file.
    if not config.validate():
        print("\n Tip: Create a .env file with GENIUS_ACCESS_TOKEN=your_token")
        return
    
    searches = FileHandler.load_searches(config.SEARCHES_FILE)
    
    if not searches:
        print(f" Error: No searches found in '{config.SEARCHES_FILE}'")
        print(f" Tip: Create '{config.SEARCHES_FILE}' with one search per line")
        return
    
    print(f"\n{'='*60}")
    print(f"🎵 Lyrics Eater - Processing {len(searches)} searches")
    print(f"{'='*60}\n")
    
    # Initialize clients
    genius_client = GeniusAPIClient(config.GENIUS_ACCESS_TOKEN)
    youtube_client = YouTubeAPIClient()  # No API key needed
    
    print("YouTube scraper enabled (no API limits!)\n")
    
    # Initialize service
    lyrics_service = LyricsService(genius_client, youtube_client)
    
    songs, successful, failed = lyrics_service.process_multiple_queries(searches)
    
    if songs:
        print(f"\n{'='*60}")
        print(" Saving results...")
        FileHandler.save_to_excel(songs, config.OUTPUT_FILE)
        
        print(f"\n Process completed!")
        print(f"    Successful: {successful}/{len(searches)}")
        print(f"    Failed: {failed}/{len(searches)}")
        print(f"    Output: {config.OUTPUT_FILE}")
    else:
        print("\n No songs were successfully processed")


if __name__ == "__main__":
    main()
