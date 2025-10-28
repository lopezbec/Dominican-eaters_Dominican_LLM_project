import requests
import re
import os
import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openpyxl.styles import Alignment

load_dotenv()

class GeniusAPIClient:
    BASE_URL = "https://api.genius.com"
    
    def __init__(self, access_token):
        """
        Initilize the client access token.

        Parameters:
        - access_token: Access token for Genius API.
        """
        self.access_token = access_token
    
    def _make_request(self, endpoint, method="GET", params=None, timeout=20):
        url = f"{self.BASE_URL}{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "User-Agent": "GeniusAPIClient/1.0"
        }
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=timeout)
            response.raise_for_status()
            result = response.json()
            
            if result.get("meta", {}).get("status") == 200:
                return result.get("response")
            else:
                print(f"Response error: {result}")
                return None
                
        except requests.exceptions.Timeout:
            print(f"⏱️  Timeout error: Request took longer than {timeout} seconds")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            return None
    
    def search(self, q, per_page=5):
        """Search songs in Genius."""
        params = {"q": q, "per_page": per_page}
        response = self._make_request("/search", params=params)
        
        if response:
            hits = response.get("hits", [])
            results = []
            
            for hit in hits:
                song = hit.get("result", {})
                results.append({
                    "id": song.get("id"),
                    "title": song.get("title"),
                    "artist": song.get("primary_artist", {}).get("name"),
                    "url": song.get("url")
                })
            
            return results
        return []
    
    def get_song(self, song_id):
        """Obtain complete information details of a song"""

        response = self._make_request(f"/songs/{song_id}")
        
        if response:
            song = response.get("song", {})
            
            album = song.get("album") or {}
            label = album.get("label", "N/A") if album else "N/A"
            
            tags = song.get("tags", [])
            genres = ", ".join([tag.get("name", "") for tag in tags]) if tags else "N/A"
            
            return {
                "id": song.get("id"),
                "title": song.get("title"),
                "artist": song.get("primary_artist", {}).get("name"),
                "url": song.get("url"),
                "genres": genres,
                "label": label,
                "release_date": song.get("release_date_for_display", "N/A"),
                "album": album.get("name", "N/A") if album else "N/A",
                "status": "success"
            }
        return {"status": "error"}
    
    def scrape_song_lyrics(self, url, timeout=20):
        """
        Scrapes the lyrics of a song from a given Genius.com URL.
        
        Parameters:
        - url: URL of the Genius.com song page.
        - timeout: Maximum time to wait for the request (in seconds)

        Returns:
        - A string containing the song lyrics, cleaned and formatted.
        """
        try:
            page = requests.get(url, timeout=timeout)
            page.raise_for_status()
            
            html = BeautifulSoup(page.text, 'html.parser')

            lyrics_divs = html.find_all('div', attrs={'data-lyrics-container': 'true'})
            
            if not lyrics_divs:
                print(f"Could not find lyrics for {url}")
                return ""

            lyrics = '\n'.join([div.get_text(separator="\n") for div in lyrics_divs])
            
            lyrics = re.sub(r'[\(\[].*?[\)\]]', '', lyrics)
            
            lyrics = os.linesep.join([s for s in lyrics.splitlines() if s])
            
            return lyrics
            
        except requests.exceptions.Timeout:
            print(f"  Timeout error: Scraping took longer than {timeout} seconds")
            return ""
        except requests.exceptions.RequestException as e:
            print(f"Error obtaining the page: {e}")
            return ""
        except Exception as e:
            print(f"Error getting the lyrics: {e}")
            return ""
    
    def save_to_excel(self, songs_data, filename="genius_songs.xlsx"):
        """
        Save songs information in one archive type excel.
        
        Parameters:
        - songs_data: Ditcionarie songs list.
        - filename: just the filename thats gonna create.
        """
        try:
            df = pd.DataFrame(songs_data)
            
            df = df[['genero', 'artista', 'cancion', 'letras', 'enlace_genius', 'discografica']]
            
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Canciones')
                
                worksheet = writer.sheets['Canciones']
                worksheet.column_dimensions['A'].width = 20  # genero
                worksheet.column_dimensions['B'].width = 25  # artista
                worksheet.column_dimensions['C'].width = 30  # cancion
                worksheet.column_dimensions['D'].width = 80  # letras
                worksheet.column_dimensions['E'].width = 50  # enlace_genius
                worksheet.column_dimensions['F'].width = 25  # discografica
                
                for row in range(2, len(songs_data) + 2):
                    cell = worksheet.cell(row=row, column=4)  # D column (letras)
                    cell.alignment = Alignment(wrap_text=True, vertical='top')
            
            print(f"Excel successfuly saved in {filename}")
            return True
            
        except Exception as e:
            print(f"Error saving the excel: {e}")
            return False


def load_searches_from_file(filename="searches.txt"):
    """
    Carga búsquedas desde un archivo de texto.
    Cada línea del archivo debe contener una búsqueda.
    
    Parameters:
    - filename: Nombre del archivo con las búsquedas
    
    Returns:
    - Lista de búsquedas
    """
    try:
        if not os.path.exists(filename):
            return None
        
        with open(filename, 'r', encoding='utf-8') as f:
            searches = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
        
        return searches if searches else None
    except Exception as e:
        print(f"Error leyendo archivo {filename}: {e}")
        return None


def main():
    ACCESS_TOKEN = os.getenv("GENIUS_ACCESS_TOKEN")
    
    if not ACCESS_TOKEN:
        print("You have to create an Access token .env")
        return
    
    client = GeniusAPIClient(ACCESS_TOKEN)
    
    searches = load_searches_from_file("searches.txt")
    
    print(f"\n{'='*60}")
    print(f" Procesando {len(searches)} búsquedas...")
    print(f"{'='*60}\n")
    
    songs_data = []
    successful = 0
    failed = 0
    
    for idx, search_query in enumerate(searches, 1):
        try:
            print(f"\n[{idx}/{len(searches)}] Buscando: '{search_query}'...")
            results = client.search(search_query, per_page=1)  # Solo el primer resultado
            
            if not results:
                print(f"  No se encontraron resultados para '{search_query}'")
                failed += 1
                continue
            
            song = results[0]  # Primer y único resultado
            print(f"  Encontrado: {song['title']} - {song['artist']}")
            
            # Obtener información detallada
            song_info = client.get_song(song['id'])
            
            if song_info.get("status") != "success":
                print(f"  No se pudieron obtener los detalles")
                failed += 1
                continue
            
            print(f"    Álbum: {song_info['album']}")
            print(f"    Género(s): {song_info['genres']}")
            print(f"    Sello: {song_info['label']}")
            
            # Scrapear letras
            print(f"    Obteniendo letras...")
            lyrics = client.scrape_song_lyrics(song_info['url'])
            
            if lyrics:
                print(f"    Letras obtenidas ({len(lyrics)} caracteres)")
            else:
                print(f"    No se pudieron obtener las letras")
                lyrics = "N/A"
            
            # Agregar a la lista
            songs_data.append({
                'genero': song_info['genres'],
                'artista': song_info['artist'],
                'cancion': song_info['title'],
                'letras': lyrics,
                'enlace_genius': song_info['url'],
                'discografica': song_info['label']
            })
            successful += 1
            
        except KeyboardInterrupt:
            print("\n\n  Proceso interrumpido por el usuario")
            print(f"Canciones procesadas hasta ahora: {successful}")
            break
        except Exception as e:
            print(f"  Error inesperado procesando '{search_query}': {e}")
            failed += 1
            continue
    
    # Guardar resultados
    if songs_data:
        print(f"\n{'='*60}")
        print(" Guardando resultados...")
        excel_filename = "genius_songs.xlsx"
        client.save_to_excel(songs_data, excel_filename)
        
        print(f"\n Proceso completado!")
        print(f"   Exitosas: {successful}/{len(searches)}")
        print(f"   Fallidas: {failed}/{len(searches)}")
        print(f"   Archivo: {excel_filename}")
    else:
        print(f"\nNo se pudo obtener ninguna canción")


if __name__ == "__main__":
    main()
