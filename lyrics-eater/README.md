# Lyrics Eater

A Python tool to automatically scrape song lyrics from Genius.com and fetch corresponding YouTube video links.

## Features

- 🔍 **Automatic song search** on Genius
- 📝 **Metadata extraction**: artist, album, genre, record label
- 🎵 **Full lyric scraping**
- 🎬 **YouTube video links** for each song (optional)
- 📊 **Professional Excel export**
- 📦 **Batch processing** from a text file
- ⚡ **Robust error handling** and timeouts

## Project Structure

```
lyrics-eater/
├── src/
│   ├── clients/
│   ├── services/
│   ├── models/
│   └── utils/
├── main.py
├── .env
├── searches.txt
└── requirements.txt
```

## Setup

1.  **Clone the repository**
    ```bash
    git clone <your-repo>
    cd lyrics-eater
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up Genius Token**
    - Go to https://genius.com/api-clients
    - Create an application and copy the "Client Access Token".
    - Create a `.env` file with your token:
      ```env
      GENIUS_ACCESS_TOKEN=your_genius_token_here
      ```
    > **Note:** YouTube links are scraped automatically without an API key.

4.  **Create Searches File**
    Edit `searches.txt` with one song per line:
    ```
    Obsesión - Aventura
    Propuesta Indecente - Romeo Santos
    Bachata Rosa - Juan Luis Guerra
    ```

## Usage

### Basic Execution
```bash
python main.py
```

### Output

The script generates a `genius_songs.xlsx` file with the following columns:

| Column | Description |
|---|---|
| **genre** | Song's musical genres |
| **artist** | Artist's name |
| **song** | Song title |
| **lyrics** | Full song lyrics |
| **genius_link** | Genius.com URL |
| **youtube_link** | YouTube video URL (scraped) |
| **label** | Record label |

## Dependencies

- `requests`
- `beautifulsoup4`
- `pandas`
- `openpyxl`
- `python-dotenv`

---

## Acknowledgment

This project has been partially supported by the Ministerio de Educación Superior, Ciencia y Tecnología (MESCyT) of the Dominican Republic through the FONDOCYT grant. The authors gratefully acknowledge this support.

Any opinions, findings, conclusions, or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of MESCyT.