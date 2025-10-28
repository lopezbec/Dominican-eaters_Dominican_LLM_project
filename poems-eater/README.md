# Poems Eater - Dominican Poem Recitations

An automated tool to find and catalog YouTube videos of Dominican poetry recitations. It uses keyless web scraping to find performances, dramatizations, and readings.

## Features

- **No API Limits**: Uses `scrapetube` for scraping without a YouTube API key.
- **Comprehensive Dataset**: Includes 100+ poems from classic and contemporary Dominican authors.
- **Smart Search**: Finds recitations, dramatizations, and readings.
- **Multiple Export Formats**: Exports results to Excel and CSV.
- **Customizable Search**: Configure the number of videos to analyze per poem.
- **Detailed Statistics**: Provides success rates, popular genres, and most-found authors.

## Project Structure

```
poems-eater/
├── src/
│   ├── clients/
│   ├── models/
│   ├── services/
│   └── utils/
├── main.py
├── poems_list.txt
├── requirements.txt
└── README.md
```

## Setup

1.  **Navigate to the directory**
    ```bash
    cd poems-eater
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Basic Mode (Default Dataset)

```bash
python main.py
```
This will search for the 100+ poems included in the built-in dataset.

### Custom File Mode

Create a `poems_list.txt` file with the format `Title | Author | Year | Genre`:

```
A la Patria | Salomé Ureña | 1874 | Patriotic
Hay un País en el Mundo | Pedro Mir | 1949 | Epic
```

The program will automatically use this file if it exists.

## Output

The script generates `dominican_poems.xlsx` and `dominican_poems.csv` files with detailed information for each poem found.

## Dependencies

- `scrapetube`
- `pandas`
- `openpyxl`

---

## Acknowledgment

This project has been partially supported by the Ministerio de Educación Superior, Ciencia y Tecnología (MESCyT) of the Dominican Republic through the FONDOCYT grant. The authors gratefully acknowledge this support.

Any opinions, findings, conclusions, or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of MESCyT.