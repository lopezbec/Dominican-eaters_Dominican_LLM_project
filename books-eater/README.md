# Books Eater - Dominican Audiobooks

A specialized tool to find and catalog audiobooks of Dominican literature available on YouTube.

## What is Books Eater?

Books Eater is a Python application that searches for audiobooks by Dominican authors on YouTube, collects information about each video, and generates a complete dataset in Excel format.

## Features

- **Automatic search** for audiobooks on YouTube.
- **Built-in catalog** of classic Dominican literature.
- **Video metadata extraction**: duration, URL, content type.
- **Export to Excel/CSV** with comprehensive statistics.
- **Keyless scraping** - no search limits.

## Project Structure

```
books-eater/
├── src/
│   ├── clients/
│   ├── models/
│   ├── services/
│   └── utils/
├── main.py
├── books_list.txt
├── requirements.txt
└── README.md
```

## Setup

1.  **Navigate to the directory**
    ```bash
    cd workspace/books-eater
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Basic Execution
```bash
python main.py
```

### Search Options

The program includes a predefined dataset of Dominican literature. You can also provide your own list by creating a `books_list.txt` file with the format: `Title | Author | Year`.

### Output

The script generates a `dominican_audiobooks.xlsx` file with details like Title, Author, Year, YouTube URL, Duration, and Availability.

## Dependencies

- `scrapetube`
- `pandas`
- `openpyxl`
- `python-dotenv`

---

## Acknowledgment

This project has been partially supported by the Ministerio de Educación Superior, Ciencia y Tecnología (MESCyT) of the Dominican Republic through the FONDOCYT grant. The authors gratefully acknowledge this support.

Any opinions, findings, conclusions, or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of MESCyT.