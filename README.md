# Dominican Eaters - Dominican LLM Project

> A collection of web scrapers for Dominican cultural content: books, poems, and music lyrics.

## Overview

This repository contains three specialized scrapers designed to collect and catalog Dominican cultural content from various online sources. Each scraper is maintained as an independent Git submodule.

## Projects

### 📚 [books-eater](./books-eater)
Scraper for Dominican audiobooks available on YouTube.
- Searches for audiobooks of Dominican literature
- Exports metadata to Excel/CSV
- Uses `scrapetube` (no API key required)

### 📝 [poems-eater](./poems-eater)
Scraper for Dominican poem recitations on YouTube.
- Finds recitations, dramatizations, and readings
- Includes 100+ classic and contemporary Dominican poems
- Detailed statistics on authors, genres, and content types

### 🎵 [lyrics-eater](./lyrics-eater)
Scraper for Dominican song lyrics.
- Integrates with Genius API and YouTube
- Catalogs Dominican music and lyrics
- Exports comprehensive song datasets

## Structure

```
Dominican-eaters_Dominican_LLM_project/
├── books-eater/          # Submodule: Audiobook scraper
├── poems-eater/          # Submodule: Poem recitation scraper
├── lyrics-eater/         # Submodule: Song lyrics scraper
└── README.md             # This file
```

## Getting Started

### Clone with Submodules

To clone this repository with all submodules:

```bash
git clone --recurse-submodules git@github.com:lopezbec/Dominican-eaters_Dominican_LLM_project.git
```

Or if you already cloned without submodules:

```bash
git clone git@github.com:lopezbec/Dominican-eaters_Dominican_LLM_project.git
cd Dominican-eaters_Dominican_LLM_project
git submodule init
git submodule update
```

### Install Dependencies

Each project has its own dependencies. Navigate to each submodule and install:

```bash
# Books Eater
cd books-eater
pip install -r requirements.txt

# Poems Eater
cd ../poems-eater
pip install -r requirements.txt

# Lyrics Eater
cd ../lyrics-eater
pip install -r requirements.txt
```

## Usage

Each submodule can be run independently:

```bash
# Run books scraper
cd books-eater
python main.py

# Run poems scraper
cd poems-eater
python main.py

# Run lyrics scraper
cd lyrics-eater
python main.py
```

## Working with Submodules

### Update All Submodules

```bash
git submodule update --remote --merge
```

### Update a Specific Submodule

```bash
cd books-eater
git pull origin main
cd ..
git add books-eater
git commit -m "Update books-eater submodule"
```

### Make Changes in a Submodule

1. Navigate to the submodule directory
2. Make your changes
3. Commit in the submodule
4. Push the submodule changes
5. Update the parent repository

```bash
cd books-eater
# Make changes...
git add .
git commit -m "Your changes"
git push origin main

cd ..
git add books-eater
git commit -m "Update books-eater reference"
git push origin main
```

## Technology Stack

- **Python 3.8+**
- **scrapetube**: YouTube scraping without API key
- **pandas**: Data manipulation and export
- **openpyxl**: Excel file generation
- **Genius API**: Lyrics fetching (lyrics-eater)

## Project Goals

This collection of scrapers aims to:

1. **Preserve Dominican Culture**: Digitally catalog Dominican literature, poetry, and music
2. **Enable Research**: Provide datasets for cultural and linguistic analysis
3. **Support Education**: Make Dominican cultural content more accessible
4. **Build LLM Training Data**: Create high-quality datasets for Dominican Spanish language models

## Use Cases

- **Academic Research**: Study Dominican literature and cultural trends
- **Natural Language Processing**: Train language models on Dominican Spanish
- **Cultural Preservation**: Archive digital copies of cultural content
- **Educational Resources**: Provide materials for teaching Dominican culture
- **Content Discovery**: Help people find Dominican cultural content online

## Contributing

Each submodule has its own contribution guidelines. To contribute:

1. Fork the specific submodule repository
2. Make your changes
3. Submit a pull request to that submodule
4. Update this parent repository if needed

## Notes

- All scrapers respect copyright and only collect publicly available content
- Scrapers are designed for educational and research purposes
- Content availability may change over time
- Always verify rights before using collected content

## License

Each submodule has its own license. Please refer to individual project READMEs.

## Authors

- Christian Lopez - Project Lead
- Contributors welcome!

## Contact

For questions or collaborations regarding this project, please open an issue in the relevant repository.

## Acknowledgment

This project has been partially supported by the Ministerio de Educación Superior, Ciencia y Tecnología (MESCyT) of the Dominican Republic through the FONDOCYT grant. The authors gratefully acknowledge this support.

Any opinions, findings, conclusions, or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of MESCyT.

---

**Preserving Dominican culture in the digital age** 🇩🇴
