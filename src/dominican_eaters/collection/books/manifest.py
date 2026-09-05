"""Strict versioned source catalogs for audiobook collection."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dominican_eaters.data import atomic_write_json

from .models import BookSeed

BOOK_MANIFEST_SCHEMA_VERSION = 1
_TOP_FIELDS = frozenset({"schema_version", "books"})
_BOOK_FIELDS = frozenset({"book_id", "title", "author", "publication_year", "source"})


@dataclass(frozen=True, slots=True)
class BookManifest:
    books: tuple[BookSeed, ...]
    schema_version: int = BOOK_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported book manifest schema_version")
        object.__setattr__(self, "books", tuple(self.books))
        if not self.books:
            raise ValueError("book manifest must not be empty")
        if not all(isinstance(book, BookSeed) for book in self.books):
            raise ValueError("books must contain only BookSeed values")
        ids = [book.book_id for book in self.books]
        duplicates = sorted({book_id for book_id in ids if ids.count(book_id) > 1})
        if duplicates:
            raise ValueError(f"duplicate book IDs: {', '.join(duplicates)}")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "books": [
                {
                    "book_id": book.book_id,
                    "title": book.title,
                    "author": book.author,
                    "publication_year": book.publication_year,
                    "source": book.source,
                }
                for book in self.books
            ],
        }


def load_book_manifest(path: Path) -> BookManifest:
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read book manifest: {exc}") from exc
    if not isinstance(raw, Mapping) or set(raw) != _TOP_FIELDS:
        raise ValueError("book manifest has an invalid top-level schema")
    books_raw = raw["books"]
    if not isinstance(books_raw, list):
        raise ValueError("books must be an array")
    books: list[BookSeed] = []
    for index, item in enumerate(books_raw):
        if not isinstance(item, Mapping) or set(item) != _BOOK_FIELDS:
            raise ValueError(f"books[{index}] has an invalid schema")
        for field in ("book_id", "title", "author"):
            if not isinstance(item[field], str):
                raise ValueError(f"books[{index}].{field} must be a string")
        year = item["publication_year"]
        if year is not None and (isinstance(year, bool) or not isinstance(year, int)):
            raise ValueError(f"books[{index}].publication_year must be an integer or null")
        source = item["source"]
        if source is not None and not isinstance(source, str):
            raise ValueError(f"books[{index}].source must be a string or null")
        books.append(
            BookSeed(
                book_id=str(item["book_id"]),
                title=str(item["title"]),
                author=str(item["author"]),
                publication_year=year,
                source=source,
            )
        )
    version = raw["schema_version"]
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError("schema_version must be an integer")
    return BookManifest(tuple(books), schema_version=version)


def write_book_manifest(manifest: BookManifest, path: Path) -> None:
    if not isinstance(manifest, BookManifest):
        raise TypeError("manifest must be a BookManifest")
    atomic_write_json(path, manifest.to_dict())
