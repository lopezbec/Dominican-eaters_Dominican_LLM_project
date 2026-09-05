"""Strict source manifest for poem collection."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, overload

from dominican_eaters.data import atomic_write_json

from .models import PoemSource

POEM_MANIFEST_SCHEMA_VERSION: Final = 1
_MANIFEST_FIELDS = frozenset({"schema_version", "poems"})
_SOURCE_FIELDS = frozenset(
    {"source_id", "title", "author", "publication_year", "genre", "reference_text", "provenance"}
)


@dataclass(frozen=True, slots=True)
class PoemManifest(Sequence[PoemSource]):
    poems: tuple[PoemSource, ...]
    schema_version: int = POEM_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported poem manifest schema_version")
        object.__setattr__(self, "poems", tuple(self.poems))
        if not self.poems:
            raise ValueError("poem manifest must not be empty")
        if not all(isinstance(poem, PoemSource) for poem in self.poems):
            raise ValueError("poems must contain only PoemSource values")
        ids = [poem.source_id for poem in self.poems]
        if len(ids) != len(set(ids)):
            raise ValueError("poem manifest contains duplicate source IDs")

    @overload
    def __getitem__(self, index: int) -> PoemSource: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[PoemSource, ...]: ...

    def __getitem__(self, index: int | slice) -> PoemSource | tuple[PoemSource, ...]:
        return self.poems[index]

    def __len__(self) -> int:
        return len(self.poems)

    def __iter__(self) -> Iterator[PoemSource]:
        return iter(self.poems)

    def to_dict(self) -> dict[str, object]:
        return {"schema_version": self.schema_version, "poems": [asdict(poem) for poem in self]}


def load_poem_manifest(path: Path) -> PoemManifest:
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read poem manifest: {exc}") from exc
    if not isinstance(raw, Mapping) or set(raw) != _MANIFEST_FIELDS:
        raise ValueError("poem manifest has an invalid top-level schema")
    version = raw["schema_version"]
    if type(version) is not int:
        raise ValueError("schema_version must be an integer")
    poems_raw = raw["poems"]
    if not isinstance(poems_raw, list):
        raise ValueError("poems must be an array")
    poems: list[PoemSource] = []
    for index, value in enumerate(poems_raw):
        if not isinstance(value, Mapping) or set(value) != _SOURCE_FIELDS:
            raise ValueError(f"poems[{index}] has an invalid schema")
        poems.append(
            PoemSource(
                source_id=value["source_id"],
                title=value["title"],
                author=value["author"],
                publication_year=value["publication_year"],
                genre=value["genre"],
                reference_text=value["reference_text"],
                provenance=value["provenance"],
            )
        )
    return PoemManifest(tuple(poems), schema_version=version)


def write_poem_manifest(manifest: PoemManifest, path: Path) -> None:
    if not isinstance(manifest, PoemManifest):
        raise TypeError("manifest must be a PoemManifest")
    atomic_write_json(path, manifest.to_dict())
