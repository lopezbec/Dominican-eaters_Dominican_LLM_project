"""Strict loading for lyrics request manifests."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from dominican_eaters.data import atomic_write_json

from .contracts import LyricsManifest, LyricsValidationError


def _reject_duplicate_keys(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LyricsValidationError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def load_lyrics_manifest(path: str | Path) -> LyricsManifest:
    manifest_path = Path(path).expanduser().resolve()
    try:
        raw = json.loads(
            manifest_path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except LyricsValidationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise LyricsValidationError(f"could not read manifest {manifest_path}: {error}") from error
    return LyricsManifest.from_dict(raw)


def write_lyrics_manifest(manifest: LyricsManifest, path: str | Path) -> None:
    if not isinstance(manifest, LyricsManifest):
        raise TypeError("manifest must be a LyricsManifest")
    atomic_write_json(path, manifest.to_dict())
