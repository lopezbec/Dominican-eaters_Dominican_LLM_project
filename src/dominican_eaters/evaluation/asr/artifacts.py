"""Atomic JSON artifacts for canonical ASR evaluation runs."""

from __future__ import annotations

from pathlib import Path

from dominican_eaters.data import ArtifactSerializationError, atomic_write_json, to_json_value


def write_artifact(path: Path, value: object) -> None:
    payload = to_json_value(value)
    if not isinstance(payload, dict):
        raise ArtifactSerializationError("artifact root must be an object")
    atomic_write_json(path, payload)
