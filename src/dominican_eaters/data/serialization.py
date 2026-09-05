"""Strict JSON conversion shared by versioned domain artifacts."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class ArtifactSerializationError(TypeError):
    """Raised when an artifact contains a value outside the JSON contract."""


def to_json_value(value: object) -> Any:
    """Convert supported immutable contract values into strict JSON values."""

    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: to_json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, tuple | list):
        return [to_json_value(item) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: to_json_value(item) for key, item in value.items()}
    raise ArtifactSerializationError(f"unsupported artifact value: {type(value).__name__}")
