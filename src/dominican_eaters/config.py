"""Canonical application configuration with explicit filesystem roots."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import yaml

CONFIG_SCHEMA_VERSION: Final = 1
_CONFIG_FIELDS: Final = frozenset({"schema_version", "data_root", "artifacts_root"})


class ConfigError(ValueError):
    """Raised when canonical application configuration is invalid."""


def _path_value(value: Any, *, field: str, base: Path) -> Path:
    if not isinstance(value, (str, os.PathLike)) or not str(value).strip():
        raise ConfigError(f"{field} must be a non-empty filesystem path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Dependency-light configuration shared by application entry points."""

    data_root: Path
    artifacts_root: Path
    schema_version: int = CONFIG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != CONFIG_SCHEMA_VERSION:
            raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")
        for field in ("data_root", "artifacts_root"):
            path = Path(getattr(self, field)).expanduser()
            if not path.is_absolute():
                raise ConfigError(f"{field} must be absolute in a loaded AppConfig")
            object.__setattr__(self, field, path.resolve())
        if self.data_root == self.artifacts_root:
            raise ConfigError("data_root and artifacts_root must be different directories")

    def dataset_root(self, name: str) -> Path:
        """Return a named dataset directory below the canonical data root."""

        if not isinstance(name, str) or not name.strip():
            raise ConfigError("dataset name must be a non-empty string")
        relative = Path(name)
        if relative.is_absolute() or len(relative.parts) != 1 or relative.name in {".", ".."}:
            raise ConfigError("dataset name must be one path component")
        return self.data_root / relative


def load_config(
    path: str | os.PathLike[str],
    *,
    data_root: str | os.PathLike[str] | None = None,
    artifacts_root: str | os.PathLike[str] | None = None,
) -> AppConfig:
    """Load the strict YAML config, applying explicit root overrides last.

    Paths selected in YAML are relative to that YAML file. Explicit overrides
    are relative to the caller's current working directory. No checkout or
    package location is inferred.
    """

    config_path = Path(path).expanduser().resolve()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ConfigError(f"could not read config {config_path}: {error}") from error
    if not isinstance(raw, Mapping):
        raise ConfigError("config must be a YAML mapping")
    unknown = sorted(set(raw) - _CONFIG_FIELDS)
    missing = sorted(_CONFIG_FIELDS - set(raw))
    if unknown or missing:
        details: list[str] = []
        if missing:
            details.append(f"missing fields: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown fields: {', '.join(unknown)}")
        raise ConfigError("invalid config: " + "; ".join(details))
    version = raw["schema_version"]
    if type(version) is not int or version != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION}, got {version!r}")
    selected_data_root = _path_value(raw["data_root"], field="data_root", base=config_path.parent)
    selected_artifacts_root = _path_value(
        raw["artifacts_root"], field="artifacts_root", base=config_path.parent
    )
    working_directory = Path.cwd()
    if data_root is not None:
        selected_data_root = _path_value(data_root, field="data_root", base=working_directory)
    if artifacts_root is not None:
        selected_artifacts_root = _path_value(
            artifacts_root, field="artifacts_root", base=working_directory
        )
    return AppConfig(
        schema_version=version,
        data_root=selected_data_root,
        artifacts_root=selected_artifacts_root,
    )
