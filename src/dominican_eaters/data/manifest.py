"""Strict, versioned manifests for speech-to-text datasets."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256 as sha256_digest
from pathlib import Path, PurePosixPath
from typing import Any, Final, overload

STT_MANIFEST_SCHEMA_VERSION: Final = 1
_SAMPLE_FIELDS: Final = frozenset(
    {"sample_id", "audio_path", "reference_text", "group_id", "split", "source", "sha256"}
)
_REQUIRED_SAMPLE_FIELDS: Final = frozenset({"sample_id", "audio_path", "reference_text"})
_MANIFEST_FIELDS: Final = frozenset({"schema_version", "dataset_root", "samples"})
_SHA256_PATTERN: Final = re.compile(r"[0-9a-f]{64}")


class ManifestValidationError(ValueError):
    """Raised when a manifest does not satisfy the canonical schema."""


class ManifestPreflightError(ManifestValidationError):
    """Raised when files named by a structurally valid manifest are unusable."""


def _require_exact_fields(
    value: Mapping[str, Any], *, expected: frozenset[str], required: frozenset[str], context: str
) -> None:
    unknown = sorted(set(value) - expected)
    missing = sorted(required - set(value))
    if unknown or missing:
        details: list[str] = []
        if missing:
            details.append(f"missing fields: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown fields: {', '.join(unknown)}")
        raise ManifestValidationError(f"{context}: {'; '.join(details)}")


def _required_string(value: Any, *, field: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ManifestValidationError(f"{field} must be a string")
    if not allow_empty and not value.strip():
        raise ManifestValidationError(f"{field} must not be empty")
    return value


def _optional_string(value: Any, *, field: str) -> str | None:
    if value is None:
        return None
    return _required_string(value, field=field)


def _validate_audio_path(value: Any) -> str:
    audio_path = _required_string(value, field="audio_path")
    if "\\" in audio_path:
        raise ManifestValidationError("audio_path must use '/' separators")
    locator = PurePosixPath(audio_path)
    if locator.is_absolute() or audio_path.startswith("/"):
        raise ManifestValidationError("audio_path must be relative to dataset_root")
    if any(part in {"", ".", ".."} for part in locator.parts):
        raise ManifestValidationError("audio_path must stay below dataset_root")
    return locator.as_posix()


@dataclass(frozen=True, slots=True)
class AudioSample:
    """One STT evaluation sample whose identity is independent of its file path."""

    sample_id: str
    audio_path: str
    reference_text: str
    group_id: str | None = None
    split: str | None = None
    source: str | None = None
    sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_id", _required_string(self.sample_id, field="sample_id"))
        object.__setattr__(self, "audio_path", _validate_audio_path(self.audio_path))
        object.__setattr__(
            self,
            "reference_text",
            _required_string(self.reference_text, field="reference_text", allow_empty=True),
        )
        for field in ("group_id", "split", "source"):
            object.__setattr__(self, field, _optional_string(getattr(self, field), field=field))
        if self.sha256 is not None:
            checksum = _required_string(self.sha256, field="sha256").lower()
            if _SHA256_PATTERN.fullmatch(checksum) is None:
                raise ManifestValidationError(
                    "sha256 must contain exactly 64 hexadecimal characters"
                )
            object.__setattr__(self, "sha256", checksum)
        if self.sample_id == self.audio_path:
            raise ManifestValidationError("sample_id must be distinct from audio_path")

    def resolved_audio_path(self, dataset_root: Path) -> Path:
        """Resolve this sample below an explicit absolute dataset root."""

        root = Path(dataset_root)
        if not root.is_absolute():
            raise ManifestValidationError("dataset_root must be absolute when resolving audio")
        resolved = (root / Path(*PurePosixPath(self.audio_path).parts)).resolve()
        try:
            resolved.relative_to(root.resolve())
        except ValueError as error:  # also protects against a dataset-root symlink escape
            raise ManifestValidationError(
                f"audio_path escapes dataset_root: {self.audio_path}"
            ) from error
        return resolved

    @classmethod
    def from_dict(cls, value: Any, *, index: int | None = None) -> AudioSample:
        context = "sample" if index is None else f"samples[{index}]"
        if not isinstance(value, Mapping):
            raise ManifestValidationError(f"{context} must be an object")
        _require_exact_fields(
            value, expected=_SAMPLE_FIELDS, required=_REQUIRED_SAMPLE_FIELDS, context=context
        )
        return cls(
            sample_id=value["sample_id"],
            audio_path=value["audio_path"],
            reference_text=value["reference_text"],
            group_id=value.get("group_id"),
            split=value.get("split"),
            source=value.get("source"),
            sha256=value.get("sha256"),
        )

    def to_dict(self) -> dict[str, str | None]:
        return {
            "sample_id": self.sample_id,
            "audio_path": self.audio_path,
            "reference_text": self.reference_text,
            "group_id": self.group_id,
            "split": self.split,
            "source": self.source,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class STTManifest(Sequence[AudioSample]):
    """A frozen STT input set rooted at an explicit filesystem location."""

    dataset_root: Path
    samples: tuple[AudioSample, ...]
    schema_version: int = STT_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != STT_MANIFEST_SCHEMA_VERSION
        ):
            raise ManifestValidationError(
                f"schema_version must be {STT_MANIFEST_SCHEMA_VERSION}, got {self.schema_version!r}"
            )
        root = Path(self.dataset_root).expanduser()
        if not root.is_absolute():
            raise ManifestValidationError("dataset_root must be absolute in a loaded STTManifest")
        object.__setattr__(self, "dataset_root", root.resolve())
        object.__setattr__(self, "samples", tuple(self.samples))
        if not self.samples:
            raise ManifestValidationError("samples must not be empty")
        if not all(isinstance(sample, AudioSample) for sample in self.samples):
            raise ManifestValidationError("samples must contain only AudioSample values")
        ids = [sample.sample_id for sample in self.samples]
        duplicates = sorted({sample_id for sample_id in ids if ids.count(sample_id) > 1})
        if duplicates:
            raise ManifestValidationError(f"duplicate sample_id values: {', '.join(duplicates)}")

    @overload
    def __getitem__(self, index: int) -> AudioSample: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[AudioSample, ...]: ...

    def __getitem__(self, index: int | slice) -> AudioSample | tuple[AudioSample, ...]:
        return self.samples[index]

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[AudioSample]:
        return iter(self.samples)

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(sample.sample_id for sample in self.samples)

    def resolve_audio(self, sample: AudioSample | str) -> Path:
        if isinstance(sample, str):
            try:
                sample = next(item for item in self.samples if item.sample_id == sample)
            except StopIteration as error:
                raise KeyError(sample) from error
        return sample.resolved_audio_path(self.dataset_root)

    def preflight(self, *, verify_hashes: bool = False) -> None:
        errors: list[str] = []
        for sample in self.samples:
            path = self.resolve_audio(sample)
            if not path.is_file():
                errors.append(f"{sample.sample_id}: audio file does not exist: {path}")
                continue
            if verify_hashes:
                if sample.sha256 is None:
                    errors.append(f"{sample.sample_id}: sha256 is required for verified preflight")
                elif _sha256_file(path) != sample.sha256:
                    errors.append(f"{sample.sample_id}: sha256 mismatch: {path}")
        if errors:
            raise ManifestPreflightError("manifest preflight failed:\n" + "\n".join(errors))

    def with_dataset_root(self, dataset_root: Path, *, verify_hashes: bool = True) -> STTManifest:
        """Explicitly remap the dataset, verifying declared hashes by default."""

        remapped = STTManifest(
            dataset_root=Path(dataset_root).expanduser().resolve(), samples=self.samples
        )
        remapped.preflight(verify_hashes=verify_hashes)
        return remapped

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_root": str(self.dataset_root),
            "samples": [sample.to_dict() for sample in self.samples],
        }


def _sha256_file(path: Path) -> str:
    digest = sha256_digest()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(
    path: str | os.PathLike[str],
    *,
    dataset_root_override: str | os.PathLike[str] | None = None,
    verify_hashes_on_override: bool = True,
) -> STTManifest:
    """Load the one canonical STT manifest schema.

    Relative ``dataset_root`` values are resolved against the input manifest's
    directory. A root override is an explicit data move and is preflighted.
    """

    manifest_path = Path(path).expanduser().resolve()
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestValidationError(
            f"could not read manifest {manifest_path}: {error}"
        ) from error
    if not isinstance(raw, Mapping):
        raise ManifestValidationError("manifest must be a JSON object")
    _require_exact_fields(
        raw, expected=_MANIFEST_FIELDS, required=_MANIFEST_FIELDS, context="manifest"
    )
    version = raw["schema_version"]
    if type(version) is not int or version != STT_MANIFEST_SCHEMA_VERSION:
        raise ManifestValidationError(
            f"schema_version must be {STT_MANIFEST_SCHEMA_VERSION}, got {version!r}"
        )
    root_value = _required_string(raw["dataset_root"], field="dataset_root")
    declared_root = Path(root_value).expanduser()
    if not declared_root.is_absolute():
        declared_root = manifest_path.parent / declared_root
    samples_raw = raw["samples"]
    if not isinstance(samples_raw, list):
        raise ManifestValidationError("samples must be an array")
    manifest = STTManifest(
        schema_version=version,
        dataset_root=declared_root.resolve(),
        samples=tuple(
            AudioSample.from_dict(item, index=index) for index, item in enumerate(samples_raw)
        ),
    )
    if dataset_root_override is not None:
        return manifest.with_dataset_root(
            Path(dataset_root_override), verify_hashes=verify_hashes_on_override
        )
    return manifest


def atomic_write_json(path: str | os.PathLike[str], value: Mapping[str, Any]) -> None:
    """Durably replace a JSON file without exposing a partial write."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(
                value,
                stream,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def write_manifest(manifest: STTManifest, path: str | os.PathLike[str]) -> None:
    """Atomically write a replayable frozen manifest.

    ``STTManifest`` always holds an absolute root, so copying the written file to
    another run directory on the same machine does not reinterpret audio paths.
    """

    if not isinstance(manifest, STTManifest):
        raise TypeError("manifest must be an STTManifest")
    atomic_write_json(path, manifest.to_dict())
