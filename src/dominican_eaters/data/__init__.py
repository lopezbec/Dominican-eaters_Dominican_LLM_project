"""Versioned data contracts for Dominican Eaters."""

from .locking import ConcurrentWriteError, exclusive_file_lock
from .manifest import (
    STT_MANIFEST_SCHEMA_VERSION,
    AudioSample,
    ManifestPreflightError,
    ManifestValidationError,
    STTManifest,
    atomic_write_json,
    load_manifest,
    write_manifest,
)
from .serialization import ArtifactSerializationError, to_json_value

__all__ = [
    "STT_MANIFEST_SCHEMA_VERSION",
    "AudioSample",
    "ArtifactSerializationError",
    "ConcurrentWriteError",
    "ManifestPreflightError",
    "ManifestValidationError",
    "STTManifest",
    "atomic_write_json",
    "exclusive_file_lock",
    "load_manifest",
    "to_json_value",
    "write_manifest",
]
