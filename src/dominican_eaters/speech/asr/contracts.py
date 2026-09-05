"""Small, runtime-independent contracts for ASR implementations."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True, slots=True)
class BackendDescriptor:
    """Run-level model identity, configuration, and effective runtime details."""

    backend_id: str
    model: str
    model_revision: str | None
    language: str
    requested_device: str
    requested_precision: str
    effective_device: str | None = None
    effective_precision: str | None = None
    runtime_versions: dict[str, str] = field(default_factory=dict)
    options: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("backend_id", "model", "language", "requested_device", "requested_precision"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class Transcript:
    """Text returned by a backend for one audio file."""

    text: str
    language: str | None = None
    audio_duration_seconds: float | None = None
    gpu_peak_allocated_bytes: int | None = None
    gpu_peak_reserved_bytes: int | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("Transcript text must be a string")
        if self.audio_duration_seconds is not None and (
            not math.isfinite(self.audio_duration_seconds) or self.audio_duration_seconds <= 0
        ):
            raise ValueError("audio_duration_seconds must be positive and finite")
        for name in ("gpu_peak_allocated_bytes", "gpu_peak_reserved_bytes"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be a nonnegative integer")


class ASRBackend(Protocol):
    """Lifecycle owned by the caller, model internals owned by the backend.

    ``close`` must tolerate a partially completed or failed ``load``. This lets
    the runner reliably ask a backend to release resources after any lifecycle
    failure.
    """

    @property
    def backend_id(self) -> str:
        """Stable identifier for the configured backend and model."""

    @property
    def descriptor(self) -> BackendDescriptor:
        """Configuration and effective runtime provenance for the current run."""

    def load(self) -> None:
        """Acquire model resources."""

    def warmup(self) -> None:
        """Perform one unmeasured backend-specific warmup."""

    def transcribe(self, audio_path: Path) -> Transcript:
        """Transcribe a resolved local audio path."""

    def close(self) -> None:
        """Release resources, including resources from a partial load."""
