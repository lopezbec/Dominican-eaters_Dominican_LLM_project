"""Typed contracts for restartable media acquisition."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

MEDIA_SCHEMA_VERSION = 1


class MediaStatus(StrEnum):
    COMPLETE = "complete"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class MediaTask:
    request_id: str
    video_id: str
    source_url: str
    lyrics: str


@dataclass(frozen=True, slots=True)
class DownloadedMedia:
    duration_seconds: float
    metadata: dict[str, object]


class MediaDownloader(Protocol):
    def preflight(self) -> tuple[tuple[str, bool, str], ...]: ...

    def download(
        self,
        task: MediaTask,
        destination: Path,
        *,
        timeout_seconds: float,
        on_log: LogSink | None = None,
    ) -> DownloadedMedia: ...


class LogSink(Protocol):
    def __call__(self, line: str) -> None: ...


class MediaDownloadError(RuntimeError):
    def __init__(self, message: str, *, code: str, retryable: bool) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


@dataclass(frozen=True, slots=True)
class MediaRecord:
    request_id: str
    video_id: str
    source_url: str
    status: MediaStatus
    attempt: int
    audio_path: str | None = None
    reference_path: str | None = None
    sha256: str | None = None
    bytes: int | None = None
    duration_seconds: float | None = None
    error_code: str | None = None
    error_message: str | None = None
    retryable: bool = False
    metadata: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "video_id": self.video_id,
            "source_url": self.source_url,
            "status": self.status.value,
            "attempt": self.attempt,
            "audio_path": self.audio_path,
            "reference_path": self.reference_path,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "duration_seconds": self.duration_seconds,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "retryable": self.retryable,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> MediaRecord:
        attempt = value.get("attempt")
        if not isinstance(attempt, int) or isinstance(attempt, bool):
            raise ValueError("media record attempt must be an integer")
        raw_metadata = value.get("metadata")
        metadata: dict[str, object] | None = None
        if isinstance(raw_metadata, dict) and all(isinstance(key, str) for key in raw_metadata):
            metadata = {str(key): item for key, item in raw_metadata.items()}
        return cls(
            request_id=str(value["request_id"]),
            video_id=str(value["video_id"]),
            source_url=str(value["source_url"]),
            status=MediaStatus(str(value["status"])),
            attempt=attempt,
            audio_path=_optional_string(value.get("audio_path")),
            reference_path=_optional_string(value.get("reference_path")),
            sha256=_optional_string(value.get("sha256")),
            bytes=_optional_int(value.get("bytes")),
            duration_seconds=_optional_float(value.get("duration_seconds")),
            error_code=_optional_string(value.get("error_code")),
            error_message=_optional_string(value.get("error_message")),
            retryable=value.get("retryable") is True,
            metadata=metadata,
        )


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _optional_float(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None
