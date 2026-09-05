"""Public contracts for automatic speech recognition backends."""

from .contracts import ASRBackend, BackendDescriptor, Transcript
from .subprocess_backend import (
    JsonlSubprocessBackend,
    SubprocessBackendSettings,
    WorkerProcessError,
    WorkerRemoteError,
    WorkerTimeoutError,
)
from .whisper import WhisperBackend, WhisperDependencyError, WhisperSettings

__all__ = [
    "ASRBackend",
    "BackendDescriptor",
    "JsonlSubprocessBackend",
    "SubprocessBackendSettings",
    "Transcript",
    "WhisperBackend",
    "WhisperDependencyError",
    "WhisperSettings",
    "WorkerProcessError",
    "WorkerRemoteError",
    "WorkerTimeoutError",
]
