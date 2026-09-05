"""Advisory single-writer locks for local atomic artifact workflows."""

from __future__ import annotations

import fcntl
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class ConcurrentWriteError(RuntimeError):
    """Raised when another process owns an artifact's writer lock."""


@contextmanager
def exclusive_file_lock(path: Path, *, label: str = "artifact") -> Iterator[None]:
    """Hold a non-blocking advisory lock beside ``path`` for one mutation run."""

    destination = path.expanduser().resolve()
    lock_path = destination.with_suffix(destination.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ConcurrentWriteError(f"another process owns the {label}: {destination}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
