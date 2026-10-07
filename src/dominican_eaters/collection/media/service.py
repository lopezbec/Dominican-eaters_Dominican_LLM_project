"""Restartable conversion from collected lyrics records to an audio dataset."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from dominican_eaters.collection.lyrics import CollectionLedger
from dominican_eaters.data import atomic_write_json, exclusive_file_lock

from .contracts import (
    MEDIA_SCHEMA_VERSION,
    LogSink,
    MediaDownloader,
    MediaDownloadError,
    MediaRecord,
    MediaStatus,
    MediaTask,
)

MEDIA_LEDGER_FILENAME = "media-download.json"
ProgressCallback = Callable[[int, int, MediaRecord], None]


@dataclass(frozen=True, slots=True)
class MediaLedger:
    source_ledger: str
    records: tuple[MediaRecord, ...] = ()
    schema_version: int = MEDIA_SCHEMA_VERSION

    @property
    def by_request_id(self) -> dict[str, MediaRecord]:
        return {record.request_id: record for record in self.records}

    def with_record(self, record: MediaRecord, order: tuple[str, ...]) -> MediaLedger:
        values = self.by_request_id
        values[record.request_id] = record
        return MediaLedger(self.source_ledger, tuple(values[key] for key in order if key in values))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source_ledger": self.source_ledger,
            "records": [record.to_dict() for record in self.records],
        }

    @classmethod
    def from_dict(cls, value: object) -> MediaLedger:
        if not isinstance(value, dict) or set(value) != {
            "schema_version",
            "source_ledger",
            "records",
        }:
            raise ValueError("invalid media download ledger")
        if value["schema_version"] != MEDIA_SCHEMA_VERSION:
            raise ValueError("unsupported media download ledger schema_version")
        records = value["records"]
        if not isinstance(records, list) or not isinstance(value["source_ledger"], str):
            raise ValueError("invalid media download ledger fields")
        parsed = tuple(
            MediaRecord.from_dict(record) for record in records if isinstance(record, dict)
        )
        if len(parsed) != len(records):
            raise ValueError("invalid media download record")
        return cls(value["source_ledger"], parsed)


class MediaDownloadRunner:
    def __init__(
        self,
        downloader: MediaDownloader,
        *,
        timeout_seconds: float = 900,
        on_progress: ProgressCallback | None = None,
        on_log: LogSink | None = None,
    ) -> None:
        self._downloader = downloader
        self._timeout_seconds = timeout_seconds
        self._on_progress = on_progress
        self._on_log = on_log

    def preflight(self) -> tuple[tuple[str, bool, str], ...]:
        return self._downloader.preflight()

    def run(
        self,
        collection_ledger: Path,
        output_dir: Path,
        *,
        force: bool = False,
    ) -> MediaLedger:
        source = collection_ledger.expanduser().resolve()
        collection = _load_collection_ledger(source)
        tasks = _media_tasks(collection)
        if not tasks:
            raise ValueError("lyrics collection ledger contains no records with video and lyrics")
        root = output_dir.expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        ledger_path = root / MEDIA_LEDGER_FILENAME
        with exclusive_file_lock(ledger_path, label="media download ledger"):
            ledger = self._load_or_initialize(ledger_path, source)
            order = tuple(task.request_id for task in tasks)
            for index, task in enumerate(tasks, start=1):
                previous = ledger.by_request_id.get(task.request_id)
                if not force and previous is not None:
                    if previous.status is MediaStatus.COMPLETE and _record_files_valid(
                        root, previous
                    ):
                        continue
                    if previous.status is MediaStatus.FAILED and not previous.retryable:
                        continue
                attempt = 1 if previous is None else previous.attempt + 1
                record = self._download_one(task, root, attempt)
                ledger = ledger.with_record(record, order)
                atomic_write_json(ledger_path, ledger.to_dict())
                if self._on_progress is not None:
                    self._on_progress(index, len(tasks), record)
            return ledger

    @staticmethod
    def _load_or_initialize(path: Path, source: Path) -> MediaLedger:
        if not path.exists():
            ledger = MediaLedger(str(source))
            atomic_write_json(path, ledger.to_dict())
            return ledger
        try:
            ledger = MediaLedger.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"could not read media download ledger {path}: {error}") from error
        if Path(ledger.source_ledger) != source:
            raise ValueError("media download ledger belongs to a different collection ledger")
        return ledger

    def _download_one(self, task: MediaTask, root: Path, attempt: int) -> MediaRecord:
        audio = root / f"{task.request_id}.wav"
        reference = root / f"{task.request_id}.txt"
        temporary = root / f".{task.request_id}.{os.getpid()}.partial.wav"
        temporary.unlink(missing_ok=True)
        try:
            downloaded = self._downloader.download(
                task,
                temporary,
                timeout_seconds=self._timeout_seconds,
                on_log=self._on_log,
            )
            if not temporary.is_file():
                raise MediaDownloadError(
                    "downloader did not create normalized audio",
                    code="missing_output",
                    retryable=True,
                )
            os.replace(temporary, audio)
            _atomic_write_text(reference, task.lyrics.strip() + "\n")
            return MediaRecord(
                request_id=task.request_id,
                video_id=task.video_id,
                source_url=task.source_url,
                status=MediaStatus.COMPLETE,
                attempt=attempt,
                audio_path=audio.name,
                reference_path=reference.name,
                sha256=_sha256(audio),
                bytes=audio.stat().st_size,
                duration_seconds=downloaded.duration_seconds,
                metadata=downloaded.metadata,
            )
        except MediaDownloadError as error:
            temporary.unlink(missing_ok=True)
            return MediaRecord(
                request_id=task.request_id,
                video_id=task.video_id,
                source_url=task.source_url,
                status=MediaStatus.FAILED,
                attempt=attempt,
                error_code=error.code,
                error_message=str(error),
                retryable=error.retryable,
            )
        except OSError as error:
            temporary.unlink(missing_ok=True)
            return MediaRecord(
                request_id=task.request_id,
                video_id=task.video_id,
                source_url=task.source_url,
                status=MediaStatus.FAILED,
                attempt=attempt,
                error_code="filesystem",
                error_message=str(error),
                retryable=True,
            )


def _load_collection_ledger(path: Path) -> CollectionLedger:
    try:
        return CollectionLedger.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read lyrics collection ledger {path}: {error}") from error


def _media_tasks(ledger: CollectionLedger) -> tuple[MediaTask, ...]:
    tasks: list[MediaTask] = []
    for result in ledger.results:
        song = result.song
        if song is None or song.video is None or song.lyrics is None:
            continue
        tasks.append(
            MediaTask(
                request_id=result.request.request_id,
                video_id=song.video.video_id,
                source_url=song.video.url,
                lyrics=song.lyrics,
            )
        )
    return tuple(tasks)


def _record_files_valid(root: Path, record: MediaRecord) -> bool:
    if record.audio_path is None or record.reference_path is None or record.sha256 is None:
        return False
    audio = root / record.audio_path
    reference = root / record.reference_path
    return audio.is_file() and reference.is_file() and _sha256(audio) == record.sha256


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
