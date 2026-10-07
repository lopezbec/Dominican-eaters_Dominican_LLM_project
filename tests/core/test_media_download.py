from __future__ import annotations

import json
from pathlib import Path

from dominican_eaters.collection.lyrics import (
    CollectionLedger,
    CollectionResult,
    CollectionStatus,
    LyricsManifest,
    LyricsRequest,
    SongRecord,
    VideoMatch,
)
from dominican_eaters.collection.media import (
    DownloadedMedia,
    MediaDownloadError,
    MediaDownloadRunner,
    MediaStatus,
    MediaTask,
)


class FakeDownloader:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[MediaTask] = []
        self.fail = fail

    def preflight(self) -> tuple[tuple[str, bool, str], ...]:
        return (("yt-dlp", True, "/bin/yt-dlp"),)

    def download(
        self,
        task: MediaTask,
        destination: Path,
        *,
        timeout_seconds: float,
        on_log: object = None,
    ) -> DownloadedMedia:
        self.calls.append(task)
        if self.fail:
            raise MediaDownloadError("offline", code="network", retryable=True)
        destination.write_bytes(b"RIFF" + b"audio" * 20)
        return DownloadedMedia(12.5, {"title": "Song"})


def _source_ledger(path: Path) -> None:
    request = LyricsRequest("song-1", "song")
    song = SongRecord(
        source_song_id="1",
        title="Song",
        artist="Artist",
        genius_url="https://genius.com/song",
        lyrics="Hola mundo",
        video=VideoMatch("abcdefghijk", "https://youtube.com/watch?v=abcdefghijk", "Song"),
    )
    ledger = CollectionLedger(
        LyricsManifest((request,)),
        (CollectionResult(request, CollectionStatus.COMPLETE, 1, "2026-01-01T00:00:00Z", song),),
    )
    path.write_text(json.dumps(ledger.to_dict()), encoding="utf-8")


def test_download_writes_stable_audio_reference_and_atomic_ledger(tmp_path: Path) -> None:
    source = tmp_path / "lyrics-collection.json"
    output = tmp_path / "audio"
    _source_ledger(source)
    downloader = FakeDownloader()

    ledger = MediaDownloadRunner(downloader).run(source, output)

    assert ledger.records[0].status is MediaStatus.COMPLETE
    assert (output / "song-1.wav").is_file()
    assert (output / "song-1.txt").read_text(encoding="utf-8") == "Hola mundo\n"
    assert (output / "media-download.json").is_file()
    assert not tuple(output.glob("*.partial.wav"))
    MediaDownloadRunner(downloader).run(source, output)
    assert len(downloader.calls) == 1


def test_retryable_failure_is_checkpointed_and_retried(tmp_path: Path) -> None:
    source = tmp_path / "lyrics-collection.json"
    output = tmp_path / "audio"
    _source_ledger(source)
    failed = MediaDownloadRunner(FakeDownloader(fail=True)).run(source, output)
    assert failed.records[0].status is MediaStatus.FAILED
    assert failed.records[0].error_code == "network"
    assert failed.records[0].retryable is True

    recovered = MediaDownloadRunner(FakeDownloader()).run(source, output)
    assert recovered.records[0].status is MediaStatus.COMPLETE
    assert recovered.records[0].attempt == 2
