from __future__ import annotations

import json
import sys
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
    YtDlpMediaDownloader,
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


def test_ytdlp_uses_selected_python_and_audio_capable_format(
    tmp_path: Path, monkeypatch: object
) -> None:
    downloader = YtDlpMediaDownloader()
    commands: list[list[str]] = []

    def fake_run_json(command: list[str], timeout: float, on_log: object) -> dict[str, object]:
        commands.append(command)
        if "ffprobe" in command[0]:
            return {"format": {"duration": "1.0"}}
        Path(command[command.index("-o") + 1].replace("%(ext)s", "webm")).write_bytes(b"audio")
        return {"id": "abcdefghijk", "title": "Song"}

    def fake_run(command: list[str], timeout: float, on_log: object) -> str:
        commands.append(command)
        Path(command[-1]).write_bytes(b"RIFF" + b"audio" * 20)
        return ""

    monkeypatch.setattr(downloader, "_run_json", fake_run_json)  # type: ignore[attr-defined]
    monkeypatch.setattr(downloader, "_run", fake_run)  # type: ignore[attr-defined]
    task = MediaTask("song-1", "abcdefghijk", "https://youtu.be/abcdefghijk", "lyrics")

    downloader.download(task, tmp_path / "song.wav", timeout_seconds=30)

    download_command = commands[0]
    assert download_command[:3] == [sys.executable, "-m", "yt_dlp"]
    assert download_command[download_command.index("-f") + 1] == "bestaudio*/best"
