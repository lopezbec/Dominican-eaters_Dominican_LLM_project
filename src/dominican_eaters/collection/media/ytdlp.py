"""yt-dlp adapter with explicit FFmpeg normalization and validation."""

from __future__ import annotations

import json
import selectors
import shutil
import subprocess
import sys
import tempfile
import time
from importlib import metadata, util
from pathlib import Path

from .contracts import DownloadedMedia, LogSink, MediaDownloadError, MediaTask


class YtDlpMediaDownloader:
    def preflight(self) -> tuple[tuple[str, bool, str], ...]:
        yt_dlp_available = util.find_spec("yt_dlp") is not None
        try:
            yt_dlp_version = metadata.version("yt-dlp")
        except metadata.PackageNotFoundError:
            yt_dlp_version = "not installed in selected Python"
        checks = [
            (
                "python-module:yt-dlp",
                yt_dlp_available,
                f"{sys.executable} (yt-dlp {yt_dlp_version})",
            )
        ]
        checks.extend(
            (name, (path := shutil.which(name)) is not None, path or "not found on PATH")
            for name in ("ffmpeg", "ffprobe")
        )
        return tuple(checks)

    def download(
        self,
        task: MediaTask,
        destination: Path,
        *,
        timeout_seconds: float,
        on_log: LogSink | None = None,
    ) -> DownloadedMedia:
        with tempfile.TemporaryDirectory(prefix=f"media-{task.request_id}-") as raw_dir:
            template = str(Path(raw_dir) / "source.%(ext)s")
            metadata = self._run_json(
                [
                    sys.executable,
                    "-m",
                    "yt_dlp",
                    "--ignore-config",
                    "--no-playlist",
                    "--newline",
                    "--no-warnings",
                    "--print-json",
                    "-f",
                    "bestaudio*/best",
                    "-o",
                    template,
                    task.source_url,
                ],
                timeout_seconds,
                on_log,
            )
            sources = tuple(path for path in Path(raw_dir).iterdir() if path.is_file())
            if len(sources) != 1:
                raise MediaDownloadError(
                    "yt-dlp did not produce exactly one media file",
                    code="download_output",
                    retryable=True,
                )
            self._run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-y",
                    "-i",
                    str(sources[0]),
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "pcm_s16le",
                    str(destination),
                ],
                timeout_seconds,
                on_log,
            )
        probe = self._run_json(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(destination),
            ],
            timeout_seconds,
            on_log,
        )
        try:
            format_value = probe["format"]
            if not isinstance(format_value, dict):
                raise TypeError
            raw_duration = format_value["duration"]
            if not isinstance(raw_duration, str | int | float):
                raise TypeError
            duration = float(raw_duration)
        except (KeyError, TypeError, ValueError) as error:
            raise MediaDownloadError(
                "ffprobe returned no valid duration", code="invalid_audio", retryable=False
            ) from error
        if duration <= 0 or destination.stat().st_size <= 44:
            raise MediaDownloadError(
                "normalized audio is empty", code="invalid_audio", retryable=False
            )
        provenance_fields = (
            "id",
            "title",
            "extractor",
            "webpage_url",
            "duration",
            "format_id",
            "ext",
            "acodec",
        )
        provenance = {key: metadata[key] for key in provenance_fields if key in metadata}
        return DownloadedMedia(duration_seconds=duration, metadata=provenance)

    def _run_json(
        self, command: list[str], timeout: float, on_log: LogSink | None
    ) -> dict[str, object]:
        output = self._run(command, timeout, on_log)
        candidates = [line for line in output.splitlines() if line.lstrip().startswith("{")]
        if not candidates:
            raise MediaDownloadError(
                f"{command[0]} returned no JSON", code="invalid_response", retryable=True
            )
        try:
            value = json.loads(candidates[-1])
        except json.JSONDecodeError as error:
            raise MediaDownloadError(
                f"{command[0]} returned invalid JSON", code="invalid_response", retryable=True
            ) from error
        if not isinstance(value, dict):
            raise MediaDownloadError(
                f"{command[0]} returned a non-object", code="invalid_response", retryable=True
            )
        return value

    @staticmethod
    def _run(command: list[str], timeout: float, on_log: LogSink | None) -> str:
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            assert process.stdout is not None
            output: list[str] = []
            deadline = time.monotonic() + timeout
            selector = selectors.DefaultSelector()
            selector.register(process.stdout, selectors.EVENT_READ)
            while process.poll() is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    process.kill()
                    process.wait()
                    raise MediaDownloadError(
                        f"{command[0]} timed out after {timeout:g}s",
                        code="timeout",
                        retryable=True,
                    )
                for _, _ in selector.select(min(remaining, 0.25)):
                    line = process.stdout.readline()
                    if line:
                        output.append(line)
                        if on_log is not None:
                            on_log(line.rstrip("\r\n"))
            remainder = process.stdout.read()
            if remainder:
                output.append(remainder)
                if on_log is not None:
                    for line in remainder.splitlines():
                        on_log(line)
            stdout = "".join(output)
        except OSError as error:
            raise MediaDownloadError(
                f"could not execute {command[0]}: {error}", code="executable", retryable=False
            ) from error
        if process.returncode:
            tail = stdout.strip().splitlines()[-1] if stdout.strip() else "no diagnostics"
            raise MediaDownloadError(
                f"{command[0]} exited with status {process.returncode}: {tail}",
                code="process_failed",
                retryable=True,
            )
        return stdout
