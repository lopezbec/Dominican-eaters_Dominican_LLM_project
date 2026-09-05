from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from dominican_eaters.cli import main
from dominican_eaters.collection.books import BookManifest, BookSeed, write_book_manifest
from dominican_eaters.collection.lyrics import LyricsManifest, LyricsRequest, write_lyrics_manifest
from dominican_eaters.collection.poems import PoemManifest, PoemSource, write_poem_manifest
from dominican_eaters.speech.asr import BackendDescriptor, Transcript, WhisperSettings


class CliFakeBackend:
    backend_id = "openai-whisper/fake"

    def __init__(self, _settings: WhisperSettings, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[str] = []

    @property
    def descriptor(self) -> BackendDescriptor:
        return BackendDescriptor(
            backend_id=self.backend_id,
            model="fake",
            model_revision="fixture",
            language="es",
            requested_device="cpu",
            requested_precision="fp32",
            effective_device="cpu",
            effective_precision="fp32",
        )

    def load(self) -> None:
        self.calls.append("load")

    def warmup(self) -> None:
        self.calls.append("warmup")

    def transcribe(self, audio_path: Path) -> Transcript:
        self.calls.append(f"transcribe:{audio_path.name}")
        if self.fail:
            raise RuntimeError("fake inference failure")
        return Transcript(text="Hola, mundo.", language="es", audio_duration_seconds=1.0)

    def close(self) -> None:
        self.calls.append("close")


class EmptyYouTubeAPI:
    instances: list[EmptyYouTubeAPI] = []

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.closed = False
        self.queries: list[str] = []
        self.instances.append(self)

    def search(self, query: str, *, max_results: int = 10) -> tuple[()]:
        self.queries.append(query)
        return ()

    def close(self) -> None:
        self.closed = True


class EmptyGeniusAPI:
    instances: list[EmptyGeniusAPI] = []

    def __init__(self, access_token: str) -> None:
        self.access_token = access_token
        self.closed = False
        self.queries: list[str] = []
        self.instances.append(self)

    def search(self, query: str) -> tuple[()]:
        self.queries.append(query)
        return ()

    def get_details(self, source_song_id: str) -> None:
        raise AssertionError(f"unexpected details lookup for {source_song_id}")

    def get_lyrics(self, url: str) -> None:
        raise AssertionError(f"unexpected lyrics lookup for {url}")

    def close(self) -> None:
        self.closed = True


def write_cli_manifest(tmp_path: Path) -> Path:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "sample.wav").write_bytes(b"offline audio fixture")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_root": str(dataset),
                "samples": [
                    {
                        "sample_id": "sample-001",
                        "audio_path": "sample.wav",
                        "reference_text": "Hola, mundo.",
                        "group_id": "recording-001",
                        "split": "eval",
                        "source": "human-reviewed",
                        "sha256": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_help_and_version_do_not_import_model_dependencies() -> None:
    runner = CliRunner()

    help_result = runner.invoke(main, ["--help"])
    version_result = runner.invoke(main, ["--version"])

    assert help_result.exit_code == 0
    assert "config" in help_result.output
    assert "stt" in help_result.output
    assert version_result.exit_code == 0
    assert "0.2.0" in version_result.output


def test_config_validation_has_nonzero_exit_for_invalid_config(tmp_path: Path) -> None:
    config = tmp_path / "invalid.yaml"
    config.write_text("schema_version: 1\ndata_root: data\n", encoding="utf-8")

    result = CliRunner().invoke(main, ["config", "validate", str(config)])

    assert result.exit_code != 0
    assert "missing fields: artifacts_root" in result.output


def test_stt_preflight_success_and_missing_file_failure(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    audio = dataset / "sample.wav"
    audio.write_bytes(b"preflight checks identity and file presence")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_root": str(dataset),
                "samples": [
                    {
                        "sample_id": "sample-001",
                        "audio_path": "sample.wav",
                        "reference_text": "Hola, mundo.",
                        "group_id": "recording-001",
                        "split": "eval",
                        "source": "human-reviewed",
                        "sha256": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    runner = CliRunner()

    success = runner.invoke(main, ["stt", "preflight", str(manifest)])
    audio.unlink()
    failure = runner.invoke(main, ["stt", "preflight", str(manifest)])

    assert success.exit_code == 0
    assert "samples=1" in success.output
    assert "preflight=passed" in success.output
    assert failure.exit_code != 0
    assert "audio file does not exist" in failure.output


def test_stt_benchmark_success_uses_canonical_runner_and_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = write_cli_manifest(tmp_path)
    backend = CliFakeBackend(WhisperSettings())
    monkeypatch.setattr("dominican_eaters.cli.app.create_asr_backend", lambda **_kwargs: backend)
    output = tmp_path / "run"

    result = CliRunner().invoke(
        main,
        ["stt", "benchmark", str(manifest), "--output-dir", str(output)],
    )

    assert result.exit_code == 0, result.output
    assert "status=complete" in result.output
    assert "backend_id=openai-whisper/fake" in result.output
    assert "succeeded_samples=1" in result.output
    assert backend.calls == ["load", "warmup", "transcribe:sample.wav", "close"]
    artifact = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert artifact["backend"]["model_revision"] == "fixture"


def test_stt_benchmark_failure_exits_nonzero_and_keeps_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = write_cli_manifest(tmp_path)
    backend = CliFakeBackend(WhisperSettings(), fail=True)
    monkeypatch.setattr("dominican_eaters.cli.app.create_asr_backend", lambda **_kwargs: backend)
    output = tmp_path / "failed-run"

    result = CliRunner().invoke(
        main,
        ["stt", "benchmark", str(manifest), "--output-dir", str(output)],
    )

    assert result.exit_code == 1
    assert "status=failed" in result.output
    assert "failure=transcribe:RuntimeError:fake inference failure" in result.output
    artifact = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert artifact["status"] == "failed"
    assert artifact["failures"][0]["stage"] == "transcribe"


def test_book_manifest_preflight_uses_installed_cli(tmp_path: Path) -> None:
    path = tmp_path / "books.json"
    write_book_manifest(BookManifest((BookSeed.create("Over", "Ramón Marrero Aristy"),)), path)

    result = CliRunner().invoke(main, ["collect", "books", "preflight", str(path)])

    assert result.exit_code == 0, result.output
    assert "books=1" in result.output
    assert "preflight=passed" in result.output


def test_lyrics_manifest_preflight_uses_installed_cli(tmp_path: Path) -> None:
    path = tmp_path / "lyrics.json"
    write_lyrics_manifest(LyricsManifest((LyricsRequest("req-1", "Juan Luis Guerra"),)), path)

    result = CliRunner().invoke(main, ["collect", "lyrics", "preflight", str(path)])

    assert result.exit_code == 0, result.output
    assert "requests=1" in result.output
    assert "preflight=passed" in result.output


def test_poem_manifest_preflight_uses_installed_cli(tmp_path: Path) -> None:
    path = tmp_path / "poems.json"
    write_poem_manifest(
        PoemManifest((PoemSource.create(title="Hay un país en el mundo", author="Pedro Mir"),)),
        path,
    )

    result = CliRunner().invoke(main, ["collect", "poems", "preflight", str(path)])

    assert result.exit_code == 0, result.output
    assert "poems=1" in result.output
    assert "preflight=passed" in result.output


def test_collection_run_requires_credentials_from_environment(tmp_path: Path) -> None:
    path = tmp_path / "books.json"
    write_book_manifest(BookManifest((BookSeed.create("Over", "Ramón Marrero Aristy"),)), path)

    result = CliRunner().invoke(
        main,
        ["collect", "books", "run", str(path), "--output-dir", str(tmp_path / "output")],
        env={"YOUTUBE_API_KEY": ""},
    )

    assert result.exit_code != 0
    assert "YOUTUBE_API_KEY is required in the environment" in result.output


def test_book_collection_run_writes_checkpoint_and_closes_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    EmptyYouTubeAPI.instances.clear()
    path = tmp_path / "books.json"
    output = tmp_path / "books-output"
    write_book_manifest(BookManifest((BookSeed.create("Over", "Ramón Marrero Aristy"),)), path)
    monkeypatch.setattr("dominican_eaters.cli.app.YouTubeDataAPI", EmptyYouTubeAPI)

    result = CliRunner().invoke(
        main,
        ["collect", "books", "run", str(path), "--output-dir", str(output)],
        env={"YOUTUBE_API_KEY": "youtube-secret"},
    )

    assert result.exit_code == 0, result.output
    assert "not_found=1" in result.output
    assert (output / "books-collection.json").is_file()
    assert EmptyYouTubeAPI.instances[0].api_key == "youtube-secret"
    assert EmptyYouTubeAPI.instances[0].closed is True


def test_lyrics_collection_run_writes_ledger_and_closes_providers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    EmptyGeniusAPI.instances.clear()
    EmptyYouTubeAPI.instances.clear()
    path = tmp_path / "lyrics.json"
    output = tmp_path / "lyrics-output"
    write_lyrics_manifest(LyricsManifest((LyricsRequest("req-1", "Juan Luis Guerra"),)), path)
    monkeypatch.setattr("dominican_eaters.cli.app.GeniusAPI", EmptyGeniusAPI)
    monkeypatch.setattr("dominican_eaters.cli.app.YouTubeDataAPI", EmptyYouTubeAPI)

    result = CliRunner().invoke(
        main,
        ["collect", "lyrics", "run", str(path), "--output-dir", str(output)],
        env={"GENIUS_ACCESS_TOKEN": "genius-secret", "YOUTUBE_API_KEY": "youtube-secret"},
    )

    assert result.exit_code == 0, result.output
    assert "not_found=1" in result.output
    assert (output / "lyrics-collection.json").is_file()
    assert EmptyGeniusAPI.instances[0].closed is True
    assert EmptyYouTubeAPI.instances[0].closed is True


def test_poem_collection_run_writes_checkpoint_and_closes_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    EmptyYouTubeAPI.instances.clear()
    path = tmp_path / "poems.json"
    output = tmp_path / "poems-output"
    write_poem_manifest(
        PoemManifest((PoemSource.create(title="Hay un país en el mundo", author="Pedro Mir"),)),
        path,
    )
    monkeypatch.setattr("dominican_eaters.cli.app.YouTubeDataAPI", EmptyYouTubeAPI)

    result = CliRunner().invoke(
        main,
        ["collect", "poems", "run", str(path), "--output-dir", str(output)],
        env={"YOUTUBE_API_KEY": "youtube-secret"},
    )

    assert result.exit_code == 0, result.output
    assert "not_found=1" in result.output
    assert (output / "poems-collection.json").is_file()
    assert EmptyYouTubeAPI.instances[0].closed is True
