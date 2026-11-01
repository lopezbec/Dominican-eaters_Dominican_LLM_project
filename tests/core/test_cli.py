from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from dominican_eaters.cli import main
from dominican_eaters.collection.books import BookManifest, BookSeed, write_book_manifest
from dominican_eaters.collection.lyrics import LyricsManifest, LyricsRequest, write_lyrics_manifest
from dominican_eaters.collection.poems import PoemManifest, PoemSource, write_poem_manifest
from dominican_eaters.speech.asr import BackendDescriptor, Transcript, WhisperSettings
from dominican_eaters.speech.asr.environment import BackendEnvironmentReport, EnvironmentCheck
from dominican_eaters.speech.asr.registry import MODEL_PRESETS


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


class EmptyYouTubeSearch:
    instances: list[EmptyYouTubeSearch] = []

    def __init__(self) -> None:
        self.queries: list[str] = []
        self.instances.append(self)

    def search(self, query: str, *, max_results: int = 10) -> tuple[()]:
        self.queries.append(query)
        return ()


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


def write_launch_manifest(project_root: Path, dataset_root: Path, audio_paths: list[str]) -> Path:
    manifest = project_root / "data" / "manifests" / "stt-all.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_root": str(dataset_root),
                "samples": [
                    {
                        "sample_id": f"sample-{index}",
                        "audio_path": audio_path,
                        "reference_text": "Hola, mundo.",
                    }
                    for index, audio_path in enumerate(audio_paths, start=1)
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


def test_bare_noninteractive_command_shows_help() -> None:
    result = CliRunner().invoke(main)

    assert result.exit_code == 0
    assert "Run without a subcommand" in result.output
    assert "tui" in result.output


def test_stt_launch_opens_autorunning_lyrics_suite_in_tmux(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_launch_manifest(tmp_path, tmp_path / "data" / "audio" / "lyrics", ["sample.wav"])
    launch_arguments: dict[str, object] = {}

    def launch(**kwargs: object) -> None:
        launch_arguments.update(kwargs)

    monkeypatch.setattr("dominican_eaters.cli.app._project_root", lambda: tmp_path)
    monkeypatch.setattr("dominican_eaters.tui.launch_tui_in_tmux", launch)

    result = CliRunner().invoke(
        main,
        [
            "stt",
            "launch",
            "--session",
            "research-stt",
            "--output-dir",
            "artifacts/custom-suite",
        ],
    )

    assert result.exit_code == 0, result.output
    assert launch_arguments == {
        "session_name": "research-stt",
        "initial_state": {
            "workflow": "stt-suite",
            "dataset": "music",
            "output": "artifacts/custom-suite",
            "device": "cuda",
            "precision": "fp16",
            "warmupRuns": "1",
            "requestTimeout": "1800",
            "verifyHashes": True,
        },
        "auto_run": True,
    }


def test_stt_launch_accepts_lyrics_subset_of_shared_audio_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_launch_manifest(
        tmp_path,
        tmp_path / "data" / "audio",
        ["lyrics/one.wav", "lyrics/album/two.mp3"],
    )
    launched = False

    def launch(**_kwargs: object) -> None:
        nonlocal launched
        launched = True

    monkeypatch.setattr("dominican_eaters.cli.app._project_root", lambda: tmp_path)
    monkeypatch.setattr("dominican_eaters.tui.launch_tui_in_tmux", launch)

    result = CliRunner().invoke(main, ["stt", "launch"])

    assert result.exit_code == 0, result.output
    assert launched


@pytest.mark.parametrize(
    ("dataset_root", "audio_paths"),
    [
        ("data/audio/books", ["sample.wav"]),
        ("data/audio", ["lyrics/one.wav", "books/two.wav"]),
    ],
)
def test_stt_launch_rejects_manifest_that_is_not_limited_to_collected_lyrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dataset_root: str,
    audio_paths: list[str],
) -> None:
    write_launch_manifest(tmp_path, tmp_path / dataset_root, audio_paths)
    monkeypatch.setattr("dominican_eaters.cli.app._project_root", lambda: tmp_path)

    result = CliRunner().invoke(main, ["stt", "launch"])

    assert result.exit_code != 0
    assert "not limited to collected lyrics" in result.output
    assert "stt manifest build data/audio/lyrics" in result.output


def test_stt_launch_requires_full_lyrics_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("dominican_eaters.cli.app._project_root", lambda: tmp_path)

    result = CliRunner().invoke(main, ["stt", "launch"])

    assert result.exit_code != 0
    assert "full lyrics STT manifest is missing" in result.output
    assert "stt manifest build" in result.output


def test_stt_model_catalog_lists_all_presets_and_explains_blocked_models() -> None:
    runner = CliRunner()

    listing = runner.invoke(main, ["stt", "models", "list"])
    details = runner.invoke(main, ["stt", "models", "show", "granite-speech-3.3-8b"])

    assert listing.exit_code == 0, listing.output
    assert "whisper-base\tcurrent" in listing.output
    assert "whisper-large-v3\tcurrent" in listing.output
    assert "whisper-turbo\tcurrent" in listing.output
    assert "granite-speech-4.1-2b\tcandidate" in listing.output
    assert "qwen2-audio-7b-instruct\texperimental" in listing.output
    assert "granite-speech-3.3-8b\tblocked" in listing.output
    assert details.exit_code == 0, details.output
    assert "runtime=granite-3.3-transformers" in details.output
    assert "execution=worker" in details.output
    assert "status=blocked" in details.output
    assert "reason=Published BF16 weights exceed T4 VRAM" in details.output


def test_stt_manifest_build_includes_every_audio_file(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "one.wav").write_bytes(b"one")
    (audio_dir / "one.txt").write_text("Uno", encoding="utf-8")
    (audio_dir / "two.mp3").write_bytes(b"two")
    output = tmp_path / "stt-all.json"

    result = CliRunner().invoke(
        main,
        ["stt", "manifest", "build", str(audio_dir), "--output-file", str(output)],
    )

    assert result.exit_code == 0, result.output
    assert "samples=2" in result.output
    assert "references=1" in result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert [sample["audio_path"] for sample in payload["samples"]] == ["one.wav", "two.mp3"]


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


def test_stt_preflight_reports_selected_model_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = write_cli_manifest(tmp_path)
    report = BackendEnvironmentReport(
        backend="whisper",
        interpreter=Path(sys.executable),
        requested_device="cuda",
        python_version="3.12.8",
        checks=(EnvironmentCheck("cuda", "torch.cuda", True, "available"),),
    )
    monkeypatch.setattr("dominican_eaters.cli.app.create_asr_backend", lambda **_kwargs: object())
    monkeypatch.setattr(
        "dominican_eaters.cli.app.preflight_asr_environment", lambda **_kwargs: report
    )

    result = CliRunner().invoke(
        main,
        [
            "stt",
            "preflight",
            str(manifest),
            "--backend",
            "whisper",
            "--model",
            "large-v3",
            "--device",
            "cuda",
            "--precision",
            "fp16",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "environment_backend=whisper" in result.output
    assert "environment_model=large-v3" in result.output
    assert "environment_check[cuda:torch.cuda]=ok:available" in result.output
    assert "environment_preflight=passed" in result.output


def test_stt_preflight_does_not_request_worker_handshake_for_whisper_preset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = write_cli_manifest(tmp_path)
    report = BackendEnvironmentReport(
        backend="whisper",
        interpreter=Path(sys.executable),
        requested_device="cuda",
        python_version="3.12.8",
        checks=(EnvironmentCheck("cuda", "torch.cuda", True, "available"),),
    )
    monkeypatch.setattr(
        "dominican_eaters.cli.app.preflight_asr_environment", lambda **_kwargs: report
    )

    result = CliRunner().invoke(
        main,
        [
            "stt",
            "preflight",
            str(manifest),
            "--preset",
            "whisper-large-v3",
            "--device",
            "cuda",
            "--precision",
            "fp16",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "preset=whisper-large-v3" in result.output
    assert "environment_preflight=passed" in result.output
    assert "worker_preflight" not in result.output


@pytest.mark.parametrize(
    ("preset_id", "status"),
    [
        ("qwen2-audio-7b-instruct", "experimental"),
        ("granite-speech-3.3-8b", "blocked"),
    ],
)
def test_stt_preflight_reports_non_runnable_preset_without_constructing_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    preset_id: str,
    status: str,
) -> None:
    manifest = write_cli_manifest(tmp_path)
    received: dict[str, object] = {}

    def report(**kwargs: object) -> BackendEnvironmentReport:
        received.update(kwargs)
        return BackendEnvironmentReport(
            backend=kwargs["backend"],  # type: ignore[arg-type]
            interpreter=Path(sys.executable),
            requested_device="cuda",
            python_version="3.12.8",
            checks=(EnvironmentCheck("module", "torch", True, "available"),),
        )

    monkeypatch.setattr("dominican_eaters.cli.app.preflight_asr_environment", report)

    def unexpected_backend(**_kwargs: object) -> object:
        raise AssertionError("non-runnable presets must not construct a backend")

    monkeypatch.setattr("dominican_eaters.cli.app.create_asr_backend", unexpected_backend)

    result = CliRunner().invoke(
        main,
        ["stt", "preflight", str(manifest), "--preset", preset_id],
    )

    assert result.exit_code != 0
    assert f"preset={preset_id}" in result.output
    assert f"preset_status={status}" in result.output
    assert "preset_runnable=no" in result.output
    assert "environment_preflight=passed" in result.output
    assert f"is {status} and cannot complete preflight" in result.output
    assert received["requested_device"] == "cuda"


def test_stt_preset_rejects_conflicting_legacy_backend(tmp_path: Path) -> None:
    manifest = write_cli_manifest(tmp_path)

    result = CliRunner().invoke(
        main,
        [
            "stt",
            "preflight",
            str(manifest),
            "--preset",
            "canary-1b-v2",
            "--backend",
            "whisper",
        ],
    )

    assert result.exit_code != 0
    assert "uses backend 'canary', not 'whisper'" in result.output


def test_stt_benchmark_rejects_non_runnable_preset_with_registered_reason(
    tmp_path: Path,
) -> None:
    manifest = write_cli_manifest(tmp_path)

    result = CliRunner().invoke(
        main,
        [
            "stt",
            "benchmark",
            str(manifest),
            "--output-dir",
            str(tmp_path / "run"),
            "--preset",
            "qwen2-audio-7b-instruct",
        ],
    )

    assert result.exit_code != 0
    assert "is experimental and cannot benchmark" in result.output
    assert "Spanish ASR" in result.output


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


def test_stt_benchmark_streams_machine_readable_sample_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = write_cli_manifest(tmp_path)
    backend = CliFakeBackend(WhisperSettings())
    monkeypatch.setattr("dominican_eaters.cli.app.create_asr_backend", lambda **_kwargs: backend)

    result = CliRunner().invoke(
        main,
        ["stt", "benchmark", str(manifest), "--output-dir", str(tmp_path / "events")],
    )

    assert result.exit_code == 0, result.output
    events = [
        json.loads(line.removeprefix("stt_event="))
        for line in result.output.splitlines()
        if line.startswith("stt_event=")
    ]
    started = next(event for event in events if event["event"] == "sample_started")
    completed = next(event for event in events if event["event"] == "sample_completed")
    assert started["sample_id"] == "sample-001"
    assert started["index"] == 1
    assert started["total"] == 1
    assert completed["status"] == "ok"
    assert completed["completed"] == 1
    assert completed["source"] == "human-reviewed"
    assert completed["audio_seconds"] == 1.0
    assert events[-1]["event"] == "run_finished"
    assert events[-1]["run_status"] == "complete"


def test_stt_benchmark_current_preset_resolves_registered_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = write_cli_manifest(tmp_path)
    backend = CliFakeBackend(WhisperSettings())
    received: dict[str, object] = {}

    def create_backend(**kwargs: object) -> CliFakeBackend:
        received.update(kwargs)
        return backend

    monkeypatch.setattr("dominican_eaters.cli.app.create_asr_backend", create_backend)
    result = CliRunner().invoke(
        main,
        [
            "stt",
            "benchmark",
            str(manifest),
            "--output-dir",
            str(tmp_path / "preset-run"),
            "--preset",
            "whisper-base",
        ],
    )

    assert result.exit_code == 0, result.output
    assert received["backend"] == "whisper"
    assert received["model"] == "base"
    assert received["language"] == "es"
    assert received["device"] == "cuda"
    assert received["precision"] == "fp16"


def test_stt_benchmark_candidate_preset_reaches_worker_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = write_cli_manifest(tmp_path)
    backend = CliFakeBackend(WhisperSettings(model="fixture"))
    received: dict[str, object] = {}

    def create_backend(**kwargs: object) -> CliFakeBackend:
        received.update(kwargs)
        return backend

    monkeypatch.setattr("dominican_eaters.cli.app.create_asr_backend", create_backend)
    result = CliRunner().invoke(
        main,
        [
            "stt",
            "benchmark",
            str(manifest),
            "--output-dir",
            str(tmp_path / "candidate-run"),
            "--preset",
            "granite-speech-4.1-2b",
            "--worker-python",
            sys.executable,
        ],
    )

    assert result.exit_code == 0, result.output
    assert received["backend"] == "granite"
    assert received["model"] == "ibm-granite/granite-speech-4.1-2b"
    assert received["worker_python"] == Path(sys.executable)
    assert received["preset"] == MODEL_PRESETS["granite-speech-4.1-2b"]


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


def test_stt_suite_runs_named_dataset_and_resumes_completed_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = write_cli_manifest(tmp_path)
    backend = CliFakeBackend(WhisperSettings(model="large-v3"))
    report = BackendEnvironmentReport(
        backend="whisper",
        interpreter=Path(sys.executable),
        requested_device="cpu",
        python_version="3.12.8",
        checks=(EnvironmentCheck("module", "whisper", True, "available"),),
    )
    monkeypatch.setattr("dominican_eaters.cli.app.create_asr_backend", lambda **_kwargs: backend)
    monkeypatch.setattr(
        "dominican_eaters.cli.app.preflight_asr_environment", lambda **_kwargs: report
    )
    output = tmp_path / "suite"
    arguments = [
        "stt",
        "suite",
        "--dataset",
        f"music={manifest}",
        "--preset",
        "whisper-large-v3",
        "--output-dir",
        str(output),
        "--device",
        "cpu",
        "--precision",
        "fp32",
        "--no-verify-hashes",
    ]

    first = CliRunner().invoke(main, arguments)
    calls_after_first = tuple(backend.calls)
    resumed = CliRunner().invoke(main, arguments)

    assert first.exit_code == 0, first.output
    assert "suite_status=complete" in first.output
    assert "suite_task[music,whisper-large-v3]=complete:1/1:attempt=1" in first.output
    assert '"event":"suite_started"' in first.output
    assert '"event":"sample_completed"' in first.output
    assert resumed.exit_code == 0, resumed.output
    assert '"event":"task_skipped"' in resumed.output
    assert tuple(backend.calls) == calls_after_first
    ledger = json.loads((output / "suite.json").read_text(encoding="utf-8"))
    assert ledger["plan"]["datasets"][0]["dataset_id"] == "music"
    assert ledger["tasks"][0]["attempt"] == 1


def test_stt_suite_rejects_malformed_dataset_spec(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        main,
        ["stt", "suite", "--dataset", "missing-separator", "--output-dir", str(tmp_path)],
    )

    assert result.exit_code != 0
    assert "ID=MANIFEST" in result.output


def test_stt_suite_fails_before_audio_when_worker_environments_are_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = write_cli_manifest(tmp_path)
    monkeypatch.setattr("dominican_eaters.cli.app.discover_worker_python", lambda **_kwargs: "")

    result = CliRunner().invoke(
        main,
        [
            "stt",
            "suite",
            "--dataset",
            f"music={manifest}",
            "--output-dir",
            str(tmp_path / "suite"),
        ],
    )

    assert result.exit_code != 0
    assert "Missing isolated STT worker environments before suite start" in result.output
    assert "parakeet-tdt-0.6b-v3" in result.output
    assert not (tmp_path / "suite" / "suite.json").exists()


def test_stt_suite_checks_every_environment_before_first_model_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = write_cli_manifest(tmp_path)
    backend = CliFakeBackend(WhisperSettings(model="large-v3"))

    def environment_report(*, backend: str, **_kwargs: object) -> BackendEnvironmentReport:
        ready = backend == "whisper"
        return BackendEnvironmentReport(
            backend=backend,
            interpreter=Path(sys.executable),
            requested_device="cuda",
            python_version="3.12.8",
            checks=(
                EnvironmentCheck(
                    "module",
                    "runtime",
                    ready,
                    "available" if ready else "worker module is unavailable",
                ),
            ),
        )

    monkeypatch.setattr("dominican_eaters.cli.app.create_asr_backend", lambda **_kwargs: backend)
    monkeypatch.setattr("dominican_eaters.cli.app.preflight_asr_environment", environment_report)
    output = tmp_path / "suite"

    result = CliRunner().invoke(
        main,
        [
            "stt",
            "suite",
            "--dataset",
            f"music={manifest}",
            "--preset",
            "whisper-large-v3",
            "--preset",
            "parakeet-tdt-0.6b-v3",
            "--worker-python",
            f"nemo-worker={sys.executable}",
            "--output-dir",
            str(output),
        ],
    )

    assert result.exit_code != 0
    assert "failed before processing audio" in result.output
    assert "parakeet-tdt-0.6b-v3" in result.output
    assert backend.calls == []
    assert not (output / "suite.json").exists()


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
    path = tmp_path / "lyrics.json"
    write_lyrics_manifest(LyricsManifest((LyricsRequest("req-1", "Juan Luis Guerra"),)), path)

    result = CliRunner().invoke(
        main,
        ["collect", "lyrics", "run", str(path), "--output-dir", str(tmp_path / "output")],
        env={"GENIUS_ACCESS_TOKEN": ""},
    )

    assert result.exit_code != 0
    assert "GENIUS_ACCESS_TOKEN is required in the environment" in result.output


def test_book_collection_run_writes_checkpoint_without_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    EmptyYouTubeSearch.instances.clear()
    path = tmp_path / "books.json"
    output = tmp_path / "books-output"
    write_book_manifest(BookManifest((BookSeed.create("Over", "Ramón Marrero Aristy"),)), path)
    monkeypatch.setattr("dominican_eaters.cli.app.ScrapeTubeSearch", EmptyYouTubeSearch)

    result = CliRunner().invoke(
        main,
        ["collect", "books", "run", str(path), "--output-dir", str(output)],
    )

    assert result.exit_code == 0, result.output
    assert "not_found=1" in result.output
    assert (output / "books-collection.json").is_file()
    assert EmptyYouTubeSearch.instances


def test_lyrics_collection_run_writes_ledger_and_closes_genius(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    EmptyGeniusAPI.instances.clear()
    EmptyYouTubeSearch.instances.clear()
    path = tmp_path / "lyrics.json"
    output = tmp_path / "lyrics-output"
    write_lyrics_manifest(LyricsManifest((LyricsRequest("req-1", "Juan Luis Guerra"),)), path)
    monkeypatch.setattr("dominican_eaters.cli.app.GeniusAPI", EmptyGeniusAPI)
    monkeypatch.setattr("dominican_eaters.cli.app.ScrapeTubeSearch", EmptyYouTubeSearch)

    result = CliRunner().invoke(
        main,
        ["collect", "lyrics", "run", str(path), "--output-dir", str(output)],
        env={"GENIUS_ACCESS_TOKEN": "genius-secret"},
    )

    assert result.exit_code == 0, result.output
    assert "progress=1/1 state=started request_id=req-1" in result.output
    assert "progress=1/1 state=saved status=not_found request_id=req-1" in result.output
    assert "not_found=1" in result.output
    assert (output / "lyrics-collection.json").is_file()
    assert EmptyGeniusAPI.instances[0].closed is True
    assert EmptyYouTubeSearch.instances


def test_poem_collection_run_writes_checkpoint_without_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    EmptyYouTubeSearch.instances.clear()
    path = tmp_path / "poems.json"
    output = tmp_path / "poems-output"
    write_poem_manifest(
        PoemManifest((PoemSource.create(title="Hay un país en el mundo", author="Pedro Mir"),)),
        path,
    )
    monkeypatch.setattr("dominican_eaters.cli.app.ScrapeTubeSearch", EmptyYouTubeSearch)

    result = CliRunner().invoke(
        main,
        ["collect", "poems", "run", str(path), "--output-dir", str(output)],
    )

    assert result.exit_code == 0, result.output
    assert "not_found=1" in result.output
    assert (output / "poems-collection.json").is_file()
    assert EmptyYouTubeSearch.instances
