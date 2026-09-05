from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from dominican_eaters.data import AudioSample, STTManifest, load_manifest
from dominican_eaters.evaluation.asr.runner import (
    BenchmarkRunner,
    FailureStage,
    OutputCollisionError,
    RunStatus,
    SampleResult,
    SampleStatus,
    ScoreStatus,
)
from dominican_eaters.evaluation.asr.scoring import ASREvaluationReport
from dominican_eaters.speech.asr import BackendDescriptor, Transcript


class FakeBackend:
    backend_id = "fake/model"

    def __init__(self, errors: dict[str, Exception] | None = None) -> None:
        self.errors = errors or {}
        self.calls: list[str] = []

    @property
    def descriptor(self) -> BackendDescriptor:
        return BackendDescriptor(
            backend_id=self.backend_id,
            model="model",
            model_revision="test",
            language="es",
            requested_device="cpu",
            requested_precision="fp32",
            effective_device="cpu",
            effective_precision="fp32",
        )

    def _call(self, name: str) -> None:
        self.calls.append(name)
        if name in self.errors:
            raise self.errors[name]

    def load(self) -> None:
        self._call("load")

    def warmup(self) -> None:
        self._call("warmup")

    def transcribe(self, audio_path: Path) -> Transcript:
        name = f"transcribe:{audio_path.name}"
        self._call(name)
        return Transcript(
            text=f"text for {audio_path.name}",
            language="es",
            audio_duration_seconds=2.0,
        )

    def close(self) -> None:
        self._call("close")


def manifest(tmp_path: Path, count: int = 2) -> STTManifest:
    samples = []
    for index in range(count):
        name = f"sample-{index}.wav"
        (tmp_path / name).write_bytes(b"offline fake audio")
        samples.append(
            AudioSample(
                sample_id=str(index),
                audio_path=name,
                reference_text=f"text for {name}",
                group_id=f"speaker-{index}",
            )
        )
    return STTManifest(dataset_root=tmp_path, samples=tuple(samples))


def test_success_owns_full_lifecycle_scores_and_persists(tmp_path: Path) -> None:
    backend = FakeBackend()
    source = manifest(tmp_path)
    output = tmp_path / "run"

    result = BenchmarkRunner(backend=backend, warmup_runs=2).run(source, output)

    assert result.status is RunStatus.COMPLETE
    assert result.successful
    assert result.coverage == 1.0
    assert result.scoring.status is ScoreStatus.OK
    assert result.scoring.report is not None
    assert result.scoring.report.speech_corpus.wer == 0.0
    assert result.performance.measured_samples == 2
    assert result.performance.total_inference_seconds >= 0
    assert result.performance.total_audio_seconds == 4.0
    assert result.performance.real_time_factor is not None
    assert result.performance.process_rss_peak_bytes is not None
    assert result.performance.process_tree_rss_peak_bytes is not None
    assert backend.calls == [
        "load",
        "warmup",
        "warmup",
        "transcribe:sample-0.wav",
        "transcribe:sample-1.wav",
        "close",
    ]
    replay = load_manifest(output / "manifest.json")
    replay.preflight()
    artifact = json.loads((output / "result.json").read_text(encoding="utf-8"))
    checkpoint = json.loads((output / "checkpoint.json").read_text(encoding="utf-8"))
    assert artifact["schema_version"] == 1
    assert artifact["performance"]["measured_samples"] == 2
    assert artifact["performance"]["total_audio_seconds"] == 4.0
    assert artifact["status"] == "complete"
    assert artifact["backend"]["backend_id"] == "fake/model"
    assert artifact["scoring"]["report"]["speech_corpus"]["words"]["substitutions"] == 0
    assert artifact["performance"]["model_load"]["elapsed_seconds"] >= 0
    assert checkpoint["phase"] == "finished"
    assert checkpoint["scoring"]["status"] == "ok"
    assert checkpoint["active_sample_id"] is None


@pytest.mark.parametrize("stage", ["load", "warmup"])
def test_lifecycle_start_failure_is_failed_and_closes(tmp_path: Path, stage: str) -> None:
    backend = FakeBackend({stage: RuntimeError(f"bad {stage}")})
    result = BenchmarkRunner(backend=backend).run(manifest(tmp_path), tmp_path / stage)

    assert result.status is RunStatus.FAILED
    assert not result.successful
    assert result.coverage == 0.0
    assert [failure.stage.value for failure in result.failures] == [stage]
    assert backend.calls[-1] == "close"
    assert not any(call.startswith("transcribe:") for call in backend.calls)


def test_transcription_failure_continues_and_reports_partial_coverage(
    tmp_path: Path,
) -> None:
    backend = FakeBackend({"transcribe:sample-0.wav": ValueError("invalid audio")})
    result = BenchmarkRunner(backend=backend).run(manifest(tmp_path), tmp_path / "partial")

    assert result.status is RunStatus.PARTIAL
    assert not result.successful
    assert result.succeeded_samples == 1
    assert result.failed_samples == 1
    assert result.coverage == 0.5
    assert [item.status for item in result.sample_results] == [
        SampleStatus.FAILED,
        SampleStatus.OK,
    ]
    assert result.scoring.report is not None
    assert result.scoring.report.coverage.failed == ("0",)
    assert result.failures[0].stage is FailureStage.TRANSCRIBE
    assert result.failures[0].sample_id == "0"
    assert backend.calls[-1] == "close"


def test_scoring_error_makes_otherwise_usable_run_partial(tmp_path: Path) -> None:
    def score(
        _samples: Sequence[AudioSample], _results: Sequence[SampleResult]
    ) -> ASREvaluationReport:
        raise ArithmeticError("metric policy failed")

    result = BenchmarkRunner(backend=FakeBackend(), scorer=score).run(
        manifest(tmp_path), tmp_path / "score-error"
    )

    assert result.status is RunStatus.PARTIAL
    assert result.coverage == 1.0
    assert result.scoring.status is ScoreStatus.FAILED
    assert result.failures[-1].stage is FailureStage.SCORE


def test_close_error_is_visible_and_non_success(tmp_path: Path) -> None:
    backend = FakeBackend({"close": OSError("release failed")})
    result = BenchmarkRunner(backend=backend).run(manifest(tmp_path), tmp_path / "close-error")

    assert result.status is RunStatus.PARTIAL
    assert result.coverage == 1.0
    assert result.failures[-1].stage is FailureStage.CLOSE
    assert not result.successful


def test_preflight_error_does_not_start_backend(tmp_path: Path) -> None:
    backend = FakeBackend()
    bad_manifest = STTManifest(
        dataset_root=tmp_path,
        samples=(
            AudioSample(
                sample_id="one",
                audio_path="missing.wav",
                reference_text="reference",
            ),
        ),
    )
    result = BenchmarkRunner(backend=backend).run(bad_manifest, tmp_path / "preflight-error")

    assert result.status is RunStatus.FAILED
    assert result.failures[0].stage is FailureStage.PREFLIGHT
    assert backend.calls == []


def test_existing_output_is_rejected_before_backend_work(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    backend = FakeBackend()

    with pytest.raises(OutputCollisionError, match="already exists"):
        BenchmarkRunner(backend=backend).run(manifest(tmp_path), output)

    assert backend.calls == []


def test_all_transcriptions_failed_is_failed(tmp_path: Path) -> None:
    errors = {
        "transcribe:sample-0.wav": RuntimeError("first"),
        "transcribe:sample-1.wav": RuntimeError("second"),
    }
    result = BenchmarkRunner(backend=FakeBackend(errors)).run(
        manifest(tmp_path), tmp_path / "all-failed"
    )

    assert result.status is RunStatus.FAILED
    assert result.coverage == 0.0
    assert len(result.failures) == 2
    assert result.scoring.report is not None
    assert result.scoring.report.coverage.failed == ("0", "1")


def test_constructor_rejects_negative_warmup_count() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        BenchmarkRunner(backend=FakeBackend(), warmup_runs=-1)


def test_invalid_backend_result_is_a_sample_failure(tmp_path: Path) -> None:
    class InvalidBackend(FakeBackend):
        def transcribe(self, audio_path: Path) -> Transcript:
            self._call(f"transcribe:{audio_path.name}")
            return "not a transcript"  # type: ignore[return-value]

    result = BenchmarkRunner(backend=InvalidBackend()).run(
        manifest(tmp_path, count=1), tmp_path / "invalid-backend"
    )

    assert result.status is RunStatus.FAILED
    assert result.failures[0].stage is FailureStage.TRANSCRIBE
    assert result.failures[0].error_type == "TypeError"


def test_keyboard_interrupt_preserves_completed_samples_and_active_sample(
    tmp_path: Path,
) -> None:
    class InterruptingBackend(FakeBackend):
        def transcribe(self, audio_path: Path) -> Transcript:
            if audio_path.name == "sample-1.wav":
                self.calls.append(f"transcribe:{audio_path.name}")
                raise KeyboardInterrupt("operator cancelled")
            return super().transcribe(audio_path)

    backend = InterruptingBackend()
    output = tmp_path / "interrupted"

    with pytest.raises(KeyboardInterrupt, match="operator cancelled"):
        BenchmarkRunner(backend=backend).run(manifest(tmp_path), output)

    checkpoint = json.loads((output / "checkpoint.json").read_text(encoding="utf-8"))
    assert (output / "manifest.json").is_file()
    assert not (output / "result.json").exists()
    assert checkpoint["phase"] == "interrupted"
    assert checkpoint["interrupted_phase"] == "transcribing"
    assert checkpoint["interruption_type"] == "KeyboardInterrupt"
    assert checkpoint["active_sample_id"] == "1"
    assert [sample["sample_id"] for sample in checkpoint["sample_results"]] == ["0"]
    assert backend.calls[-1] == "close"
