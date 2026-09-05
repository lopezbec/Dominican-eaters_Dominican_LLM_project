"""Dependency-injected orchestration for reproducible ASR benchmarks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import Protocol

from dominican_eaters.data import AudioSample, STTManifest, write_manifest
from dominican_eaters.evaluation.asr.artifacts import write_artifact
from dominican_eaters.evaluation.asr.scoring import (
    ASREvaluationReport,
    Recognition,
    ReferenceSample,
    evaluate_asr,
)
from dominican_eaters.evaluation.resources import ResourceMeasurement, measure_call
from dominican_eaters.speech.asr import ASRBackend, BackendDescriptor, Transcript

BENCHMARK_RESULT_SCHEMA_VERSION = 1
BENCHMARK_CHECKPOINT_SCHEMA_VERSION = 1


class OutputCollisionError(FileExistsError):
    """Raised before model work when the requested output path already exists."""


class RunStatus(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"


class SampleStatus(StrEnum):
    OK = "ok"
    FAILED = "failed"


class ScoreStatus(StrEnum):
    OK = "ok"
    FAILED = "failed"
    SKIPPED = "skipped"


class FailureStage(StrEnum):
    PREFLIGHT = "preflight"
    LOAD = "load"
    WARMUP = "warmup"
    TRANSCRIBE = "transcribe"
    SCORE = "score"
    CLOSE = "close"
    PERSIST = "persist"


class BenchmarkPhase(StrEnum):
    CREATED = "created"
    PREFLIGHT = "preflight"
    LOADING = "loading"
    WARMING_UP = "warming_up"
    TRANSCRIBING = "transcribing"
    SCORING = "scoring"
    CLOSING = "closing"
    FINISHED = "finished"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True, slots=True)
class Failure:
    stage: FailureStage
    error_type: str
    message: str
    sample_id: str | None = None

    @classmethod
    def from_exception(
        cls,
        stage: FailureStage,
        exception: Exception,
        *,
        sample_id: str | None = None,
    ) -> Failure:
        return cls(
            stage=stage,
            error_type=type(exception).__name__,
            message=str(exception),
            sample_id=sample_id,
        )


@dataclass(frozen=True, slots=True)
class SampleResult:
    sample_id: str
    status: SampleStatus
    transcript: Transcript | None = None
    measurement: ResourceMeasurement | None = None
    failure: Failure | None = None

    def __post_init__(self) -> None:
        valid_success = (
            self.status is SampleStatus.OK and self.transcript is not None and self.failure is None
        )
        valid_failure = (
            self.status is SampleStatus.FAILED
            and self.transcript is None
            and self.failure is not None
        )
        if not (valid_success or valid_failure):
            raise ValueError("SampleResult status does not match its payload")


@dataclass(frozen=True, slots=True)
class ScoreResult:
    status: ScoreStatus
    report: ASREvaluationReport | None = None
    failure: Failure | None = None

    def __post_init__(self) -> None:
        if self.status is ScoreStatus.OK:
            if self.report is None or self.failure is not None:
                raise ValueError("Successful scoring requires a report and no failure")
        elif self.status is ScoreStatus.FAILED:
            if self.report is not None or self.failure is None:
                raise ValueError("Failed scoring requires a failure and no report")
        elif self.report is not None or self.failure is not None:
            raise ValueError("Skipped scoring cannot contain a report or failure")


@dataclass(frozen=True, slots=True)
class PerformanceSummary:
    model_load: ResourceMeasurement | None
    measured_samples: int
    total_inference_seconds: float
    total_audio_seconds: float
    real_time_factor: float | None
    process_rss_peak_bytes: int | None
    process_tree_rss_peak_bytes: int | None
    gpu_peak_allocated_bytes: int | None
    gpu_peak_reserved_bytes: int | None
    measurement_scope: str


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    backend: BackendDescriptor
    status: RunStatus
    output_dir: Path
    expected_samples: int
    sample_results: tuple[SampleResult, ...]
    performance: PerformanceSummary
    scoring: ScoreResult
    failures: tuple[Failure, ...]
    schema_version: int = BENCHMARK_RESULT_SCHEMA_VERSION

    @property
    def backend_id(self) -> str:
        return self.backend.backend_id

    @property
    def succeeded_samples(self) -> int:
        return sum(result.status is SampleStatus.OK for result in self.sample_results)

    @property
    def failed_samples(self) -> int:
        return self.expected_samples - self.succeeded_samples

    @property
    def coverage(self) -> float:
        if self.expected_samples == 0:
            return 0.0
        return self.succeeded_samples / self.expected_samples

    @property
    def successful(self) -> bool:
        """Whether an automation process should treat this run as successful."""

        return self.status is RunStatus.COMPLETE


@dataclass(frozen=True, slots=True)
class BenchmarkCheckpoint:
    """Latest durable lifecycle state for a run that may still be incomplete."""

    backend: BackendDescriptor
    phase: BenchmarkPhase
    expected_samples: int
    sample_results: tuple[SampleResult, ...]
    scoring: ScoreResult
    failures: tuple[Failure, ...]
    load_measurement: ResourceMeasurement | None
    updated_at: str
    active_sample_id: str | None = None
    interrupted_phase: BenchmarkPhase | None = None
    interruption_type: str | None = None
    interruption_message: str | None = None
    schema_version: int = BENCHMARK_CHECKPOINT_SCHEMA_VERSION


class Scorer(Protocol):
    """Adapter between orchestration records and a scientific scoring policy."""

    def __call__(
        self,
        samples: Sequence[AudioSample],
        results: Sequence[SampleResult],
    ) -> ASREvaluationReport: ...


def score_benchmark_results(
    samples: Sequence[AudioSample],
    results: Sequence[SampleResult],
) -> ASREvaluationReport:
    """Adapt canonical runner records to the canonical scientific policy."""

    references = tuple(
        ReferenceSample(
            sample_id=sample.sample_id,
            group_id=sample.group_id or sample.sample_id,
            text=sample.reference_text,
        )
        for sample in samples
    )
    recognitions = tuple(
        Recognition(
            sample_id=result.sample_id,
            text=result.transcript.text if result.transcript is not None else None,
            status="ok" if result.status is SampleStatus.OK else "failed",
            error=result.failure.message if result.failure is not None else None,
        )
        for result in results
    )
    return evaluate_asr(references, recognitions)


class BenchmarkRunner:
    """Own an ASR backend lifecycle and expose all non-success outcomes."""

    def __init__(
        self,
        *,
        backend: ASRBackend,
        scorer: Scorer = score_benchmark_results,
        warmup_runs: int = 1,
        verify_hashes: bool = False,
    ) -> None:
        if warmup_runs < 0:
            raise ValueError("warmup_runs must be nonnegative")
        self._backend = backend
        self._scorer = scorer
        self._warmup_runs = warmup_runs
        self._verify_hashes = verify_hashes

    def run(self, manifest: STTManifest, output_dir: Path) -> BenchmarkResult:
        """Run once into a new output directory.

        Existing paths are rejected even when empty. This prevents an old or
        concurrent run from being mistaken for the current result.
        """

        destination = output_dir.resolve()
        try:
            destination.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise OutputCollisionError(f"Benchmark output already exists: {destination}") from exc

        samples = tuple(manifest.samples)
        failures: list[Failure] = []
        results: list[SampleResult] = []
        scoring = ScoreResult(ScoreStatus.SKIPPED)
        load_measurement: ResourceMeasurement | None = None
        lifecycle_started = False
        ready = True
        checkpoint_available = True
        interrupted: BaseException | None = None
        interruption_phase: BenchmarkPhase | None = None
        phase = BenchmarkPhase.CREATED
        active_sample_id: str | None = None

        try:
            write_manifest(manifest, destination / "manifest.json")
            self._write_checkpoint(
                destination,
                BenchmarkPhase.CREATED,
                samples,
                results,
                failures,
                load_measurement,
            )
        except Exception as exc:
            failures.append(Failure.from_exception(FailureStage.PERSIST, exc))
            checkpoint_available = False
            ready = False

        try:
            if ready:
                phase = BenchmarkPhase.PREFLIGHT
                checkpoint_available = self._checkpoint_or_record_failure(
                    destination,
                    BenchmarkPhase.PREFLIGHT,
                    samples,
                    results,
                    failures,
                    load_measurement,
                )
                ready = checkpoint_available

            if ready:
                try:
                    manifest.preflight(verify_hashes=self._verify_hashes)
                except Exception as exc:
                    failures.append(Failure.from_exception(FailureStage.PREFLIGHT, exc))
                    ready = False

            if ready:
                phase = BenchmarkPhase.LOADING
                checkpoint_available = self._checkpoint_or_record_failure(
                    destination,
                    BenchmarkPhase.LOADING,
                    samples,
                    results,
                    failures,
                    load_measurement,
                )
                ready = checkpoint_available

            if ready:
                lifecycle_started = True
                try:
                    _, load_measurement = measure_call(self._backend.load)
                except Exception as exc:
                    failures.append(Failure.from_exception(FailureStage.LOAD, exc))
                    ready = False

            if ready:
                phase = BenchmarkPhase.WARMING_UP
                checkpoint_available = self._checkpoint_or_record_failure(
                    destination,
                    BenchmarkPhase.WARMING_UP,
                    samples,
                    results,
                    failures,
                    load_measurement,
                )
                ready = checkpoint_available

            if ready:
                for _ in range(self._warmup_runs):
                    try:
                        self._backend.warmup()
                    except Exception as exc:
                        failures.append(Failure.from_exception(FailureStage.WARMUP, exc))
                        ready = False
                        break
                    checkpoint_available = self._checkpoint_or_record_failure(
                        destination,
                        BenchmarkPhase.WARMING_UP,
                        samples,
                        results,
                        failures,
                        load_measurement,
                    )
                    if not checkpoint_available:
                        ready = False
                        break

            if ready:
                phase = BenchmarkPhase.TRANSCRIBING
                checkpoint_available = self._checkpoint_or_record_failure(
                    destination,
                    BenchmarkPhase.TRANSCRIBING,
                    samples,
                    results,
                    failures,
                    load_measurement,
                )
                ready = checkpoint_available

            if ready:
                for sample in samples:
                    active_sample_id = sample.sample_id
                    checkpoint_available = self._checkpoint_or_record_failure(
                        destination,
                        BenchmarkPhase.TRANSCRIBING,
                        samples,
                        results,
                        failures,
                        load_measurement,
                        active_sample_id=active_sample_id,
                    )
                    if not checkpoint_available:
                        ready = False
                        break
                    try:
                        audio_path = sample.resolved_audio_path(manifest.dataset_root)
                        transcript, measurement = measure_call(
                            partial(self._backend.transcribe, audio_path)
                        )
                        if not isinstance(transcript, Transcript):
                            raise TypeError("ASR backend must return a Transcript")
                    except Exception as exc:
                        failure = Failure.from_exception(
                            FailureStage.TRANSCRIBE,
                            exc,
                            sample_id=sample.sample_id,
                        )
                        failures.append(failure)
                        results.append(
                            SampleResult(
                                sample_id=sample.sample_id,
                                status=SampleStatus.FAILED,
                                failure=failure,
                            )
                        )
                    else:
                        results.append(
                            SampleResult(
                                sample_id=sample.sample_id,
                                status=SampleStatus.OK,
                                transcript=transcript,
                                measurement=measurement,
                            )
                        )
                    active_sample_id = None
                    checkpoint_available = self._checkpoint_or_record_failure(
                        destination,
                        BenchmarkPhase.TRANSCRIBING,
                        samples,
                        results,
                        failures,
                        load_measurement,
                    )
                    if not checkpoint_available:
                        ready = False
                        break

            if ready:
                phase = BenchmarkPhase.SCORING
                checkpoint_available = self._checkpoint_or_record_failure(
                    destination,
                    BenchmarkPhase.SCORING,
                    samples,
                    results,
                    failures,
                    load_measurement,
                )
                ready = checkpoint_available

            if ready:
                try:
                    report = self._scorer(samples, results)
                    scoring = ScoreResult(ScoreStatus.OK, report=report)
                except Exception as exc:
                    failure = Failure.from_exception(FailureStage.SCORE, exc)
                    failures.append(failure)
                    scoring = ScoreResult(ScoreStatus.FAILED, failure=failure)
        except BaseException as exc:
            interrupted = exc
            interruption_phase = phase
        finally:
            if lifecycle_started:
                if checkpoint_available:
                    phase = BenchmarkPhase.CLOSING
                    checkpoint_available = self._checkpoint_or_record_failure(
                        destination,
                        BenchmarkPhase.CLOSING,
                        samples,
                        results,
                        failures,
                        load_measurement,
                    )
                try:
                    self._backend.close()
                except BaseException as exc:
                    if isinstance(exc, Exception):
                        failures.append(Failure.from_exception(FailureStage.CLOSE, exc))
                    elif interrupted is None:
                        interrupted = exc
                        interruption_phase = BenchmarkPhase.CLOSING

        if interrupted is not None:
            if checkpoint_available:
                self._checkpoint_or_record_failure(
                    destination,
                    BenchmarkPhase.INTERRUPTED,
                    samples,
                    results,
                    failures,
                    load_measurement,
                    interruption=interrupted,
                    interrupted_phase=interruption_phase,
                    active_sample_id=active_sample_id,
                )
            raise interrupted.with_traceback(interrupted.__traceback__)

        result = self._result(destination, samples, results, scoring, failures, load_measurement)
        try:
            write_artifact(destination / "result.json", result)
        except Exception as exc:
            failures.append(Failure.from_exception(FailureStage.PERSIST, exc))
            result = self._result(
                destination, samples, results, scoring, failures, load_measurement
            )
            try:
                write_artifact(destination / "result.json", result)
            except Exception:
                pass
        if checkpoint_available:
            checkpoint_available = self._checkpoint_or_record_failure(
                destination,
                BenchmarkPhase.FINISHED,
                samples,
                results,
                failures,
                load_measurement,
                scoring=scoring,
            )
            if not checkpoint_available:
                result = self._result(
                    destination, samples, results, scoring, failures, load_measurement
                )
                try:
                    write_artifact(destination / "result.json", result)
                except Exception:
                    pass
        return result

    def _checkpoint_or_record_failure(
        self,
        destination: Path,
        phase: BenchmarkPhase,
        samples: Sequence[AudioSample],
        results: Sequence[SampleResult],
        failures: list[Failure],
        load_measurement: ResourceMeasurement | None,
        *,
        interruption: BaseException | None = None,
        interrupted_phase: BenchmarkPhase | None = None,
        active_sample_id: str | None = None,
        scoring: ScoreResult | None = None,
    ) -> bool:
        try:
            self._write_checkpoint(
                destination,
                phase,
                samples,
                results,
                failures,
                load_measurement,
                interruption=interruption,
                interrupted_phase=interrupted_phase,
                active_sample_id=active_sample_id,
                scoring=scoring,
            )
        except Exception as exc:
            failures.append(Failure.from_exception(FailureStage.PERSIST, exc))
            return False
        return True

    def _write_checkpoint(
        self,
        destination: Path,
        phase: BenchmarkPhase,
        samples: Sequence[AudioSample],
        results: Sequence[SampleResult],
        failures: Sequence[Failure],
        load_measurement: ResourceMeasurement | None,
        *,
        interruption: BaseException | None = None,
        interrupted_phase: BenchmarkPhase | None = None,
        active_sample_id: str | None = None,
        scoring: ScoreResult | None = None,
    ) -> None:
        checkpoint = BenchmarkCheckpoint(
            backend=self._backend.descriptor,
            phase=phase,
            expected_samples=len(samples),
            sample_results=tuple(results),
            scoring=scoring or ScoreResult(ScoreStatus.SKIPPED),
            failures=tuple(failures),
            load_measurement=load_measurement,
            updated_at=datetime.now(UTC).isoformat(),
            active_sample_id=active_sample_id,
            interrupted_phase=interrupted_phase,
            interruption_type=type(interruption).__name__ if interruption is not None else None,
            interruption_message=str(interruption) if interruption is not None else None,
        )
        write_artifact(destination / "checkpoint.json", checkpoint)

    def _result(
        self,
        output_dir: Path,
        samples: Sequence[AudioSample],
        sample_results: Sequence[SampleResult],
        scoring: ScoreResult,
        failures: Sequence[Failure],
        load_measurement: ResourceMeasurement | None,
    ) -> BenchmarkResult:
        succeeded = sum(result.status is SampleStatus.OK for result in sample_results)
        score_succeeded = scoring.status is ScoreStatus.OK
        if (
            len(sample_results) == len(samples)
            and succeeded == len(samples)
            and score_succeeded
            and not failures
            and len(samples) > 0
        ):
            status = RunStatus.COMPLETE
        elif succeeded > 0:
            status = RunStatus.PARTIAL
        else:
            status = RunStatus.FAILED
        return BenchmarkResult(
            backend=self._backend.descriptor,
            status=status,
            output_dir=output_dir,
            expected_samples=len(samples),
            sample_results=tuple(sample_results),
            performance=self._performance(sample_results, load_measurement),
            scoring=scoring,
            failures=tuple(failures),
        )

    @staticmethod
    def _performance(
        sample_results: Sequence[SampleResult],
        load_measurement: ResourceMeasurement | None,
    ) -> PerformanceSummary:
        measured = tuple(
            result
            for result in sample_results
            if result.transcript is not None and result.measurement is not None
        )
        inference_seconds = sum(
            result.measurement.elapsed_seconds
            for result in measured
            if result.measurement is not None
        )
        audio_seconds = sum(
            result.transcript.audio_duration_seconds
            for result in measured
            if result.transcript is not None
            and result.transcript.audio_duration_seconds is not None
        )
        rss_values = [
            result.measurement.process_rss_peak_bytes
            for result in measured
            if result.measurement is not None
            and result.measurement.process_rss_peak_bytes is not None
        ]
        tree_rss_values = [
            result.measurement.process_tree_rss_peak_bytes
            for result in measured
            if result.measurement is not None
            and result.measurement.process_tree_rss_peak_bytes is not None
        ]
        allocated_values = [
            result.transcript.gpu_peak_allocated_bytes
            for result in measured
            if result.transcript is not None
            and result.transcript.gpu_peak_allocated_bytes is not None
        ]
        reserved_values = [
            result.transcript.gpu_peak_reserved_bytes
            for result in measured
            if result.transcript is not None
            and result.transcript.gpu_peak_reserved_bytes is not None
        ]
        return PerformanceSummary(
            model_load=load_measurement,
            measured_samples=len(measured),
            total_inference_seconds=inference_seconds,
            total_audio_seconds=audio_seconds,
            real_time_factor=inference_seconds / audio_seconds if audio_seconds else None,
            process_rss_peak_bytes=max(rss_values) if rss_values else None,
            process_tree_rss_peak_bytes=max(tree_rss_values) if tree_rss_values else None,
            gpu_peak_allocated_bytes=max(allocated_values) if allocated_values else None,
            gpu_peak_reserved_bytes=max(reserved_values) if reserved_values else None,
            measurement_scope=(
                "backend transcribe call including adapter audio loading; excludes model load, "
                "warmup, scoring, and artifact writes; host and host-plus-descendant RSS are "
                "reported separately"
            ),
        )
