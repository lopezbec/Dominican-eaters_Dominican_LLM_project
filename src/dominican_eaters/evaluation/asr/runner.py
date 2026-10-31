"""Dependency-injected orchestration for reproducible ASR benchmarks."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import Any, Protocol, cast

from dominican_eaters.data import AudioSample, STTManifest, load_manifest, write_manifest
from dominican_eaters.evaluation.asr.artifacts import (
    ArtifactProvenance,
    GPUDeviceSnapshot,
    HostSnapshot,
    capture_host_snapshot,
    write_artifact,
)
from dominican_eaters.evaluation.asr.scoring import (
    ASREvaluationReport,
    Recognition,
    ReferenceSample,
    evaluate_asr,
)
from dominican_eaters.evaluation.resources import ResourceMeasurement, measure_call
from dominican_eaters.speech.asr import (
    ASRBackend,
    BackendDescriptor,
    Transcript,
    WorkerRemoteError,
)

BENCHMARK_RESULT_SCHEMA_VERSION = 2
BENCHMARK_CHECKPOINT_SCHEMA_VERSION = 2
BENCHMARK_EVENT_SCHEMA_VERSION = 1


class OutputCollisionError(FileExistsError):
    """Raised before model work when the requested output path already exists."""


class ResumeConflictError(ValueError):
    """Raised when an existing checkpoint cannot safely continue this run."""


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


class BenchmarkEventKind(StrEnum):
    """Stable lifecycle events exposed to interactive and automation clients."""

    RUN_STARTED = "run_started"
    RUN_RESUMED = "run_resumed"
    PHASE_STARTED = "phase_started"
    SAMPLE_STARTED = "sample_started"
    SAMPLE_COMPLETED = "sample_completed"
    SAMPLE_RESTORED = "sample_restored"
    RUN_FINISHED = "run_finished"
    RUN_INTERRUPTED = "run_interrupted"


@dataclass(frozen=True, slots=True)
class Failure:
    stage: FailureStage
    error_type: str
    message: str
    sample_id: str | None = None
    error_code: str | None = None
    retryable: bool | None = None

    @classmethod
    def from_exception(
        cls,
        stage: FailureStage,
        exception: Exception,
        *,
        sample_id: str | None = None,
    ) -> Failure:
        remote_error = exception if isinstance(exception, WorkerRemoteError) else None
        return cls(
            stage=stage,
            error_type=type(exception).__name__,
            message=str(exception),
            sample_id=sample_id,
            error_code=None if remote_error is None else remote_error.code,
            retryable=None if remote_error is None else remote_error.retryable,
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
class BenchmarkEvent:
    """One structured progress update emitted synchronously by a benchmark run."""

    kind: BenchmarkEventKind
    phase: BenchmarkPhase
    dataset: str
    preset: str | None
    model: str
    backend_id: str
    completed: int
    succeeded: int
    failed: int
    total: int
    index: int | None = None
    sample_id: str | None = None
    audio_path: str | None = None
    group_id: str | None = None
    split: str | None = None
    source: str | None = None
    status: SampleStatus | None = None
    elapsed_seconds: float | None = None
    audio_seconds: float | None = None
    error_type: str | None = None
    message: str | None = None
    error_code: str | None = None
    retryable: bool | None = None
    run_status: RunStatus | None = None
    schema_version: int = BENCHMARK_EVENT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        """Return the versioned JSON-safe representation consumed by the TUI."""

        return {
            "schema_version": self.schema_version,
            "event": self.kind.value,
            "phase": self.phase.value,
            "dataset": self.dataset,
            "preset": self.preset,
            "model": self.model,
            "backend_id": self.backend_id,
            "completed": self.completed,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "total": self.total,
            "index": self.index,
            "sample_id": self.sample_id,
            "audio_path": self.audio_path,
            "group_id": self.group_id,
            "split": self.split,
            "source": self.source,
            "status": self.status.value if self.status is not None else None,
            "elapsed_seconds": self.elapsed_seconds,
            "audio_seconds": self.audio_seconds,
            "error_type": self.error_type,
            "message": self.message,
            "error_code": self.error_code,
            "retryable": self.retryable,
            "run_status": self.run_status.value if self.run_status is not None else None,
        }


BenchmarkEventCallback = Callable[[BenchmarkEvent], None]


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
    provenance: ArtifactProvenance
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
    provenance: ArtifactProvenance
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


@dataclass(frozen=True, slots=True)
class _ResumeState:
    checkpoint: BenchmarkCheckpoint
    results: tuple[SampleResult, ...]
    failures: tuple[Failure, ...]


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
        host_snapshot: HostSnapshot | None = None,
        on_event: BenchmarkEventCallback | None = None,
    ) -> None:
        if warmup_runs < 0:
            raise ValueError("warmup_runs must be nonnegative")
        self._backend = backend
        self._scorer = scorer
        self._warmup_runs = warmup_runs
        self._verify_hashes = verify_hashes
        self._host_snapshot = host_snapshot
        self._on_event = on_event

    def run(
        self,
        manifest: STTManifest,
        output_dir: Path,
        *,
        resume: bool = False,
    ) -> BenchmarkResult:
        """Run a benchmark, optionally continuing a strictly compatible checkpoint.

        A fresh run still rejects every existing path. Resume mode may create a
        new destination, or continue an incomplete destination whose frozen
        manifest, backend identity, host, and completed sample prefix match.
        """

        destination = output_dir.resolve()
        samples = tuple(manifest.samples)
        dataset = str(manifest.dataset_root)
        resume_state: _ResumeState | None = None
        if destination.exists():
            if not resume:
                raise OutputCollisionError(f"Benchmark output already exists: {destination}")
            resume_state = self._load_resume_state(manifest, destination)
        else:
            destination.mkdir(parents=True, exist_ok=False)

        current_host = self._host_snapshot or capture_host_snapshot()
        if resume_state is not None:
            stored_host = resume_state.checkpoint.provenance.host
            _validate_resume_host(current_host, stored_host)
            # Preserve one provenance identity across all measurements in the run.
            self._host_snapshot = stored_host
        else:
            self._host_snapshot = current_host

        failures = list(resume_state.failures) if resume_state is not None else []
        results = list(resume_state.results) if resume_state is not None else []
        scoring = ScoreResult(ScoreStatus.SKIPPED)
        load_measurement: ResourceMeasurement | None = None
        lifecycle_started = False
        ready = True
        checkpoint_available = True
        interrupted: BaseException | None = None
        interruption_phase: BenchmarkPhase | None = None
        phase = BenchmarkPhase.CREATED
        active_sample_id: str | None = None

        if resume_state is None:
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
                self._emit_event(
                    BenchmarkEventKind.RUN_STARTED,
                    BenchmarkPhase.CREATED,
                    dataset,
                    samples,
                    results,
                )
            except Exception as exc:
                failures.append(Failure.from_exception(FailureStage.PERSIST, exc))
                checkpoint_available = False
                ready = False
        else:
            self._emit_event(
                BenchmarkEventKind.RUN_RESUMED,
                resume_state.checkpoint.phase,
                dataset,
                samples,
                results,
            )
            for sample_index, sample_result in enumerate(results, start=1):
                self._emit_event(
                    BenchmarkEventKind.SAMPLE_RESTORED,
                    BenchmarkPhase.TRANSCRIBING,
                    dataset,
                    samples,
                    results[:sample_index],
                    sample=samples[sample_index - 1],
                    sample_index=sample_index,
                    sample_result=sample_result,
                )

        try:
            if ready:
                phase = BenchmarkPhase.PREFLIGHT
                if resume_state is None:
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
                    self._emit_event(
                        BenchmarkEventKind.PHASE_STARTED,
                        BenchmarkPhase.PREFLIGHT,
                        dataset,
                        samples,
                        results,
                    )

            if ready:
                try:
                    manifest.preflight(verify_hashes=self._verify_hashes)
                except Exception as exc:
                    failures.append(Failure.from_exception(FailureStage.PREFLIGHT, exc))
                    ready = False

            if ready:
                phase = BenchmarkPhase.LOADING
                if resume_state is None:
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
                    self._emit_event(
                        BenchmarkEventKind.PHASE_STARTED,
                        BenchmarkPhase.LOADING,
                        dataset,
                        samples,
                        results,
                    )

            if ready:
                lifecycle_started = True
                try:
                    _, load_measurement = measure_call(self._backend.load)
                    if resume_state is not None and resume_state.results:
                        _validate_resolved_backend(
                            self._backend.descriptor,
                            resume_state.checkpoint.backend,
                        )
                except ResumeConflictError:
                    raise
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
                    self._emit_event(
                        BenchmarkEventKind.PHASE_STARTED,
                        BenchmarkPhase.WARMING_UP,
                        dataset,
                        samples,
                        results,
                    )

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
                    self._emit_event(
                        BenchmarkEventKind.PHASE_STARTED,
                        BenchmarkPhase.TRANSCRIBING,
                        dataset,
                        samples,
                        results,
                    )

            if ready:
                completed_count = len(results)
                for sample_index, sample in enumerate(
                    samples[completed_count:], start=completed_count + 1
                ):
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
                    self._emit_event(
                        BenchmarkEventKind.SAMPLE_STARTED,
                        BenchmarkPhase.TRANSCRIBING,
                        dataset,
                        samples,
                        results,
                        sample=sample,
                        sample_index=sample_index,
                    )
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
                        sample_result = SampleResult(
                            sample_id=sample.sample_id,
                            status=SampleStatus.FAILED,
                            failure=failure,
                        )
                    else:
                        sample_result = SampleResult(
                            sample_id=sample.sample_id,
                            status=SampleStatus.OK,
                            transcript=transcript,
                            measurement=measurement,
                        )
                    results.append(sample_result)
                    self._emit_event(
                        BenchmarkEventKind.SAMPLE_COMPLETED,
                        BenchmarkPhase.TRANSCRIBING,
                        dataset,
                        samples,
                        results,
                        sample=sample,
                        sample_index=sample_index,
                        sample_result=sample_result,
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
                    self._emit_event(
                        BenchmarkEventKind.PHASE_STARTED,
                        BenchmarkPhase.SCORING,
                        dataset,
                        samples,
                        results,
                    )

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
            if isinstance(exc, ResumeConflictError):
                # Keep the last known-good checkpoint identity intact.
                checkpoint_available = False
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
                    if checkpoint_available:
                        self._emit_event(
                            BenchmarkEventKind.PHASE_STARTED,
                            BenchmarkPhase.CLOSING,
                            dataset,
                            samples,
                            results,
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
            active_sample = next(
                (sample for sample in samples if sample.sample_id == active_sample_id), None
            )
            self._emit_event(
                BenchmarkEventKind.RUN_INTERRUPTED,
                BenchmarkPhase.INTERRUPTED,
                dataset,
                samples,
                results,
                sample=active_sample,
                sample_index=(
                    samples.index(active_sample) + 1 if active_sample is not None else None
                ),
                error=interrupted,
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
        self._emit_event(
            BenchmarkEventKind.RUN_FINISHED,
            BenchmarkPhase.FINISHED,
            dataset,
            samples,
            results,
            reported_failure=failures[-1] if failures else None,
            run_status=result.status,
        )
        return result

    def _load_resume_state(self, manifest: STTManifest, destination: Path) -> _ResumeState:
        if not destination.is_dir():
            raise ResumeConflictError(f"Benchmark resume path is not a directory: {destination}")
        if (destination / "result.json").exists():
            raise ResumeConflictError("Benchmark already has a final result and cannot be resumed")
        try:
            frozen_manifest = load_manifest(destination / "manifest.json")
        except Exception as exc:
            raise ResumeConflictError(f"Could not load frozen benchmark manifest: {exc}") from exc
        if frozen_manifest != manifest:
            raise ResumeConflictError(
                "Benchmark manifest does not match the frozen resume manifest"
            )
        checkpoint = _load_benchmark_checkpoint(destination / "checkpoint.json")
        if checkpoint.phase is BenchmarkPhase.FINISHED:
            raise ResumeConflictError("Finished benchmark checkpoint cannot be resumed")
        if checkpoint.expected_samples != len(manifest):
            raise ResumeConflictError("Checkpoint sample count does not match the manifest")
        if checkpoint.scoring.status is not ScoreStatus.SKIPPED:
            raise ResumeConflictError("Incomplete checkpoint contains finalized scoring")
        _validate_configured_backend(self._backend.descriptor, checkpoint.backend)

        results = checkpoint.sample_results
        expected_prefix = tuple(sample.sample_id for sample in manifest.samples[: len(results)])
        actual_prefix = tuple(result.sample_id for result in results)
        if actual_prefix != expected_prefix:
            raise ResumeConflictError(
                "Checkpoint results are not an ordered prefix of the frozen manifest"
            )
        next_sample_id = (
            manifest.samples[len(results)].sample_id if len(results) < len(manifest) else None
        )
        if checkpoint.active_sample_id not in {None, next_sample_id}:
            raise ResumeConflictError(
                "Checkpoint active sample is not the next uncommitted manifest sample"
            )
        failed_results = tuple(result.failure for result in results if result.failure is not None)
        for failure in failed_results:
            if failure not in checkpoint.failures:
                raise ResumeConflictError(
                    "Checkpoint failures do not include every failed sample result"
                )
        return _ResumeState(checkpoint, results, checkpoint.failures)

    def _emit_event(
        self,
        kind: BenchmarkEventKind,
        phase: BenchmarkPhase,
        dataset: str,
        samples: Sequence[AudioSample],
        results: Sequence[SampleResult],
        *,
        sample: AudioSample | None = None,
        sample_index: int | None = None,
        sample_result: SampleResult | None = None,
        reported_failure: Failure | None = None,
        error: BaseException | None = None,
        run_status: RunStatus | None = None,
    ) -> None:
        if self._on_event is None:
            return
        succeeded = sum(result.status is SampleStatus.OK for result in results)
        failed = sum(result.status is SampleStatus.FAILED for result in results)
        failure = sample_result.failure if sample_result is not None else reported_failure
        measurement = sample_result.measurement if sample_result is not None else None
        transcript = sample_result.transcript if sample_result is not None else None
        descriptor = self._backend.descriptor
        self._on_event(
            BenchmarkEvent(
                kind=kind,
                phase=phase,
                dataset=dataset,
                preset=descriptor.preset_id,
                model=descriptor.model,
                backend_id=descriptor.backend_id,
                completed=len(results),
                succeeded=succeeded,
                failed=failed,
                total=len(samples),
                index=sample_index,
                sample_id=sample.sample_id if sample is not None else None,
                audio_path=sample.audio_path if sample is not None else None,
                group_id=sample.group_id if sample is not None else None,
                split=sample.split if sample is not None else None,
                source=sample.source if sample is not None else None,
                status=sample_result.status if sample_result is not None else None,
                elapsed_seconds=(measurement.elapsed_seconds if measurement is not None else None),
                audio_seconds=(
                    transcript.audio_duration_seconds if transcript is not None else None
                ),
                error_type=(
                    failure.error_type
                    if failure is not None
                    else type(error).__name__
                    if error is not None
                    else None
                ),
                message=(
                    failure.message
                    if failure is not None
                    else str(error)
                    if error is not None
                    else None
                ),
                error_code=failure.error_code if failure is not None else None,
                retryable=failure.retryable if failure is not None else None,
                run_status=run_status,
            )
        )

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
            provenance=self._provenance(),
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
            provenance=self._provenance(),
            status=status,
            output_dir=output_dir,
            expected_samples=len(samples),
            sample_results=tuple(sample_results),
            performance=self._performance(sample_results, load_measurement),
            scoring=scoring,
            failures=tuple(failures),
        )

    def _provenance(self) -> ArtifactProvenance:
        host = self._host_snapshot
        if host is None:
            raise RuntimeError("benchmark host snapshot has not been captured")
        return ArtifactProvenance.from_backend(self._backend.descriptor, host)

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


def _load_benchmark_checkpoint(path: Path) -> BenchmarkCheckpoint:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ResumeConflictError(f"Could not read benchmark checkpoint {path}: {exc}") from exc
    raw = _resume_mapping(value, "checkpoint")
    _resume_fields(
        raw,
        {
            "backend",
            "provenance",
            "phase",
            "expected_samples",
            "sample_results",
            "scoring",
            "failures",
            "load_measurement",
            "updated_at",
            "active_sample_id",
            "interrupted_phase",
            "interruption_type",
            "interruption_message",
            "schema_version",
        },
        "checkpoint",
    )
    schema_version = _resume_int(raw["schema_version"], "checkpoint.schema_version")
    if schema_version != BENCHMARK_CHECKPOINT_SCHEMA_VERSION:
        raise ResumeConflictError(f"Unsupported benchmark checkpoint schema: {schema_version}")
    phase = _resume_enum(BenchmarkPhase, raw["phase"], "checkpoint.phase")
    if phase is BenchmarkPhase.FINISHED:
        raise ResumeConflictError("Finished benchmark checkpoint cannot be resumed")
    results_raw = _resume_array(raw["sample_results"], "checkpoint.sample_results")
    failures_raw = _resume_array(raw["failures"], "checkpoint.failures")
    scoring_raw = _resume_mapping(raw["scoring"], "checkpoint.scoring")
    _resume_fields(scoring_raw, {"status", "report", "failure"}, "checkpoint.scoring")
    score_status = _resume_enum(ScoreStatus, scoring_raw["status"], "checkpoint.scoring.status")
    if score_status is not ScoreStatus.SKIPPED:
        raise ResumeConflictError("Incomplete checkpoint contains finalized scoring")
    if scoring_raw["report"] is not None or scoring_raw["failure"] is not None:
        raise ResumeConflictError("Skipped checkpoint scoring must not contain payloads")
    interrupted_phase_raw = raw["interrupted_phase"]
    return BenchmarkCheckpoint(
        backend=_resume_backend(raw["backend"]),
        provenance=_resume_provenance(raw["provenance"]),
        phase=phase,
        expected_samples=_resume_int(raw["expected_samples"], "checkpoint.expected_samples"),
        sample_results=tuple(
            _resume_sample_result(item, index) for index, item in enumerate(results_raw)
        ),
        scoring=ScoreResult(ScoreStatus.SKIPPED),
        failures=tuple(_resume_failure(item, "checkpoint.failures") for item in failures_raw),
        load_measurement=_resume_measurement(
            raw["load_measurement"], "checkpoint.load_measurement"
        ),
        updated_at=_resume_string(raw["updated_at"], "checkpoint.updated_at"),
        active_sample_id=_resume_optional_string(
            raw["active_sample_id"], "checkpoint.active_sample_id"
        ),
        interrupted_phase=(
            None
            if interrupted_phase_raw is None
            else _resume_enum(BenchmarkPhase, interrupted_phase_raw, "checkpoint.interrupted_phase")
        ),
        interruption_type=_resume_optional_string(
            raw["interruption_type"], "checkpoint.interruption_type"
        ),
        interruption_message=_resume_optional_string(
            raw["interruption_message"],
            "checkpoint.interruption_message",
            allow_empty=True,
        ),
        schema_version=schema_version,
    )


def _resume_backend(value: object) -> BackendDescriptor:
    raw = _resume_mapping(value, "checkpoint.backend")
    fields = {
        "backend_id",
        "model",
        "model_revision",
        "language",
        "requested_device",
        "requested_precision",
        "effective_device",
        "effective_precision",
        "runtime_versions",
        "options",
        "preset_id",
        "runtime_id",
        "model_revision_requested",
        "model_revision_resolved",
        "quantization",
        "prompt_template_id",
        "audio_preprocessing",
        "environment_lock_id",
        "telemetry_source",
    }
    _resume_fields(raw, fields, "checkpoint.backend")
    versions = _resume_mapping(raw["runtime_versions"], "checkpoint.backend.runtime_versions")
    if not all(isinstance(key, str) and isinstance(item, str) for key, item in versions.items()):
        raise ResumeConflictError("checkpoint backend runtime versions must contain strings")
    options = _resume_mapping(raw["options"], "checkpoint.backend.options")
    preprocessing_raw = raw["audio_preprocessing"]
    preprocessing = (
        None
        if preprocessing_raw is None
        else dict(_resume_mapping(preprocessing_raw, "checkpoint.backend.audio_preprocessing"))
    )
    return BackendDescriptor(
        backend_id=_resume_string(raw["backend_id"], "checkpoint.backend.backend_id"),
        model=_resume_string(raw["model"], "checkpoint.backend.model"),
        model_revision=_resume_optional_string(
            raw["model_revision"], "checkpoint.backend.model_revision"
        ),
        language=_resume_string(raw["language"], "checkpoint.backend.language"),
        requested_device=_resume_string(
            raw["requested_device"], "checkpoint.backend.requested_device"
        ),
        requested_precision=_resume_string(
            raw["requested_precision"], "checkpoint.backend.requested_precision"
        ),
        effective_device=_resume_optional_string(
            raw["effective_device"], "checkpoint.backend.effective_device"
        ),
        effective_precision=_resume_optional_string(
            raw["effective_precision"], "checkpoint.backend.effective_precision"
        ),
        runtime_versions=cast(dict[str, str], dict(versions)),
        options=dict(options),
        preset_id=_resume_optional_string(raw["preset_id"], "checkpoint.backend.preset_id"),
        runtime_id=_resume_optional_string(raw["runtime_id"], "checkpoint.backend.runtime_id"),
        model_revision_requested=_resume_optional_string(
            raw["model_revision_requested"], "checkpoint.backend.model_revision_requested"
        ),
        model_revision_resolved=_resume_optional_string(
            raw["model_revision_resolved"], "checkpoint.backend.model_revision_resolved"
        ),
        quantization=_resume_optional_string(
            raw["quantization"], "checkpoint.backend.quantization"
        ),
        prompt_template_id=_resume_optional_string(
            raw["prompt_template_id"], "checkpoint.backend.prompt_template_id"
        ),
        audio_preprocessing=preprocessing,
        environment_lock_id=_resume_optional_string(
            raw["environment_lock_id"], "checkpoint.backend.environment_lock_id"
        ),
        telemetry_source=_resume_optional_string(
            raw["telemetry_source"], "checkpoint.backend.telemetry_source"
        ),
    )


def _resume_provenance(value: object) -> ArtifactProvenance:
    raw = _resume_mapping(value, "checkpoint.provenance")
    fields = {
        "preset_id",
        "runtime_id",
        "model_revision_requested",
        "model_revision_resolved",
        "quantization",
        "prompt_template_id",
        "audio_preprocessing",
        "environment_lock_id",
        "telemetry_source",
        "host",
    }
    _resume_fields(raw, fields, "checkpoint.provenance")
    preprocessing_raw = raw["audio_preprocessing"]
    return ArtifactProvenance(
        preset_id=_resume_optional_string(raw["preset_id"], "checkpoint.provenance.preset_id"),
        runtime_id=_resume_optional_string(raw["runtime_id"], "checkpoint.provenance.runtime_id"),
        model_revision_requested=_resume_optional_string(
            raw["model_revision_requested"], "checkpoint.provenance.model_revision_requested"
        ),
        model_revision_resolved=_resume_optional_string(
            raw["model_revision_resolved"], "checkpoint.provenance.model_revision_resolved"
        ),
        quantization=_resume_optional_string(
            raw["quantization"], "checkpoint.provenance.quantization"
        ),
        prompt_template_id=_resume_optional_string(
            raw["prompt_template_id"], "checkpoint.provenance.prompt_template_id"
        ),
        audio_preprocessing=(
            None
            if preprocessing_raw is None
            else dict(
                _resume_mapping(preprocessing_raw, "checkpoint.provenance.audio_preprocessing")
            )
        ),
        environment_lock_id=_resume_optional_string(
            raw["environment_lock_id"], "checkpoint.provenance.environment_lock_id"
        ),
        telemetry_source=_resume_optional_string(
            raw["telemetry_source"], "checkpoint.provenance.telemetry_source"
        ),
        host=_resume_host(raw["host"]),
    )


def _resume_host(value: object) -> HostSnapshot:
    raw = _resume_mapping(value, "checkpoint.provenance.host")
    _resume_fields(
        raw,
        {
            "operating_system",
            "operating_system_release",
            "architecture",
            "python_version",
            "logical_cpu_count",
            "ram_total_bytes",
            "gpu_devices",
            "cuda_runtime_version",
        },
        "checkpoint.provenance.host",
    )
    devices = _resume_array(raw["gpu_devices"], "checkpoint.provenance.host.gpu_devices")
    parsed_devices: list[GPUDeviceSnapshot] = []
    for index, value_item in enumerate(devices):
        item = _resume_mapping(value_item, f"checkpoint.provenance.host.gpu_devices[{index}]")
        _resume_fields(
            item,
            {"index", "name", "memory_total_bytes", "driver_version"},
            f"checkpoint.provenance.host.gpu_devices[{index}]",
        )
        parsed_devices.append(
            GPUDeviceSnapshot(
                index=_resume_int(item["index"], "GPU index"),
                name=_resume_string(item["name"], "GPU name"),
                memory_total_bytes=_resume_int(item["memory_total_bytes"], "GPU memory"),
                driver_version=_resume_string(item["driver_version"], "GPU driver"),
            )
        )
    logical_cpus_raw = raw["logical_cpu_count"]
    return HostSnapshot(
        operating_system=_resume_string(raw["operating_system"], "host operating_system"),
        operating_system_release=_resume_string(
            raw["operating_system_release"], "host operating_system_release"
        ),
        architecture=_resume_string(raw["architecture"], "host architecture"),
        python_version=_resume_string(raw["python_version"], "host python_version"),
        logical_cpu_count=(
            None
            if logical_cpus_raw is None
            else _resume_int(logical_cpus_raw, "host logical_cpu_count")
        ),
        ram_total_bytes=_resume_int(raw["ram_total_bytes"], "host ram_total_bytes"),
        gpu_devices=tuple(parsed_devices),
        cuda_runtime_version=_resume_optional_string(
            raw["cuda_runtime_version"], "host cuda_runtime_version"
        ),
    )


def _resume_sample_result(value: object, index: int) -> SampleResult:
    context = f"checkpoint.sample_results[{index}]"
    raw = _resume_mapping(value, context)
    _resume_fields(raw, {"sample_id", "status", "transcript", "measurement", "failure"}, context)
    status = _resume_enum(SampleStatus, raw["status"], f"{context}.status")
    transcript = (
        None if raw["transcript"] is None else _resume_transcript(raw["transcript"], context)
    )
    failure = None if raw["failure"] is None else _resume_failure(raw["failure"], context)
    return SampleResult(
        sample_id=_resume_string(raw["sample_id"], f"{context}.sample_id"),
        status=status,
        transcript=transcript,
        measurement=_resume_measurement(raw["measurement"], f"{context}.measurement"),
        failure=failure,
    )


def _resume_transcript(value: object, context: str) -> Transcript:
    raw = _resume_mapping(value, f"{context}.transcript")
    _resume_fields(
        raw,
        {
            "text",
            "language",
            "audio_duration_seconds",
            "gpu_peak_allocated_bytes",
            "gpu_peak_reserved_bytes",
            "metadata",
        },
        f"{context}.transcript",
    )
    return Transcript(
        text=_resume_string(raw["text"], f"{context}.transcript.text", allow_empty=True),
        language=_resume_optional_string(raw["language"], f"{context}.transcript.language"),
        audio_duration_seconds=_resume_optional_number(
            raw["audio_duration_seconds"], f"{context}.transcript.audio_duration_seconds"
        ),
        gpu_peak_allocated_bytes=_resume_optional_int(
            raw["gpu_peak_allocated_bytes"], f"{context}.transcript.gpu_peak_allocated_bytes"
        ),
        gpu_peak_reserved_bytes=_resume_optional_int(
            raw["gpu_peak_reserved_bytes"], f"{context}.transcript.gpu_peak_reserved_bytes"
        ),
        metadata=dict(_resume_mapping(raw["metadata"], f"{context}.transcript.metadata")),
    )


def _resume_measurement(value: object, context: str) -> ResourceMeasurement | None:
    if value is None:
        return None
    raw = _resume_mapping(value, context)
    _resume_fields(
        raw,
        {
            "elapsed_seconds",
            "process_rss_peak_bytes",
            "process_tree_rss_peak_bytes",
            "rss_sampling_interval_seconds",
            "rss_sampling_error",
        },
        context,
    )
    return ResourceMeasurement(
        elapsed_seconds=_resume_number(raw["elapsed_seconds"], f"{context}.elapsed_seconds"),
        process_rss_peak_bytes=_resume_optional_int(
            raw["process_rss_peak_bytes"], f"{context}.process_rss_peak_bytes"
        ),
        process_tree_rss_peak_bytes=_resume_optional_int(
            raw["process_tree_rss_peak_bytes"], f"{context}.process_tree_rss_peak_bytes"
        ),
        rss_sampling_interval_seconds=_resume_number(
            raw["rss_sampling_interval_seconds"], f"{context}.rss_sampling_interval_seconds"
        ),
        rss_sampling_error=_resume_optional_string(
            raw["rss_sampling_error"], f"{context}.rss_sampling_error"
        ),
    )


def _resume_failure(value: object, context: str) -> Failure:
    raw = _resume_mapping(value, context)
    _resume_fields(
        raw,
        {"stage", "error_type", "message", "sample_id", "error_code", "retryable"},
        context,
    )
    retryable = raw["retryable"]
    if retryable is not None and type(retryable) is not bool:
        raise ResumeConflictError(f"{context}.retryable must be boolean or null")
    return Failure(
        stage=_resume_enum(FailureStage, raw["stage"], f"{context}.stage"),
        error_type=_resume_string(raw["error_type"], f"{context}.error_type"),
        message=_resume_string(raw["message"], f"{context}.message", allow_empty=True),
        sample_id=_resume_optional_string(raw["sample_id"], f"{context}.sample_id"),
        error_code=_resume_optional_string(raw["error_code"], f"{context}.error_code"),
        retryable=retryable,
    )


def _validate_configured_backend(current: BackendDescriptor, stored: BackendDescriptor) -> None:
    fields = ("backend_id", "model", "language", "requested_device", "requested_precision")
    mismatched = [name for name in fields if getattr(current, name) != getattr(stored, name)]
    for name in ("preset_id", "runtime_id", "model_revision_requested", "quantization"):
        current_value = getattr(current, name)
        if current_value is not None and current_value != getattr(stored, name):
            mismatched.append(name)
    if mismatched:
        raise ResumeConflictError(
            "Checkpoint backend does not match the configured backend: " + ", ".join(mismatched)
        )


def _validate_resolved_backend(current: BackendDescriptor, stored: BackendDescriptor) -> None:
    if current != stored:
        raise ResumeConflictError(
            "Loaded backend identity differs from the checkpoint; refusing to mix results"
        )


def _validate_resume_host(current: HostSnapshot, stored: HostSnapshot) -> None:
    if replace(current, cuda_runtime_version=None) != replace(stored, cuda_runtime_version=None):
        raise ResumeConflictError(
            "Current host differs from the checkpoint host; refusing to mix measurements"
        )


def _resume_mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ResumeConflictError(f"{context} must be an object with string keys")
    return cast(Mapping[str, Any], value)


def _resume_array(value: object, context: str) -> list[object]:
    if not isinstance(value, list):
        raise ResumeConflictError(f"{context} must be an array")
    return cast(list[object], value)


def _resume_fields(raw: Mapping[str, Any], expected: set[str], context: str) -> None:
    missing = sorted(expected - raw.keys())
    unknown = sorted(raw.keys() - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append("missing fields: " + ", ".join(missing))
        if unknown:
            details.append("unknown fields: " + ", ".join(unknown))
        raise ResumeConflictError(f"{context} has {'; '.join(details)}")


def _resume_string(value: object, context: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        requirement = "a string" if allow_empty else "a nonempty string"
        raise ResumeConflictError(f"{context} must be {requirement}")
    return value


def _resume_optional_string(
    value: object, context: str, *, allow_empty: bool = False
) -> str | None:
    if value is None:
        return None
    return _resume_string(value, context, allow_empty=allow_empty)


def _resume_int(value: object, context: str) -> int:
    if type(value) is not int:
        raise ResumeConflictError(f"{context} must be an integer")
    return value


def _resume_optional_int(value: object, context: str) -> int | None:
    if value is None:
        return None
    return _resume_int(value, context)


def _resume_number(value: object, context: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ResumeConflictError(f"{context} must be a number")
    return float(value)


def _resume_optional_number(value: object, context: str) -> float | None:
    if value is None:
        return None
    return _resume_number(value, context)


def _resume_enum(enum_type: type[Any], value: object, context: str) -> Any:
    if not isinstance(value, str):
        raise ResumeConflictError(f"{context} must be a string")
    try:
        return enum_type(value)
    except ValueError as exc:
        raise ResumeConflictError(f"{context} has unsupported value {value!r}") from exc
