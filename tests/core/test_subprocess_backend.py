from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from collections.abc import Sequence
from pathlib import Path

import pytest

from dominican_eaters.speech.asr.subprocess_backend import (
    JsonlSubprocessBackend,
    SubprocessBackendSettings,
    WorkerProcessError,
    WorkerRemoteError,
    WorkerTimeoutError,
    _descriptor_from_payload,
    _SubprocessLineTransport,
)
from dominican_eaters.speech.asr.worker_protocol import (
    PreflightCheck,
    WorkerRequest,
    decode_request,
    encode_message,
    error_response,
    make_preflight_result,
    make_request,
    success_response,
)


class FakeTransport:
    def __init__(self) -> None:
        self.argv: tuple[str, ...] | None = None
        self.requests: list[WorkerRequest] = []
        self.closed = 0
        self.started = 0
        self.read_timeouts: list[float] = []
        self.close_timeouts: list[float] = []
        self.response_override: bytes | BaseException | None = None

    @property
    def stderr_tail(self) -> str:
        return "fake diagnostic"

    def start(self, argv: Sequence[str]) -> None:
        self.started += 1
        self.argv = tuple(argv)

    def write(self, frame: bytes) -> None:
        self.requests.append(decode_request(frame))

    def read(self, timeout_seconds: float) -> bytes:
        self.read_timeouts.append(timeout_seconds)
        if isinstance(self.response_override, BaseException):
            raise self.response_override
        if self.response_override is not None:
            return self.response_override
        request = self.requests[-1]
        if request.method == "preflight":
            payload = make_preflight_result(
                [
                    PreflightCheck(
                        "runtime",
                        "passed",
                        "error",
                        "runtime import succeeded",
                        {"module": "nemo.collections.asr"},
                    )
                ],
                environment={"python_version": "3.11", "lock_id": "nemo-test-lock"},
            )
        elif request.method == "describe":
            payload = {
                "descriptor": {
                    "backend_id": "nemo/parakeet",
                    "model": "parakeet",
                    "model_revision": "abc123",
                    "language": "es",
                    "requested_device": "auto",
                    "requested_precision": "auto",
                    "effective_device": "cuda",
                    "effective_precision": "fp16",
                    "runtime_versions": {"python": "3.11", "nemo_toolkit": "2.0"},
                    "options": {"batch_size": 1},
                }
            }
        elif request.method == "transcribe":
            payload = {
                "text": "Hola mundo",
                "language": "es",
                "audio_duration_seconds": 2.5,
                "gpu_peak_allocated_bytes": 1024,
                "gpu_peak_reserved_bytes": 2048,
                "metadata": {"rtf": 0.2},
            }
        else:
            payload = {}
        return encode_message(success_response(request, payload))

    def close(self, timeout_seconds: float) -> None:
        self.close_timeouts.append(timeout_seconds)
        self.closed += 1


def settings(tmp_path: Path) -> SubprocessBackendSettings:
    return SubprocessBackendSettings(
        interpreter=tmp_path / "venv" / "bin" / "python",
        worker_module="dominican_eaters_nemo",
        backend="parakeet",
        model="nvidia/parakeet-tdt-0.6b-v3",
    )


def test_preflight_sends_complete_v2_runtime_and_reuses_process_for_load(
    tmp_path: Path,
) -> None:
    transport = FakeTransport()
    request_ids = iter(("startup-0", "preflight-1", "load-2", "describe-3", "close-4"))
    backend = JsonlSubprocessBackend(
        SubprocessBackendSettings(
            interpreter=tmp_path / "venv" / "bin" / "python",
            worker_module="dominican_eaters_nemo",
            backend="parakeet",
            model="nvidia/parakeet-tdt-0.6b-v3",
            language="es",
            device="cuda",
            precision="fp16",
            options={"timestamps": True},
            preset="parakeet-tdt-0.6b-v3",
            model_revision="revision-1",
            quantization=None,
            prompt_template_id=None,
        ),
        transport_factory=lambda: transport,
        request_id_factory=lambda: next(request_ids),
    )

    report = backend.preflight()
    backend.load()
    backend.close()

    expected_runtime = {
        "backend": "parakeet",
        "preset": "parakeet-tdt-0.6b-v3",
        "model": "nvidia/parakeet-tdt-0.6b-v3",
        "model_revision": "revision-1",
        "language": "es",
        "device": "cuda",
        "precision": "fp16",
        "quantization": None,
        "prompt_template_id": None,
        "options": {"timestamps": True},
    }
    assert report.ready is True
    assert report.checks == (
        PreflightCheck(
            "runtime",
            "passed",
            "error",
            "runtime import succeeded",
            {"module": "nemo.collections.asr"},
        ),
    )
    assert report.environment == {"python_version": "3.11", "lock_id": "nemo-test-lock"}
    assert transport.started == 1
    assert [request.method for request in transport.requests] == [
        "describe",
        "preflight",
        "load",
        "describe",
        "close",
    ]
    assert transport.requests[1].params == expected_runtime
    assert transport.requests[2].params == expected_runtime
    assert transport.closed == 1


def test_preflight_only_closes_worker_through_protocol(tmp_path: Path) -> None:
    transport = FakeTransport()
    request_ids = iter(("startup-0", "preflight-1", "close-2"))
    backend = JsonlSubprocessBackend(
        settings(tmp_path),
        transport_factory=lambda: transport,
        request_id_factory=lambda: next(request_ids),
    )

    report = backend.preflight(close_after=True)
    backend.close()

    assert report.ready is True
    assert [request.method for request in transport.requests] == [
        "describe",
        "preflight",
        "close",
    ]
    assert transport.started == 1
    assert transport.closed == 1


def test_failed_preflight_is_structured_and_prevents_load(tmp_path: Path) -> None:
    class FailedPreflightTransport(FakeTransport):
        def read(self, timeout_seconds: float) -> bytes:
            if self.requests[-1].method != "preflight":
                return super().read(timeout_seconds)
            request = self.requests[-1]
            return encode_message(
                success_response(
                    request,
                    make_preflight_result(
                        [
                            PreflightCheck(
                                "cuda",
                                "failed",
                                "error",
                                "CUDA is unavailable",
                                {"requested_device": "cuda"},
                            )
                        ],
                        environment={"python_version": "3.11"},
                    ),
                )
            )

    transport = FailedPreflightTransport()
    backend = JsonlSubprocessBackend(settings(tmp_path), transport_factory=lambda: transport)

    report = backend.preflight()
    with pytest.raises(RuntimeError, match="preflight did not pass"):
        backend.load()
    backend.close()

    assert report.ready is False
    assert report.checks[0].name == "cuda"
    assert [request.method for request in transport.requests] == [
        "describe",
        "preflight",
        "close",
    ]


def test_malformed_preflight_response_aborts_worker(tmp_path: Path) -> None:
    class MalformedPreflightTransport(FakeTransport):
        def read(self, timeout_seconds: float) -> bytes:
            del timeout_seconds
            request = self.requests[-1]
            return (
                json.dumps(
                    {
                        "protocol_version": request.protocol_version,
                        "request_id": request.request_id,
                        "ok": True,
                        "result": {"ready": True, "checks": "invalid", "environment": {}},
                    }
                ).encode("utf-8")
                + b"\n"
            )

    transport = MalformedPreflightTransport()
    backend = JsonlSubprocessBackend(settings(tmp_path), transport_factory=lambda: transport)

    with pytest.raises(WorkerProcessError, match="checks must be an array"):
        backend.preflight()

    assert transport.closed == 1


def test_lifecycle_uses_list_argv_unique_ids_and_converts_transcript(tmp_path: Path) -> None:
    transport = FakeTransport()
    request_ids = iter(("startup-0", "load-1", "describe-2", "warmup-3", "transcribe-4", "close-5"))
    backend = JsonlSubprocessBackend(
        settings(tmp_path),
        transport_factory=lambda: transport,
        request_id_factory=lambda: next(request_ids),
    )

    backend.load()
    backend.warmup()
    transcript = backend.transcribe(tmp_path / "audio" / "sample.wav")
    backend.close()
    backend.close()

    assert transport.argv == (
        str(tmp_path / "venv" / "bin" / "python"),
        "-m",
        "dominican_eaters_nemo",
    )
    assert [request.request_id for request in transport.requests] == [
        "startup-0",
        "load-1",
        "describe-2",
        "warmup-3",
        "transcribe-4",
        "close-5",
    ]
    assert transcript.text == "Hola mundo"
    assert transcript.language == "es"
    assert transcript.audio_duration_seconds == 2.5
    assert transcript.gpu_peak_allocated_bytes == 1024
    assert transcript.gpu_peak_reserved_bytes == 2048
    assert transcript.metadata == {"rtf": 0.2}
    assert transport.requests[4].params["audio_path"] == str(
        (tmp_path / "audio" / "sample.wav").resolve()
    )
    assert backend.descriptor.effective_device == "cuda"
    assert backend.descriptor.model_revision == "abc123"
    assert transport.closed == 1


@pytest.mark.parametrize(
    "payload, message",
    [
        ({}, "missing fields: descriptor"),
        ({"backend_id": "unwrapped"}, "missing fields: descriptor"),
        ({"descriptor": {}}, "missing fields"),
        ({"descriptor": {}, "extra": True}, "unknown fields: extra"),
    ],
)
def test_descriptor_requires_the_single_canonical_shape(
    payload: dict[str, object], message: str
) -> None:
    with pytest.raises(WorkerProcessError, match=message):
        _descriptor_from_payload(payload)  # type: ignore[arg-type]


def test_transcript_requires_all_canonical_fields(tmp_path: Path) -> None:
    class MissingMetadataTransport(FakeTransport):
        def read(self, timeout_seconds: float) -> bytes:
            if self.requests[-1].method != "transcribe":
                return super().read(timeout_seconds)
            request = self.requests[-1]
            return encode_message(
                success_response(
                    request,
                    {
                        "text": "Hola mundo",
                        "language": "es",
                        "audio_duration_seconds": 2.5,
                        "gpu_peak_allocated_bytes": None,
                        "gpu_peak_reserved_bytes": None,
                    },
                )
            )

    transport = MissingMetadataTransport()
    backend = JsonlSubprocessBackend(settings(tmp_path), transport_factory=lambda: transport)
    backend.load()

    with pytest.raises(WorkerProcessError, match="missing fields: metadata"):
        backend.transcribe(tmp_path / "sample.wav")


def test_timeout_aborts_worker_and_close_remains_idempotent(tmp_path: Path) -> None:
    class LoadTimeoutTransport(FakeTransport):
        def read(self, timeout_seconds: float) -> bytes:
            if self.requests[-1].method == "load":
                self.read_timeouts.append(timeout_seconds)
                raise TimeoutError
            return super().read(timeout_seconds)

    transport = LoadTimeoutTransport()
    backend = JsonlSubprocessBackend(settings(tmp_path), transport_factory=lambda: transport)

    with pytest.raises(WorkerTimeoutError, match="load timed out"):
        backend.load()

    backend.close()
    assert transport.closed == 1


def test_each_worker_phase_uses_its_configured_timeout(tmp_path: Path) -> None:
    transport = FakeTransport()
    backend = JsonlSubprocessBackend(
        SubprocessBackendSettings(
            interpreter=tmp_path / "venv" / "bin" / "python",
            worker_module="dominican_eaters_nemo",
            backend="parakeet",
            model="nvidia/parakeet-tdt-0.6b-v3",
            request_timeout_seconds=99,
            startup_timeout_seconds=1,
            preflight_timeout_seconds=2,
            load_timeout_seconds=3,
            inference_timeout_seconds=4,
            shutdown_timeout_seconds=5,
        ),
        transport_factory=lambda: transport,
    )

    backend.preflight()
    backend.load()
    backend.warmup()
    backend.transcribe(tmp_path / "sample.wav")
    backend.close()

    assert [request.method for request in transport.requests] == [
        "describe",
        "preflight",
        "load",
        "describe",
        "warmup",
        "transcribe",
        "close",
    ]
    assert transport.read_timeouts == [1, 2, 3, 3, 3, 4, 5]
    assert transport.close_timeouts == [5]


def test_legacy_request_timeout_remains_the_phase_fallback(tmp_path: Path) -> None:
    configured = SubprocessBackendSettings(
        interpreter=tmp_path / "venv" / "bin" / "python",
        worker_module="dominican_eaters_nemo",
        backend="parakeet",
        model="nvidia/parakeet-tdt-0.6b-v3",
        request_timeout_seconds=7.5,
    )

    assert configured.timeout_for("startup") == 7.5
    assert configured.timeout_for("preflight") == 7.5
    assert configured.timeout_for("load") == 7.5
    assert configured.timeout_for("inference") == 7.5


def test_startup_handshake_timeout_is_classified_and_aborts(tmp_path: Path) -> None:
    transport = FakeTransport()
    transport.response_override = TimeoutError()
    backend = JsonlSubprocessBackend(
        SubprocessBackendSettings(
            interpreter=tmp_path / "venv" / "bin" / "python",
            worker_module="dominican_eaters_nemo",
            backend="parakeet",
            model="nvidia/parakeet-tdt-0.6b-v3",
            startup_timeout_seconds=1.25,
        ),
        transport_factory=lambda: transport,
    )

    with pytest.raises(WorkerTimeoutError, match="startup timed out after 1.25 seconds"):
        backend.load()

    assert transport.closed == 1


@pytest.mark.parametrize(
    "field",
    [
        "request_timeout_seconds",
        "startup_timeout_seconds",
        "preflight_timeout_seconds",
        "load_timeout_seconds",
        "inference_timeout_seconds",
        "shutdown_timeout_seconds",
    ],
)
def test_settings_reject_non_positive_phase_timeouts(tmp_path: Path, field: str) -> None:
    kwargs = {field: 0.0}

    with pytest.raises(ValueError, match=field):
        SubprocessBackendSettings(
            interpreter=tmp_path / "venv" / "bin" / "python",
            worker_module="dominican_eaters_nemo",
            backend="parakeet",
            model="nvidia/parakeet-tdt-0.6b-v3",
            **kwargs,  # type: ignore[arg-type]
        )


def test_close_timeout_does_not_cleanup_transport_twice(tmp_path: Path) -> None:
    class CloseTimeoutTransport(FakeTransport):
        def read(self, timeout_seconds: float) -> bytes:
            if self.requests[-1].method == "close":
                raise TimeoutError
            return super().read(timeout_seconds)

    transport = CloseTimeoutTransport()
    backend = JsonlSubprocessBackend(settings(tmp_path), transport_factory=lambda: transport)
    backend.load()

    with pytest.raises(WorkerTimeoutError, match="close timed out"):
        backend.close()

    backend.close()
    assert transport.closed == 1


def test_rejects_mismatched_response_id_and_aborts(tmp_path: Path) -> None:
    transport = FakeTransport()
    transport.response_override = encode_message(
        success_response(make_request("warmup", request_id="wrong-id"))
    )
    backend = JsonlSubprocessBackend(
        settings(tmp_path),
        transport_factory=lambda: transport,
        request_id_factory=lambda: "expected-id",
    )

    with pytest.raises(WorkerProcessError, match="does not match"):
        backend.load()

    assert transport.closed == 1


def test_structured_worker_error_is_exposed_and_worker_is_cleaned_up(tmp_path: Path) -> None:
    transport = FakeTransport()
    transport.response_override = encode_message(
        error_response(
            make_request("warmup", request_id="load-id"),
            code="ModelUnavailable",
            message="model download disabled",
        )
    )
    backend = JsonlSubprocessBackend(
        settings(tmp_path),
        transport_factory=lambda: transport,
        request_id_factory=lambda: "load-id",
    )

    with pytest.raises(WorkerRemoteError, match="model download disabled") as caught:
        backend.load()

    assert caught.value.code == "ModelUnavailable"
    assert caught.value.error_type == "ModelUnavailable"
    assert caught.value.retryable is False
    assert caught.value.error.code == "ModelUnavailable"
    assert caught.value.error.message == "model download disabled"
    assert caught.value.error.retryable is False
    assert transport.closed == 1


def test_settings_require_absolute_interpreter() -> None:
    with pytest.raises(ValueError, match="absolute"):
        SubprocessBackendSettings(
            interpreter=Path(".venv/bin/python"),
            worker_module="worker",
            backend="parakeet",
            model="parakeet",
        )


def test_real_transport_launches_without_a_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeProcess:
        stdin = None
        stdout = None
        stderr = None

        def kill(self) -> None:
            captured["killed"] = True

    def fake_popen(argv: list[str], **kwargs: object) -> FakeProcess:
        captured["argv"] = argv
        captured.update(kwargs)
        return FakeProcess()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    transport = _SubprocessLineTransport()

    with pytest.raises(WorkerProcessError, match="pipes"):
        transport.start(("/usr/bin/python3", "-m", "worker"))

    assert captured["argv"] == ["/usr/bin/python3", "-m", "worker"]
    assert captured["shell"] is False
    assert captured["killed"] is True


def test_real_transport_reports_early_process_exit() -> None:
    stderr_chunks: list[str] = []
    transport = _SubprocessLineTransport(stderr_sink=stderr_chunks.append)
    transport.start(
        (
            sys.executable,
            "-c",
            "import sys, time; print('model loading', file=sys.stderr, flush=True); "
            "time.sleep(0.1); raise SystemExit(7)",
        )
    )

    with pytest.raises(WorkerProcessError, match="stderr: model loading"):
        transport.read(2.0)

    assert "".join(stderr_chunks) == "model loading\n"
    assert transport.stderr_tail == "model loading"
    transport.close(1.0)


def test_stderr_sink_failure_does_not_interrupt_protocol_transport() -> None:
    def broken_sink(message: str) -> None:
        raise RuntimeError(f"cannot display {message}")

    transport = _SubprocessLineTransport(stderr_sink=broken_sink)
    transport.start(
        (
            sys.executable,
            "-c",
            "import sys; print('diagnostic', file=sys.stderr, flush=True); "
            "print('protocol frame', flush=True)",
        )
    )

    assert transport.read(2.0) == b"protocol frame\n"
    transport.close(1.0)
    assert transport.stderr_tail == "diagnostic"


def test_real_process_completes_protocol_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = tmp_path / "fixture_worker.py"
    module.write_text(
        textwrap.dedent(
            """
            import sys
            from dominican_eaters.speech.asr.worker_protocol import (
                PreflightCheck, decode_request, encode_message,
                make_preflight_result, success_response,
            )

            print("fixture worker ready", file=sys.stderr, flush=True)
            for line in sys.stdin.buffer:
                request = decode_request(line)
                if request.method == "preflight":
                    result = make_preflight_result(
                        [PreflightCheck(
                            "runtime", "passed", "error", "runtime ready",
                        )],
                        environment={"python_version": sys.version.split()[0]},
                    )
                elif request.method == "describe":
                    result = {"descriptor": {
                        "backend_id": "fixture/model",
                        "model": "model",
                        "model_revision": "revision",
                        "language": "es",
                        "requested_device": "cpu",
                        "requested_precision": "fp32",
                        "effective_device": "cpu",
                        "effective_precision": "fp32",
                        "runtime_versions": {"fixture": "1"},
                        "options": {},
                    }}
                elif request.method == "transcribe":
                    result = {
                        "text": "hola",
                        "language": "es",
                        "audio_duration_seconds": 1.5,
                        "gpu_peak_allocated_bytes": None,
                        "gpu_peak_reserved_bytes": None,
                        "metadata": {},
                    }
                else:
                    result = {}
                sys.stdout.buffer.write(encode_message(success_response(request, result)))
                sys.stdout.buffer.flush()
                if request.method == "close":
                    break
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    stderr_chunks: list[str] = []
    backend = JsonlSubprocessBackend(
        SubprocessBackendSettings(
            interpreter=Path(sys.executable),
            worker_module="fixture_worker",
            backend="fixture",
            model="model",
            request_timeout_seconds=2,
        ),
        stderr_sink=stderr_chunks.append,
    )

    report = backend.preflight()
    backend.load()
    backend.warmup()
    transcript = backend.transcribe(tmp_path / "áudio sample.wav")
    backend.close()

    assert backend.descriptor.model_revision == "revision"
    assert report.ready is True
    assert report.checks[0].name == "runtime"
    assert transcript.text == "hola"
    assert "".join(stderr_chunks) == "fixture worker ready\n"
    assert transcript.audio_duration_seconds == 1.5
