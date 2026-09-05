from __future__ import annotations

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
    WorkerRequest,
    decode_request,
    encode_message,
    error_response,
    make_request,
    success_response,
)


class FakeTransport:
    def __init__(self) -> None:
        self.argv: tuple[str, ...] | None = None
        self.requests: list[WorkerRequest] = []
        self.closed = 0
        self.response_override: bytes | BaseException | None = None

    @property
    def stderr_tail(self) -> str:
        return "fake diagnostic"

    def start(self, argv: Sequence[str]) -> None:
        self.argv = tuple(argv)

    def write(self, frame: bytes) -> None:
        self.requests.append(decode_request(frame))

    def read(self, timeout_seconds: float) -> bytes:
        del timeout_seconds
        if isinstance(self.response_override, BaseException):
            raise self.response_override
        if self.response_override is not None:
            return self.response_override
        request = self.requests[-1]
        if request.method == "describe":
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
        del timeout_seconds
        self.closed += 1


def settings(tmp_path: Path) -> SubprocessBackendSettings:
    return SubprocessBackendSettings(
        interpreter=tmp_path / "venv" / "bin" / "python",
        worker_module="dominican_eaters_nemo",
        backend="parakeet",
        model="nvidia/parakeet-tdt-0.6b-v3",
    )


def test_lifecycle_uses_list_argv_unique_ids_and_converts_transcript(tmp_path: Path) -> None:
    transport = FakeTransport()
    request_ids = iter(("load-1", "describe-2", "warmup-3", "transcribe-4", "close-5"))
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
    assert transport.requests[3].params["audio_path"] == str(
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
    transport = FakeTransport()
    transport.response_override = TimeoutError()
    backend = JsonlSubprocessBackend(settings(tmp_path), transport_factory=lambda: transport)

    with pytest.raises(WorkerTimeoutError, match="load timed out"):
        backend.load()

    backend.close()
    assert transport.closed == 1


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

    assert caught.value.error_type == "ModelUnavailable"
    assert caught.value.retryable is False
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
    transport = _SubprocessLineTransport()
    transport.start((sys.executable, "-c", "raise SystemExit(7)"))

    with pytest.raises(WorkerProcessError, match="worker exited with code"):
        transport.read(2.0)

    transport.close(1.0)


def test_real_process_completes_protocol_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = tmp_path / "fixture_worker.py"
    module.write_text(
        textwrap.dedent(
            """
            import sys
            from dominican_eaters.speech.asr.worker_protocol import (
                decode_request, encode_message, success_response,
            )

            for line in sys.stdin.buffer:
                request = decode_request(line)
                if request.method == "describe":
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
    backend = JsonlSubprocessBackend(
        SubprocessBackendSettings(
            interpreter=Path(sys.executable),
            worker_module="fixture_worker",
            backend="fixture",
            model="model",
            request_timeout_seconds=2,
        )
    )

    backend.load()
    backend.warmup()
    transcript = backend.transcribe(tmp_path / "áudio sample.wav")
    backend.close()

    assert backend.descriptor.model_revision == "revision"
    assert transcript.text == "hola"
    assert transcript.audio_duration_seconds == 1.5
