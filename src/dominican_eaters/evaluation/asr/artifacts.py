"""Atomic JSON artifacts for canonical ASR evaluation runs."""

from __future__ import annotations

import platform
import shutil
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

import psutil

from dominican_eaters.data import ArtifactSerializationError, atomic_write_json, to_json_value
from dominican_eaters.speech.asr import BackendDescriptor


@dataclass(frozen=True, slots=True)
class GPUDeviceSnapshot:
    """Publication-safe identity and capacity for one visible NVIDIA GPU."""

    index: int
    name: str
    memory_total_bytes: int
    driver_version: str

    def __post_init__(self) -> None:
        if type(self.index) is not int or self.index < 0:
            raise ValueError("GPU index must be a nonnegative integer")
        if type(self.memory_total_bytes) is not int or self.memory_total_bytes < 0:
            raise ValueError("GPU memory must be a nonnegative integer")
        for name in ("name", "driver_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"GPU {name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class HostSnapshot:
    """Lightweight host facts required to interpret benchmark measurements."""

    operating_system: str
    operating_system_release: str
    architecture: str
    python_version: str
    logical_cpu_count: int | None
    ram_total_bytes: int
    gpu_devices: tuple[GPUDeviceSnapshot, ...]
    cuda_runtime_version: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "operating_system",
            "operating_system_release",
            "architecture",
            "python_version",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"host {name} must be a non-empty string")
        if self.logical_cpu_count is not None and (
            type(self.logical_cpu_count) is not int or self.logical_cpu_count <= 0
        ):
            raise ValueError("host logical_cpu_count must be a positive integer or None")
        if type(self.ram_total_bytes) is not int or self.ram_total_bytes <= 0:
            raise ValueError("host ram_total_bytes must be a positive integer")
        if not isinstance(self.gpu_devices, tuple) or not all(
            isinstance(device, GPUDeviceSnapshot) for device in self.gpu_devices
        ):
            raise TypeError("host gpu_devices must be a tuple of GPUDeviceSnapshot values")
        if self.cuda_runtime_version is not None and (
            not isinstance(self.cuda_runtime_version, str) or not self.cuda_runtime_version.strip()
        ):
            raise ValueError("host cuda_runtime_version must be a non-empty string or None")


@dataclass(frozen=True, slots=True)
class ArtifactProvenance:
    """Resolved publication identity for a benchmark result or checkpoint."""

    preset_id: str | None
    runtime_id: str | None
    model_revision_requested: str | None
    model_revision_resolved: str | None
    quantization: str | None
    prompt_template_id: str | None
    audio_preprocessing: dict[str, object] | None
    environment_lock_id: str | None
    telemetry_source: str | None
    host: HostSnapshot

    def __post_init__(self) -> None:
        for name in (
            "preset_id",
            "runtime_id",
            "model_revision_requested",
            "model_revision_resolved",
            "quantization",
            "prompt_template_id",
            "environment_lock_id",
            "telemetry_source",
        ):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"provenance {name} must be a non-empty string or None")
        if self.audio_preprocessing is not None and not isinstance(self.audio_preprocessing, dict):
            raise TypeError("provenance audio_preprocessing must be a dictionary or None")
        if not isinstance(self.host, HostSnapshot):
            raise TypeError("provenance host must be a HostSnapshot")

    @classmethod
    def from_backend(
        cls,
        descriptor: BackendDescriptor,
        host: HostSnapshot,
    ) -> ArtifactProvenance:
        """Consolidate v2 provenance while retaining legacy revision semantics."""

        cuda_version = _cuda_runtime_version(descriptor.runtime_versions)
        effective_host = (
            replace(host, cuda_runtime_version=cuda_version)
            if host.cuda_runtime_version is None and cuda_version is not None
            else host
        )
        return cls(
            preset_id=descriptor.preset_id,
            runtime_id=descriptor.runtime_id,
            model_revision_requested=descriptor.model_revision_requested,
            model_revision_resolved=descriptor.resolved_model_revision,
            quantization=descriptor.quantization,
            prompt_template_id=descriptor.prompt_template_id,
            audio_preprocessing=(
                None
                if descriptor.audio_preprocessing is None
                else dict(descriptor.audio_preprocessing)
            ),
            environment_lock_id=descriptor.environment_lock_id,
            telemetry_source=descriptor.telemetry_source,
            host=effective_host,
        )


def capture_host_snapshot() -> HostSnapshot:
    """Capture bounded local facts without importing a model runtime."""

    return HostSnapshot(
        operating_system=platform.system() or "unknown",
        operating_system_release=platform.release() or "unknown",
        architecture=platform.machine() or "unknown",
        python_version=platform.python_version(),
        logical_cpu_count=psutil.cpu_count(logical=True),
        ram_total_bytes=psutil.virtual_memory().total,
        gpu_devices=_nvidia_gpu_snapshots(),
    )


def _nvidia_gpu_snapshots() -> tuple[GPUDeviceSnapshot, ...]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return ()
    try:
        completed = subprocess.run(
            (
                executable,
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ),
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    if completed.returncode != 0:
        return ()

    devices: list[GPUDeviceSnapshot] = []
    for line in completed.stdout.splitlines():
        fields = tuple(part.strip() for part in line.split(",", maxsplit=3))
        if len(fields) != 4:
            continue
        try:
            index = int(fields[0])
            memory_total_bytes = int(fields[2]) * 1024 * 1024
        except ValueError:
            continue
        if not fields[1] or not fields[3] or index < 0 or memory_total_bytes < 0:
            continue
        devices.append(
            GPUDeviceSnapshot(
                index=index,
                name=fields[1],
                memory_total_bytes=memory_total_bytes,
                driver_version=fields[3],
            )
        )
    return tuple(devices)


def _cuda_runtime_version(runtime_versions: dict[str, str]) -> str | None:
    for key in ("cuda", "cuda_runtime", "cuda_version", "torch_cuda"):
        value = runtime_versions.get(key)
        if value and value != "unknown":
            return value
    return None


def write_artifact(path: Path, value: object) -> None:
    payload = to_json_value(value)
    if not isinstance(payload, dict):
        raise ArtifactSerializationError("artifact root must be an object")
    atomic_write_json(path, payload)
