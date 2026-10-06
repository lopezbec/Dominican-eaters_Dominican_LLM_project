"""Side-effect-bounded environment checks for registered ASR runtimes."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .registry import BACKEND_SPECS, BackendName, RuntimeSpec

EnvironmentCheckKind = Literal["interpreter", "module", "executable", "cuda", "probe"]


@dataclass(frozen=True, slots=True)
class EnvironmentCheck:
    """One prerequisite checked without downloading or loading model weights."""

    kind: EnvironmentCheckKind
    name: str
    available: bool
    detail: str


@dataclass(frozen=True, slots=True)
class BackendEnvironmentReport:
    """Structured readiness result suitable for CLI and TUI rendering."""

    backend: BackendName
    interpreter: Path | None
    requested_device: str
    python_version: str | None
    checks: tuple[EnvironmentCheck, ...]

    @property
    def ready(self) -> bool:
        return all(check.available for check in self.checks)


_ENVIRONMENT_PROBE = """
import importlib.util
import json
import platform
import sys

modules = json.loads(sys.argv[1])
result = {"python_version": platform.python_version(), "modules": {}}
for module in modules:
    try:
        result["modules"][module] = importlib.util.find_spec(module) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        result["modules"][module] = False
if sys.argv[2] == "cuda":
    try:
        import torch
        result["cuda_available"] = bool(torch.cuda.is_available())
    except Exception as error:
        result["cuda_available"] = False
        result["cuda_error"] = f"{type(error).__name__}: {error}"
print(json.dumps(result, ensure_ascii=True))
"""


def preflight_asr_environment(
    *,
    backend: BackendName,
    worker_python: Path | None,
    requested_device: str,
    timeout_seconds: float = 15.0,
) -> BackendEnvironmentReport:
    """Check a backend's registered runtime without loading model weights."""

    runtime = BACKEND_SPECS[backend].runtime
    interpreter = Path(sys.executable) if runtime.execution == "inline" else worker_python
    return _probe_environment(
        backend=backend,
        runtime=runtime,
        interpreter=interpreter,
        requested_device=requested_device,
        timeout_seconds=timeout_seconds,
    )


def _probe_environment(
    *,
    backend: BackendName,
    runtime: RuntimeSpec,
    interpreter: Path | None,
    requested_device: str,
    timeout_seconds: float,
) -> BackendEnvironmentReport:
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if interpreter is None:
        detail = f"--worker-python is required for the {backend} backend"
        if runtime.interpreter_env_var:
            detail += f" ({runtime.interpreter_env_var} may provide it)"
        return BackendEnvironmentReport(
            backend=backend,
            interpreter=None,
            requested_device=requested_device,
            python_version=None,
            checks=(EnvironmentCheck("interpreter", "python", False, detail),),
        )

    # abspath deliberately preserves the selected virtualenv symlink. Path.resolve() could switch
    # to the base interpreter and make dependency discovery inspect the wrong environment.
    selected_interpreter = Path(os.path.abspath(interpreter.expanduser()))
    if not selected_interpreter.is_file() or not os.access(selected_interpreter, os.X_OK):
        return BackendEnvironmentReport(
            backend=backend,
            interpreter=selected_interpreter,
            requested_device=requested_device,
            python_version=None,
            checks=(
                EnvironmentCheck(
                    "interpreter",
                    "python",
                    False,
                    f"Python is not an executable file: {selected_interpreter}",
                ),
            ),
        )

    checks = [EnvironmentCheck("interpreter", "python", True, str(selected_interpreter))]
    for executable in runtime.required_executables:
        location = shutil.which(executable)
        checks.append(
            EnvironmentCheck(
                "executable",
                executable,
                location is not None,
                location or f"{executable} was not found on PATH",
            )
        )

    try:
        completed = subprocess.run(
            [
                str(selected_interpreter),
                "-c",
                _ENVIRONMENT_PROBE,
                json.dumps(runtime.required_modules),
                requested_device,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        checks.append(EnvironmentCheck("probe", "python", False, str(error)))
        return BackendEnvironmentReport(
            backend, selected_interpreter, requested_device, None, tuple(checks)
        )

    try:
        payload = json.loads(completed.stdout)
        if completed.returncode != 0 or not isinstance(payload, dict):
            raise ValueError("probe did not return a JSON object")
        python_version = payload.get("python_version")
        modules = payload.get("modules")
        if not isinstance(python_version, str) or not isinstance(modules, dict):
            raise ValueError("probe response is missing required fields")
    except (json.JSONDecodeError, ValueError) as error:
        detail = completed.stderr.strip() or completed.stdout.strip() or str(error)
        checks.append(EnvironmentCheck("probe", "python", False, detail))
        return BackendEnvironmentReport(
            backend, selected_interpreter, requested_device, None, tuple(checks)
        )

    for module in runtime.required_modules:
        available = modules.get(module) is True
        checks.append(
            EnvironmentCheck(
                "module",
                module,
                available,
                "available" if available else f"Python module {module!r} is unavailable",
            )
        )
    if requested_device == "cuda":
        cuda_available = payload.get("cuda_available") is True
        cuda_detail = payload.get("cuda_error")
        checks.append(
            EnvironmentCheck(
                "cuda",
                "torch.cuda",
                cuda_available,
                "available"
                if cuda_available
                else str(cuda_detail or "CUDA is unavailable in the selected Python"),
            )
        )
    return BackendEnvironmentReport(
        backend, selected_interpreter, requested_device, python_version, tuple(checks)
    )


__all__ = [
    "BackendEnvironmentReport",
    "EnvironmentCheck",
    "EnvironmentCheckKind",
    "preflight_asr_environment",
]
