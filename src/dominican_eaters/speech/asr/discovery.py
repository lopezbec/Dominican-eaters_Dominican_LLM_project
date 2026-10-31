"""Discovery of isolated Python interpreters for ASR worker runtimes."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping
from pathlib import Path

from .registry import RUNTIME_SPECS

WORKER_PYTHON_ENV_VARS = (
    "DOMINICAN_EATERS_WORKER_PYTHON",
    "NEMO_WORKER_PYTHON",
)

RUNTIME_VENV_DIRS = {
    "nemo-worker": (".venvs/nemo", ".venv-nemo"),
    "granite-4.1-transformers": (".venvs/granite", ".venv-granite"),
    "granite-3.3-transformers": (".venvs/granite", ".venv-granite"),
    "qwen3-asr-transformers": (".venvs/qwen3-asr", ".venv-qwen3-asr"),
    "voxtral-transformers": (".venvs/voxtral", ".venv-voxtral"),
    "qwen2-audio-transformers": (".venvs/qwen2-audio", ".venv-qwen2-audio"),
}


def _runtime_environment_hints(runtime_id: str) -> tuple[str, ...]:
    base = runtime_id.removesuffix("-transformers").removesuffix("-worker")
    values = {base, base.replace("-asr", ""), base.replace("-", "_")}
    return tuple(value for value in values if value)


def _python_provides_module(python: Path, module: str) -> bool:
    probe = f"import importlib.util,sys;sys.exit(0 if importlib.util.find_spec({module!r}) else 1)"
    try:
        return (
            subprocess.run(
                [python, "-c", probe],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=3,
                check=False,
            ).returncode
            == 0
        )
    except (OSError, subprocess.TimeoutExpired):
        return False


def discover_worker_python(
    *,
    runtime_id: str = "nemo-worker",
    cwd: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Find the interpreter configured for one isolated ASR runtime."""

    environment = os.environ if environ is None else environ
    root = Path.cwd() if cwd is None else cwd
    try:
        runtime = RUNTIME_SPECS[runtime_id]
    except KeyError as error:
        raise ValueError(f"Unknown ASR runtime: {runtime_id}") from error
    if runtime.execution == "inline":
        return ""
    runtime_variables = (
        (runtime.interpreter_env_var,) if runtime.interpreter_env_var is not None else ()
    )
    candidates = [
        Path(environment[name]).expanduser()
        for name in (*runtime_variables, *WORKER_PYTHON_ENV_VARS)
        if environment.get(name)
    ]
    candidates.extend(
        root / directory / "bin" / "python" for directory in RUNTIME_VENV_DIRS.get(runtime_id, ())
    )

    hints = _runtime_environment_hints(runtime_id)
    discovered = (*root.glob(".venv*/bin/python"), *root.glob(".venvs/*/bin/python"))
    candidates.extend(
        candidate
        for candidate in discovered
        if any(hint in str(candidate.parent.parent).lower() for hint in hints)
    )

    active_environment = environment.get("VIRTUAL_ENV", "")
    runtime_hint = runtime_id.split("-")[0]
    if runtime_hint in Path(active_environment).name.lower():
        candidates.append(Path(active_environment) / "bin" / "python")

    for candidate in candidates:
        absolute = Path(os.path.abspath(candidate))
        if absolute.is_file() and os.access(absolute, os.X_OK):
            return str(absolute)

    if runtime.worker_module is not None:
        generic_candidates = (
            root / ".venv" / "bin" / "python",
            *root.glob(".venv*/bin/python"),
            *root.glob(".venvs/*/bin/python"),
        )
        for candidate in generic_candidates:
            absolute = Path(os.path.abspath(candidate))
            if (
                absolute.is_file()
                and os.access(absolute, os.X_OK)
                and _python_provides_module(absolute, runtime.worker_module)
            ):
                return str(absolute)
    return ""


__all__ = ["RUNTIME_VENV_DIRS", "WORKER_PYTHON_ENV_VARS", "discover_worker_python"]
