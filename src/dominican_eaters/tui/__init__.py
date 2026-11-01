"""OpenTUI launcher for discovering and running CLI workflows."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

from dominican_eaters.evaluation.asr.suite import DEFAULT_T4_PRESETS
from dominican_eaters.speech.asr.registry import MODEL_PRESETS, RUNTIME_SPECS

from .commands import discover_worker_python


class TUIUnavailableError(RuntimeError):
    """Raised when the optional terminal-interface dependency is unavailable."""


_TMUX_SESSION_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
_TUI_BOOTSTRAP_ENV = "DOMINICAN_EATERS_TUI_BOOTSTRAP"
_MINIMUM_NODE_VERSION = (26, 4, 0)


def _manifest_sample_count(path: Path) -> int | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    samples = raw.get("samples")
    return len(samples) if isinstance(samples, list) and samples else None


def _discover_stt_datasets(project_root: Path) -> list[dict[str, object]]:
    """Return stable TUI dataset choices from canonical STT manifests."""

    manifests = project_root / "data" / "manifests"
    full_manifests = sorted(manifests.glob("stt-*.json"))
    if not full_manifests:
        smoke = manifests / "stt.json"
        if smoke.is_file():
            full_manifests = [smoke]
        else:
            full_manifests = [manifests / "stt-all.json"]

    datasets: list[dict[str, object]] = []
    seen: set[str] = set()
    for path in full_manifests:
        if path.name == "stt-all.json":
            dataset_id = "music"
            label = "Music / collected lyrics"
        elif path.name == "stt.json":
            dataset_id = "smoke"
            label = "Smoke test"
        else:
            dataset_id = path.stem.removeprefix("stt-")
            label = dataset_id.replace("-", " ").replace("_", " ").title()
        if not dataset_id or dataset_id in seen:
            continue
        seen.add(dataset_id)
        value: dict[str, object] = {
            "id": dataset_id,
            "label": label,
            "manifest": path.relative_to(project_root).as_posix(),
        }
        sample_count = _manifest_sample_count(path)
        if sample_count is not None:
            value["sampleCount"] = sample_count
        datasets.append(value)
    return datasets


def _bootstrap_from_environment() -> tuple[dict[str, object], bool]:
    raw = os.environ.get(_TUI_BOOTSTRAP_ENV)
    if raw is None:
        return {}, False
    try:
        bootstrap = json.loads(raw)
    except json.JSONDecodeError as error:
        raise TUIUnavailableError("TUI bootstrap configuration is not valid JSON.") from error
    if not isinstance(bootstrap, dict):
        raise TUIUnavailableError("TUI bootstrap configuration must be a JSON object.")
    initial_state = bootstrap.get("initialState", {})
    auto_run = bootstrap.get("autoRun", False)
    if not isinstance(initial_state, dict) or not isinstance(auto_run, bool):
        raise TUIUnavailableError("TUI bootstrap initialState or autoRun has an invalid type.")
    return initial_state, auto_run


def run_tui(
    *,
    initial_state: Mapping[str, object] | None = None,
    auto_run: bool | None = None,
) -> None:
    """Start the OpenTUI application on its native Zig renderer."""

    node, project_root, entrypoint = _require_tui_runtime()
    environment_initial_state, environment_auto_run = _bootstrap_from_environment()
    selected_initial_state = dict(
        environment_initial_state if initial_state is None else initial_state
    )
    selected_auto_run = environment_auto_run if auto_run is None else auto_run
    presets = []
    for preset in MODEL_PRESETS.values():
        runtime = RUNTIME_SPECS[preset.runtime_id]
        presets.append(
            {
                "id": preset.preset_id,
                "label": preset.label,
                "status": preset.status,
                "reason": preset.reason or "Ready to run",
                "runnable": preset.runnable,
                "execution": runtime.execution,
                "precision": preset.precision or "auto",
                "devices": list(preset.capabilities.supported_devices),
                "workerPython": discover_worker_python(runtime_id=preset.runtime_id),
            }
        )
    environment = {
        **os.environ,
        "NODE_NO_WARNINGS": "1",
        "DOMINICAN_EATERS_TUI_DATA": json.dumps(
            {
                "python": sys.executable,
                "presets": presets,
                "datasets": _discover_stt_datasets(project_root),
                "suitePresets": list(DEFAULT_T4_PRESETS),
                "initialState": selected_initial_state,
                "autoRun": selected_auto_run,
            },
            ensure_ascii=False,
        ),
    }
    result = subprocess.run(
        [node, "--experimental-ffi", "--experimental-strip-types", str(entrypoint)],
        cwd=project_root,
        env=environment,
        check=False,
    )
    if result.returncode:
        raise TUIUnavailableError(f"OpenTUI exited with status {result.returncode}.")


def _require_supported_node() -> str:
    node = shutil.which("node")
    if node is None:
        raise TUIUnavailableError("OpenTUI requires Node.js 26.4 or newer.")
    result = subprocess.run(
        [node, "--version"],
        check=False,
        capture_output=True,
        text=True,
    )
    raw_version = result.stdout.strip()
    match = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)", raw_version)
    if result.returncode or match is None:
        raise TUIUnavailableError(
            "Could not determine the Node.js version. OpenTUI requires Node.js 26.4 or newer."
        )
    version = tuple(int(part) for part in match.groups())
    if version < _MINIMUM_NODE_VERSION:
        raise TUIUnavailableError(f"OpenTUI requires Node.js 26.4 or newer; found {raw_version}.")
    return node


def _require_tui_runtime() -> tuple[str, Path, Path]:
    node = _require_supported_node()
    project_root = Path(__file__).resolve().parents[3]
    entrypoint = Path(__file__).with_name("opentui") / "app.ts"
    core = project_root / "node_modules" / "@opentui" / "core"
    if not core.is_dir():
        raise TUIUnavailableError(
            "OpenTUI dependencies are missing. Run `npm install` in the project directory."
        )
    return node, project_root, entrypoint


def launch_tui_in_tmux(
    *,
    session_name: str,
    initial_state: Mapping[str, object],
    auto_run: bool,
) -> None:
    """Open an initialized TUI in a persistent tmux session.

    An existing session is attached without replacing its running workflow.
    Calls from another tmux client switch to the target session.
    """

    if _TMUX_SESSION_PATTERN.fullmatch(session_name) is None:
        raise TUIUnavailableError(
            "tmux session names must use 1-64 letters, numbers, dots, dashes, or underscores."
        )
    tmux = shutil.which("tmux")
    if tmux is None:
        raise TUIUnavailableError("This command requires tmux. Install tmux and try again.")

    target_session = f"={session_name}"
    session_check = subprocess.run(
        [tmux, "has-session", "-t", target_session],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if session_check.returncode not in {0, 1}:
        raise TUIUnavailableError(
            f"tmux could not inspect session {session_name!r} (status {session_check.returncode})."
        )

    session_exists = session_check.returncode == 0
    if session_exists:
        print(
            f"tmux session {session_name!r} already exists; "
            "its current configuration remains active."
        )

    project_root = Path(__file__).resolve().parents[3]
    if not session_exists:
        _, project_root, _ = _require_tui_runtime()
        bootstrap = json.dumps(
            {"initialState": dict(initial_state), "autoRun": auto_run},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        create_result = subprocess.run(
            [
                tmux,
                "new-session",
                "-d",
                "-s",
                session_name,
                "-c",
                str(project_root),
                "-e",
                f"{_TUI_BOOTSTRAP_ENV}={bootstrap}",
                sys.executable,
                "-m",
                "dominican_eaters",
                "tui",
            ],
            check=False,
        )
        if create_result.returncode:
            raise TUIUnavailableError(
                f"tmux could not create session {session_name!r} "
                f"(status {create_result.returncode})."
            )

    action = "switch-client" if os.environ.get("TMUX") else "attach-session"
    attach_result = subprocess.run(
        [tmux, action, "-t", target_session],
        check=False,
    )
    if attach_result.returncode:
        raise TUIUnavailableError(
            f"tmux could not {action} session {session_name!r} (status {attach_result.returncode})."
        )


__all__ = ["TUIUnavailableError", "launch_tui_in_tmux", "run_tui"]
