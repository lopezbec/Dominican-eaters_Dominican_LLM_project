"""OpenTUI launcher for discovering and running CLI workflows."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from dominican_eaters.speech.asr.registry import MODEL_PRESETS, RUNTIME_SPECS

from .commands import discover_worker_python


class TUIUnavailableError(RuntimeError):
    """Raised when the optional terminal-interface dependency is unavailable."""


def run_tui() -> None:
    """Start the OpenTUI application on its native Zig renderer."""

    node = shutil.which("node")
    if node is None:
        raise TUIUnavailableError("OpenTUI requires Node.js 26.4 or newer.")
    project_root = Path(__file__).resolve().parents[3]
    entrypoint = Path(__file__).with_name("opentui") / "app.ts"
    core = project_root / "node_modules" / "@opentui" / "core"
    if not core.is_dir():
        raise TUIUnavailableError(
            "OpenTUI dependencies are missing. Run `npm install` in the project directory."
        )
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
            {"python": sys.executable, "presets": presets}, ensure_ascii=False
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


__all__ = ["TUIUnavailableError", "run_tui"]
