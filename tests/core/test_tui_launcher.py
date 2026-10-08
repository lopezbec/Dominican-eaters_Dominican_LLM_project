from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from dominican_eaters.tui import run_tui


def test_launcher_passes_registry_metadata_to_opentui() -> None:
    with patch("dominican_eaters.tui.subprocess.run") as run:
        run.return_value.returncode = 0
        run_tui()
    command = run.call_args.args[0]
    metadata = json.loads(run.call_args.kwargs["env"]["DOMINICAN_EATERS_TUI_DATA"])
    assert command[1:3] == ["--experimental-ffi", "--experimental-strip-types"]
    assert Path(command[3]).name == "app.ts"
    assert metadata["python"]
    assert any(item["id"] == "whisper-base" for item in metadata["presets"])
