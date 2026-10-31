from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from dominican_eaters.tui import _discover_stt_datasets, run_tui


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
    assert metadata["datasets"]


def test_launcher_discovers_full_stt_datasets_and_counts_samples(tmp_path: Path) -> None:
    manifests = tmp_path / "data" / "manifests"
    manifests.mkdir(parents=True)
    (manifests / "stt.json").write_text(
        json.dumps({"samples": [{"sample_id": "smoke"}]}), encoding="utf-8"
    )
    (manifests / "stt-all.json").write_text(
        json.dumps({"samples": [{"sample_id": "one"}, {"sample_id": "two"}]}),
        encoding="utf-8",
    )
    (manifests / "stt-speech.json").write_text(
        json.dumps({"samples": [{"sample_id": "speech"}]}), encoding="utf-8"
    )

    assert _discover_stt_datasets(tmp_path) == [
        {
            "id": "music",
            "label": "Music / collected lyrics",
            "manifest": "data/manifests/stt-all.json",
            "sampleCount": 2,
        },
        {
            "id": "speech",
            "label": "Speech",
            "manifest": "data/manifests/stt-speech.json",
            "sampleCount": 1,
        },
    ]


def test_launcher_uses_smoke_or_music_fallback(tmp_path: Path) -> None:
    manifests = tmp_path / "data" / "manifests"
    manifests.mkdir(parents=True)
    smoke = manifests / "stt.json"
    smoke.write_text(json.dumps({"samples": [{"sample_id": "smoke"}]}), encoding="utf-8")
    assert _discover_stt_datasets(tmp_path)[0]["id"] == "smoke"

    smoke.unlink()
    assert _discover_stt_datasets(tmp_path) == [
        {
            "id": "music",
            "label": "Music / collected lyrics",
            "manifest": "data/manifests/stt-all.json",
        }
    ]
