from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from dominican_eaters.tui import (
    TUIUnavailableError,
    _discover_stt_datasets,
    launch_tui_in_tmux,
    run_tui,
)


def test_launcher_passes_registry_metadata_to_opentui() -> None:
    with (
        patch(
            "dominican_eaters.tui._require_tui_runtime",
            return_value=(
                "/usr/bin/node",
                Path(__file__).resolve().parents[2],
                Path("/package/tui/opentui/app.ts"),
            ),
        ),
        patch("dominican_eaters.tui.subprocess.run") as run,
    ):
        run.return_value.returncode = 0
        run_tui()
    command = run.call_args.args[0]
    metadata = json.loads(run.call_args.kwargs["env"]["DOMINICAN_EATERS_TUI_DATA"])
    assert command[1:3] == ["--experimental-ffi", "--experimental-strip-types"]
    assert Path(command[3]).name == "app.ts"
    assert metadata["python"]
    assert any(item["id"] == "whisper-base" for item in metadata["presets"])
    assert metadata["datasets"]
    assert metadata["suitePresets"] == [
        "whisper-large-v3",
        "parakeet-tdt-0.6b-v3",
        "canary-1b-v2",
        "granite-speech-4.1-2b",
        "qwen3-asr-1.7b",
        "voxtral-mini-3b-2507",
    ]
    assert metadata["initialState"] == {}
    assert metadata["autoRun"] is False


def test_launcher_passes_explicit_bootstrap_to_opentui() -> None:
    initial_state = {
        "workflow": "stt-suite",
        "output": "artifacts/stt-suite/t4-all-models",
    }
    with (
        patch(
            "dominican_eaters.tui._require_tui_runtime",
            return_value=(
                "/usr/bin/node",
                Path(__file__).resolve().parents[2],
                Path("/package/tui/opentui/app.ts"),
            ),
        ),
        patch("dominican_eaters.tui.subprocess.run") as run,
    ):
        run.return_value.returncode = 0
        run_tui(initial_state=initial_state, auto_run=True)

    metadata = json.loads(run.call_args.kwargs["env"]["DOMINICAN_EATERS_TUI_DATA"])
    assert metadata["initialState"] == initial_state
    assert metadata["autoRun"] is True


def test_tmux_launcher_creates_detached_session_then_attaches_with_bootstrap() -> None:
    initial_state = {
        "workflow": "stt-suite",
        "datasetIds": ["music"],
        "output": "artifacts/stt-suite/t4-all-models",
    }
    with (
        patch.dict("dominican_eaters.tui.os.environ", {}, clear=True),
        patch(
            "dominican_eaters.tui.shutil.which",
            side_effect=lambda name: f"/usr/bin/{name}",
        ),
        patch("dominican_eaters.tui.subprocess.run") as run,
    ):
        has_session = type("Result", (), {"returncode": 1})()
        node_version = type("Result", (), {"returncode": 0, "stdout": "v26.7.0\n"})()
        success = type("Result", (), {"returncode": 0})()
        run.side_effect = [has_session, node_version, success, success]
        launch_tui_in_tmux(
            session_name="stt-lyrics",
            initial_state=initial_state,
            auto_run=True,
        )

    assert run.call_args_list[0].args[0] == [
        "/usr/bin/tmux",
        "has-session",
        "-t",
        "=stt-lyrics",
    ]
    assert run.call_args_list[1].args[0] == ["/usr/bin/node", "--version"]
    command = run.call_args_list[2].args[0]
    assert command[:7] == [
        "/usr/bin/tmux",
        "new-session",
        "-d",
        "-s",
        "stt-lyrics",
        "-c",
        str(Path(__file__).resolve().parents[2]),
    ]
    assert command[7] == "-e"
    environment_name, bootstrap_json = command[8].split("=", 1)
    assert environment_name == "DOMINICAN_EATERS_TUI_BOOTSTRAP"
    assert json.loads(bootstrap_json) == {
        "initialState": initial_state,
        "autoRun": True,
    }
    assert command[9:] == [sys.executable, "-m", "dominican_eaters", "tui"]
    assert run.call_args_list[3].args[0] == [
        "/usr/bin/tmux",
        "attach-session",
        "-t",
        "=stt-lyrics",
    ]


def test_tmux_launcher_attaches_existing_session_without_reconfiguring(
    capsys: pytest.CaptureFixture[str],
) -> None:
    initial_state = {"workflow": "stt-suite"}
    with (
        patch.dict("dominican_eaters.tui.os.environ", {}, clear=True),
        patch("dominican_eaters.tui.shutil.which", return_value="/usr/bin/tmux"),
        patch("dominican_eaters.tui.subprocess.run") as run,
    ):
        success = type("Result", (), {"returncode": 0})()
        run.side_effect = [success, success]
        launch_tui_in_tmux(
            session_name="stt-lyrics",
            initial_state=initial_state,
            auto_run=True,
        )

    assert [call.args[0] for call in run.call_args_list] == [
        ["/usr/bin/tmux", "has-session", "-t", "=stt-lyrics"],
        ["/usr/bin/tmux", "attach-session", "-t", "=stt-lyrics"],
    ]
    assert "current configuration remains active" in capsys.readouterr().out


def test_tmux_launcher_switches_client_when_already_inside_tmux() -> None:
    with (
        patch.dict("dominican_eaters.tui.os.environ", {"TMUX": "/tmp/tmux"}, clear=True),
        patch("dominican_eaters.tui.shutil.which", return_value="/usr/bin/tmux"),
        patch("dominican_eaters.tui.subprocess.run") as run,
    ):
        success = type("Result", (), {"returncode": 0})()
        run.side_effect = [success, success]
        launch_tui_in_tmux(
            session_name="stt-lyrics",
            initial_state={"workflow": "stt-suite"},
            auto_run=True,
        )

    assert [call.args[0] for call in run.call_args_list] == [
        ["/usr/bin/tmux", "has-session", "-t", "=stt-lyrics"],
        ["/usr/bin/tmux", "switch-client", "-t", "=stt-lyrics"],
    ]


def test_tmux_launcher_rejects_old_node_before_creating_session() -> None:
    with (
        patch.dict("dominican_eaters.tui.os.environ", {}, clear=True),
        patch(
            "dominican_eaters.tui.shutil.which",
            side_effect=lambda name: f"/usr/bin/{name}",
        ),
        patch("dominican_eaters.tui.subprocess.run") as run,
        pytest.raises(TUIUnavailableError, match=r"requires Node\.js 26\.4.*v18\.19\.1"),
    ):
        missing = type("Result", (), {"returncode": 1})()
        old_node = type("Result", (), {"returncode": 0, "stdout": "v18.19.1\n"})()
        run.side_effect = [missing, old_node]
        launch_tui_in_tmux(
            session_name="stt-lyrics",
            initial_state={"workflow": "stt-suite"},
            auto_run=True,
        )

    assert [call.args[0] for call in run.call_args_list] == [
        ["/usr/bin/tmux", "has-session", "-t", "=stt-lyrics"],
        ["/usr/bin/node", "--version"],
    ]


@pytest.mark.parametrize("session_name", ["", "bad name", "bad;name", "a" * 65])
def test_tmux_launcher_rejects_unsafe_session_names(session_name: str) -> None:
    with pytest.raises(TUIUnavailableError, match="tmux session names"):
        launch_tui_in_tmux(session_name=session_name, initial_state={}, auto_run=False)


def test_tmux_launcher_reports_missing_tmux() -> None:
    with (
        patch.dict("dominican_eaters.tui.os.environ", {}, clear=True),
        patch("dominican_eaters.tui.shutil.which", return_value=None),
        pytest.raises(TUIUnavailableError, match="requires tmux"),
    ):
        launch_tui_in_tmux(
            session_name="stt-lyrics",
            initial_state={},
            auto_run=True,
        )


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
