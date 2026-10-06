from __future__ import annotations

import pytest

pytest.importorskip("textual")

from textual.widgets import Input, RichLog, Select, Static  # noqa: E402

from dominican_eaters.tui.app import DominicanEatersApp  # noqa: E402
from dominican_eaters.tui.commands import Workflow  # noqa: E402


@pytest.mark.asyncio
async def test_tui_mounts_with_a_valid_default_command_preview() -> None:
    app = DominicanEatersApp()

    async with app.run_test(size=(110, 40)) as pilot:
        await pilot.pause()

        assert str(app.query_one("#preview", Static).content) == (
            "dominican-eaters config validate config/default.yaml"
        )
        assert app.query_one("#output-field").has_class("hidden")


@pytest.mark.asyncio
async def test_tui_reveals_benchmark_fields_and_builds_command() -> None:
    app = DominicanEatersApp()

    async with app.run_test(size=(110, 40)) as pilot:
        app.query_one("#workflow", Select).value = Workflow.STT_BENCHMARK
        await pilot.pause()

        preview = str(app.query_one("#preview", Static).content)
        assert preview.startswith("dominican-eaters stt benchmark data/manifests/stt.json")
        assert "--output-dir artifacts/stt-run" in preview
        assert "--preset whisper-base" in preview
        assert not app.query_one("#preset-field").has_class("hidden")
        assert app.query_one("#worker-field").has_class("hidden")

        app.query_one("#preset", Select).value = "parakeet-tdt-0.6b-v3"
        await pilot.pause()
        assert not app.query_one("#worker-field").has_class("hidden")
        assert "Select the isolated worker Python executable" in str(
            app.query_one("#preview", Static).content
        )


@pytest.mark.asyncio
async def test_tui_stt_preflight_includes_environment_controls() -> None:
    app = DominicanEatersApp()

    async with app.run_test(size=(110, 40)) as pilot:
        app.query_one("#workflow", Select).value = Workflow.STT_PREFLIGHT
        await pilot.pause()

        preview = str(app.query_one("#preview", Static).content)
        assert "stt preflight data/manifests/stt.json" in preview
        assert "--preset whisper-base --device cuda --precision fp16" in preview
        assert not app.query_one("#preset-field").has_class("hidden")
        assert not app.query_one("#runtime-fields").has_class("hidden")

        app.query_one("#preset", Select).value = "granite-speech-4.1-2b"
        await pilot.pause()
        assert "PLANNED" in str(app.query_one("#preset-state", Static).content)
        assert "--preset granite-speech-4.1-2b" in str(app.query_one("#preview", Static).content)


@pytest.mark.asyncio
async def test_tui_selects_domain_specific_manifest_defaults() -> None:
    app = DominicanEatersApp()

    async with app.run_test(size=(110, 40)) as pilot:
        app.query_one("#workflow", Select).value = Workflow.BOOKS_RUN
        await pilot.pause()

        assert app.query_one("#source", Input).value == "data/manifests/books.json"
        assert app.query_one("#output", Input).value == "artifacts/books"


@pytest.mark.asyncio
async def test_tui_runs_a_workflow_and_reports_success() -> None:
    app = DominicanEatersApp()

    async with app.run_test(size=(110, 40)):
        worker = app.run_selected_workflow()
        await worker.wait()

        assert str(app.query_one("#status", Static).content) == "Completed successfully"
        output_state = str(app.query_one("#output-state", Static).content)
        assert "lines" in output_state
        assert "finished" in output_state
        assert app._log_line_count > 0


@pytest.mark.asyncio
async def test_tui_can_maximize_console_and_restore_workflow_focus() -> None:
    app = DominicanEatersApp()

    async with app.run_test(size=(110, 40)) as pilot:
        await pilot.press("ctrl+o")
        await pilot.pause()

        assert app.has_class("console-maximized")
        assert app.query_one("#output-log", RichLog).has_focus

        await pilot.press("ctrl+p")
        await pilot.pause()

        assert not app.has_class("console-maximized")
        assert app.query_one("#workflow", Select).has_focus


@pytest.mark.asyncio
async def test_tui_clear_shortcut_resets_console_state() -> None:
    app = DominicanEatersApp()

    async with app.run_test(size=(110, 40)) as pilot:
        app.query_one("#output-log", RichLog).write("temporary output")
        await pilot.press("ctrl+l")
        await pilot.pause()

        assert str(app.query_one("#output-state", Static).content) == "log cleared"
