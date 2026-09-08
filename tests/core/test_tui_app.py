from __future__ import annotations

import pytest

pytest.importorskip("textual")

from textual.widgets import Input, Select, Static  # noqa: E402

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
        app.query_one("#source", Input).value = "manifest.json"
        app.query_one("#output", Input).value = "artifacts/run"
        await pilot.pause()

        preview = str(app.query_one("#preview", Static).content)
        assert preview.startswith("dominican-eaters stt benchmark manifest.json")
        assert "--output-dir artifacts/run" in preview
        assert "--backend whisper" in preview
        assert not app.query_one("#backend-fields").has_class("hidden")
        assert app.query_one("#worker-field").has_class("hidden")

        app.query_one("#backend", Select).value = "parakeet"
        await pilot.pause()
        assert not app.query_one("#worker-field").has_class("hidden")
        assert "Select the isolated worker Python executable" in str(
            app.query_one("#preview", Static).content
        )


@pytest.mark.asyncio
async def test_tui_runs_a_workflow_and_reports_success() -> None:
    app = DominicanEatersApp()

    async with app.run_test(size=(110, 40)):
        worker = app.run_selected_workflow()
        await worker.wait()

        assert str(app.query_one("#status", Static).content) == "Completed successfully"
