"""Textual wizard for running Dominican Eaters workflows."""

from __future__ import annotations

import asyncio
import os
import shlex
import sys
import time
from datetime import datetime
from typing import Literal

from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Button,
    Checkbox,
    Footer,
    Input,
    Label,
    RichLog,
    Select,
    Static,
)

from dominican_eaters.speech.asr.registry import MODEL_PRESETS, RUNTIME_SPECS

from .commands import (
    COLLECTION_RUNS,
    DEFAULT_OUTPUT_DIRS,
    DEFAULT_SOURCE_PATHS,
    OUTPUT_WORKFLOWS,
    WORKFLOW_OPTIONS,
    CommandValidationError,
    Workflow,
    WorkflowRequest,
    build_cli_args,
    discover_worker_python,
)


class DominicanEatersApp(App[None]):
    """Discover, configure, and run every supported CLI workflow."""

    CSS_PATH = "styles.tcss"
    TITLE = "Dominican Eaters"
    SUB_TITLE = "Workflows"
    BINDINGS = [
        Binding("ctrl+r", "run_workflow", "Run", priority=True),
        Binding("ctrl+l", "clear_log", "Clear log", priority=True),
        Binding("ctrl+o", "toggle_console", "Focus console", priority=True),
        Binding("ctrl+p", "focus_workflow", "Workflow", priority=True),
        Binding("escape", "focus_log", "Log", priority=True),
        Binding("ctrl+c", "cancel_run", "Cancel", priority=True),
        Binding("ctrl+q", "quit", "Quit", priority=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._process: asyncio.subprocess.Process | None = None
        self._cancel_requested = False
        self._active_workflow = Workflow.CONFIG_VALIDATE
        self._console_maximized = False
        self._detected_worker_python = ""
        self._run_started_at: float | None = None
        self._run_active = False
        self._log_line_count = 0

    def compose(self) -> ComposeResult:
        with Horizontal(id="topbar"):
            yield Static("DOMINICAN EATERS", id="brand")
            yield Static("WORKFLOW CONSOLE", id="product")
            yield Static("● READY", id="run-state")
        with Horizontal(id="workspace"):
            with Vertical(id="control-panel"):
                yield Label("RUN CONFIGURATION", classes="panel-title")
                with VerticalScroll(id="form-scroll"):
                    yield Label("Workflow", classes="field-label")
                    yield Select(WORKFLOW_OPTIONS, value=Workflow.CONFIG_VALIDATE, id="workflow")
                    with Vertical(classes="field", id="source-field"):
                        yield Label("Configuration file", id="source-label")
                        yield Input(
                            value="config/default.yaml",
                            placeholder="Path to JSON or YAML",
                            id="source",
                        )
                    with Vertical(classes="field hidden", id="output-field"):
                        yield Label("Output directory")
                        yield Input(placeholder="artifacts/my-run", id="output")
                    with Vertical(classes="field hidden", id="data-root-field"):
                        yield Label("Dataset/data root override (optional)")
                        yield Input(placeholder="/srv/datasets", id="data-root")
                    with Vertical(classes="field hidden", id="artifacts-root-field"):
                        yield Label("Artifacts root override (optional)")
                        yield Input(placeholder="/srv/artifacts", id="artifacts-root")
                    with Vertical(classes="field hidden", id="preset-field"):
                        yield Label("Model preset")
                        yield Select(
                            tuple(
                                (f"{preset.label} [{preset.status}]", preset.preset_id)
                                for preset in MODEL_PRESETS.values()
                            ),
                            value="whisper-base",
                            allow_blank=False,
                            id="preset",
                        )
                        yield Static("", id="preset-state")
                    with Horizontal(classes="field-row hidden", id="runtime-fields"):
                        with Vertical(classes="field"):
                            yield Label("Device")
                            yield Select(
                                (("Auto", "auto"), ("CPU", "cpu"), ("CUDA", "cuda")),
                                value="cuda",
                                allow_blank=False,
                                id="device",
                            )
                        with Vertical(classes="field"):
                            yield Label("Precision")
                            yield Select(
                                (
                                    ("Auto", "auto"),
                                    ("FP16", "fp16"),
                                    ("FP32", "fp32"),
                                    ("BF16", "bf16"),
                                ),
                                value="fp16",
                                allow_blank=False,
                                id="precision",
                            )
                    with Vertical(classes="field hidden", id="worker-field"):
                        yield Label("Worker Python", id="worker-label")
                        yield Input(
                            value=self._detected_worker_python,
                            placeholder="/absolute/path/to/.venv-nemo/bin/python",
                            id="worker-python",
                        )
                    with Horizontal(classes="field-row hidden", id="benchmark-number-fields"):
                        with Vertical(classes="field"):
                            yield Label("Warmup runs")
                            yield Input(value="1", type="integer", id="warmup-runs")
                        with Vertical(classes="field"):
                            yield Label("Timeout (seconds)")
                            yield Input(value="300", type="number", id="request-timeout")
                    with Horizontal(classes="field-row hidden", id="audio-policy-fields"):
                        with Vertical(classes="field"):
                            yield Label("Short audio")
                            yield Select(
                                (("Reject", "reject"), ("Allow", "allow")),
                                value="reject",
                                allow_blank=False,
                                id="short-audio-policy",
                            )
                        with Vertical(classes="field"):
                            yield Label("Minimum seconds")
                            yield Input(value="0.1", type="number", id="minimum-audio-seconds")
                    with Vertical(classes="options"):
                        yield Checkbox("Force reprocessing", id="force", classes="hidden")
                        yield Checkbox("Verify hashes", id="verify", classes="hidden")
                        yield Checkbox("Request timestamps", id="timestamps", classes="hidden")
                with Vertical(id="command-dock"):
                    yield Label("COMMAND", classes="panel-title")
                    yield Static("", id="preview")
                    yield Static("Ready", id="status")
                    with Horizontal(id="actions"):
                        yield Button("Run  ^R", id="run", variant="primary")
                        yield Button("Stop  ^C", id="cancel", variant="warning", disabled=True)
            with Vertical(id="console-panel"):
                with Horizontal(id="console-header"):
                    yield Label("CONSOLE", classes="panel-title")
                    yield Static("Esc focus  ·  ^O maximize  ·  ^L clear", id="console-hints")
                yield RichLog(
                    id="output-log", highlight=True, markup=False, wrap=True, auto_scroll=True
                )
                with Horizontal(id="console-footer"):
                    yield Static("OUTPUT", id="output-label")
                    yield Static("waiting for a run", id="output-state")
                    yield Button("Clear", id="clear", variant="default")
        yield Footer()

    def on_mount(self) -> None:
        self._sync_fields()
        self.set_interval(1, self._refresh_run_metrics)
        self.query_one("#source", Input).focus()

    @on(Select.Changed, "#workflow")
    def workflow_changed(self) -> None:
        self._sync_fields()

    @on(Select.Changed, "#preset")
    def preset_changed(self) -> None:
        preset = MODEL_PRESETS[self._selected("#preset")]
        default_device = "auto"
        if preset.precision in {"fp16", "bf16"} and "cuda" in (
            preset.capabilities.supported_devices
        ):
            default_device = "cuda"
        elif len(preset.capabilities.supported_devices) == 1:
            default_device = preset.capabilities.supported_devices[0]
        self.query_one("#device", Select).value = default_device
        self.query_one("#precision", Select).value = preset.precision or "auto"
        self._sync_fields()

    @on(Select.Changed, "#device")
    @on(Select.Changed, "#precision")
    @on(Select.Changed, "#short-audio-policy")
    def benchmark_option_changed(self) -> None:
        self._update_preview()

    @on(Input.Changed)
    @on(Checkbox.Changed)
    def form_changed(self) -> None:
        self._update_preview()

    @on(Button.Pressed)
    def button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "run":
            self.run_selected_workflow()
        elif event.button.id == "cancel":
            self.action_cancel_run()
        elif event.button.id == "clear":
            self.action_clear_log()

    def action_run_workflow(self) -> None:
        if self._process is None:
            self.run_selected_workflow()

    def action_clear_log(self) -> None:
        self.query_one("#output-log", RichLog).clear()
        self._log_line_count = 0
        if not self._run_active:
            self._run_started_at = None
        self.query_one("#output-state", Static).update("log cleared")

    def action_toggle_console(self) -> None:
        self._console_maximized = not self._console_maximized
        self.set_class(self._console_maximized, "console-maximized")
        hint = (
            "^O restore  ·  ^L clear"
            if self._console_maximized
            else "Esc focus  ·  ^O maximize  ·  ^L clear"
        )
        self.query_one("#console-hints", Static).update(hint)
        self.action_focus_log()

    def action_focus_workflow(self) -> None:
        if self._console_maximized:
            self.action_toggle_console()
        self.query_one("#workflow", Select).focus()

    def action_focus_log(self) -> None:
        self.query_one("#output-log", RichLog).focus()

    @work(exclusive=True)
    async def run_selected_workflow(self) -> None:
        try:
            args = build_cli_args(self._request())
        except CommandValidationError as error:
            self._set_status(str(error), error=True)
            return
        command = (sys.executable, "-m", "dominican_eaters", *args)
        if not self._console_maximized:
            self.action_focus_log()
        self._run_started_at = time.monotonic()
        self._log_line_count = 0
        self._write_log("─" * 64, level="muted", count=False)
        self._write_log(f"Starting {self._selected_workflow().value}", level="status", count=False)
        self._write_log(f"$ {shlex.join(command)}", level="command", count=False)
        self._set_running(True)
        self._cancel_requested = False
        try:
            self._process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            stdout = self._process.stdout
            if stdout is None:
                raise RuntimeError("Could not capture workflow output")
            while line := await stdout.readline():
                self._write_log(line.decode("utf-8", errors="replace").rstrip())
            return_code = await self._process.wait()
            if self._cancel_requested:
                self._set_status("Cancelled", error=True)
                self._write_log("Run cancelled", level="warning", count=False)
            elif return_code == 0:
                self._set_status("Completed successfully")
                self._write_log(
                    f"Completed successfully in {self._elapsed_label()}",
                    level="success",
                    count=False,
                )
            else:
                self._set_status(f"Exited with status {return_code}", error=True)
                self._write_log(
                    f"Process exited with status {return_code} after {self._elapsed_label()}",
                    level="error",
                    count=False,
                )
        except asyncio.CancelledError:
            await self._stop_process()
            self._set_status("Cancelled", error=True)
            self._write_log("Run cancelled", level="warning", count=False)
        except Exception as error:
            await self._stop_process()
            self._write_log(f"{type(error).__name__}: {error}", level="error")
            self._set_status("Could not run workflow", error=True)
        finally:
            self._process = None
            self._cancel_requested = False
            self._set_running(False)

    def action_cancel_run(self) -> None:
        if self._process is None:
            self.exit()
            return
        self._cancel_requested = True
        self._process.terminate()
        self._set_status("Stopping…")

    async def action_quit(self) -> None:
        await self._stop_process()
        self.exit()

    async def _stop_process(self) -> None:
        process = self._process
        if process is None or process.returncode is not None:
            return
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=3)
        except TimeoutError:
            process.kill()
            await process.wait()

    def _sync_fields(self) -> None:
        workflow = self._selected_workflow()
        benchmark = workflow is Workflow.STT_BENCHMARK
        stt_workflow = workflow in {Workflow.STT_PREFLIGHT, Workflow.STT_BENCHMARK}
        source = self.query_one("#source", Input)
        output = self.query_one("#output", Input)
        previous_source = DEFAULT_SOURCE_PATHS[self._active_workflow]
        previous_output = DEFAULT_OUTPUT_DIRS.get(self._active_workflow, "")
        if not source.value.strip() or source.value == previous_source:
            source.value = DEFAULT_SOURCE_PATHS[workflow]
        if not output.value.strip() or output.value == previous_output:
            output.value = DEFAULT_OUTPUT_DIRS.get(workflow, "")
        self._active_workflow = workflow
        self._show("#output-field", workflow in OUTPUT_WORKFLOWS)
        self._show(
            "#data-root-field", workflow in {Workflow.CONFIG_VALIDATE, Workflow.STT_PREFLIGHT}
        )
        self._show("#artifacts-root-field", workflow is Workflow.CONFIG_VALIDATE)
        self._show("#preset-field", stt_workflow)
        self._show("#runtime-fields", stt_workflow)
        preset = MODEL_PRESETS[self._selected("#preset")]
        runtime = RUNTIME_SPECS[preset.runtime_id]
        requires_worker = runtime.execution == "worker"
        detected_worker = discover_worker_python(runtime_id=preset.runtime_id)
        worker_input = self.query_one("#worker-python", Input)
        if not worker_input.value.strip() or worker_input.value == self._detected_worker_python:
            worker_input.value = detected_worker
        self._detected_worker_python = detected_worker
        self._show("#worker-field", stt_workflow and requires_worker)
        self.query_one("#worker-label", Label).update(
            "Worker Python · detected"
            if detected_worker
            else f"Worker Python · {runtime.interpreter_env_var or 'current Python'}"
        )
        self.query_one("#preset-state", Static).update(
            f"{preset.status.upper()} · {preset.reason or 'Ready to run'}"
        )
        self._show("#benchmark-number-fields", benchmark)
        self._show("#audio-policy-fields", benchmark)
        self._show("#force", workflow in COLLECTION_RUNS)
        self._show("#verify", workflow in {Workflow.STT_PREFLIGHT, Workflow.STT_BENCHMARK})
        self._show("#timestamps", benchmark)
        self.query_one("#source-label", Label).update(
            "Configuration file" if workflow is Workflow.CONFIG_VALIDATE else "Manifest file"
        )
        self._update_preview()

    def _update_preview(self) -> None:
        try:
            args = build_cli_args(self._request())
            preview = shlex.join(("dominican-eaters", *args))
        except CommandValidationError as error:
            preview = f"Complete the form: {error}"
        self.query_one("#preview", Static).update(preview)

    def _request(self) -> WorkflowRequest:
        return WorkflowRequest(
            workflow=self._selected_workflow(),
            source_path=self.query_one("#source", Input).value,
            output_dir=self.query_one("#output", Input).value,
            data_root=self.query_one("#data-root", Input).value,
            artifacts_root=self.query_one("#artifacts-root", Input).value,
            preset=self._selected("#preset"),
            device=self._selected("#device"),
            precision=self._selected("#precision"),
            worker_python=self.query_one("#worker-python", Input).value,
            warmup_runs=self.query_one("#warmup-runs", Input).value,
            request_timeout=self.query_one("#request-timeout", Input).value,
            short_audio_policy=self._selected("#short-audio-policy"),
            minimum_audio_seconds=self.query_one("#minimum-audio-seconds", Input).value,
            force=self.query_one("#force", Checkbox).value,
            verify_hashes=self.query_one("#verify", Checkbox).value,
            timestamps=self.query_one("#timestamps", Checkbox).value,
        )

    def _selected_workflow(self) -> Workflow:
        return Workflow(self._selected("#workflow"))

    def _selected(self, selector: str) -> str:
        value = self.query_one(selector, Select).value
        if not isinstance(value, str):
            raise CommandValidationError(f"Select a value for {selector.removeprefix('#')}.")
        return value

    def _show(self, selector: str, visible: bool) -> None:
        self.query_one(selector).set_class(not visible, "hidden")

    def _set_status(self, message: str, *, error: bool = False) -> None:
        status = self.query_one("#status", Static)
        status.update(message)
        status.set_class(error, "error")

    def _write_log(
        self,
        message: str,
        *,
        level: Literal[
            "output", "command", "status", "success", "warning", "error", "muted"
        ] = "output",
        count: bool = True,
    ) -> None:
        if level == "output":
            lowered = message.lower().lstrip()
            if lowered.startswith(("error", "exception", "traceback")) or " failed:" in lowered:
                level = "error"
            elif lowered.startswith(("warning", "warn:", "skipping")):
                level = "warning"

        styles = {
            "output": ("", "#e6e9e7"),
            "command": ("CMD", "bold #77c38b"),
            "status": ("RUN", "bold #8bb8df"),
            "success": ("OK", "bold #77c38b"),
            "warning": ("WARN", "bold #d6ad65"),
            "error": ("ERR", "bold #e07373"),
            "muted": ("", "#596162"),
        }
        label, style = styles[level]
        rendered = Text()
        rendered.append(datetime.now().strftime("%H:%M:%S"), style="#687071")
        if label:
            rendered.append(f"  {label:<4}", style=style)
        else:
            rendered.append("      ")
        rendered.append(f"  {message}", style=style)
        self.query_one("#output-log", RichLog).write(rendered)
        if count:
            self._log_line_count += 1
        self._refresh_run_metrics()

    def _elapsed_label(self) -> str:
        elapsed = (
            0 if self._run_started_at is None else int(time.monotonic() - self._run_started_at)
        )
        minutes, seconds = divmod(elapsed, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    def _refresh_run_metrics(self) -> None:
        if self._run_started_at is None:
            return
        state = "running" if self._run_active else "finished"
        self.query_one("#output-state", Static).update(
            f"{self._log_line_count} lines  ·  {self._elapsed_label()}  ·  {state}"
        )

    def _set_running(self, running: bool) -> None:
        self._run_active = running
        self.query_one("#run", Button).disabled = running
        self.query_one("#cancel", Button).disabled = not running
        self.query_one("#workflow", Select).disabled = running
        run_state = self.query_one("#run-state", Static)
        run_state.update("● RUNNING" if running else "● READY")
        run_state.set_class(running, "running")
        self._refresh_run_metrics()
