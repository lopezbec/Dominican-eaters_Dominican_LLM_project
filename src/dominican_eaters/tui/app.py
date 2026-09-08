"""Textual wizard for running Dominican Eaters workflows."""

from __future__ import annotations

import asyncio
import shlex
import sys

from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Button,
    Checkbox,
    Footer,
    Header,
    Input,
    Label,
    RichLog,
    Select,
    Static,
)

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
)


class DominicanEatersApp(App[None]):
    """Discover, configure, and run every supported CLI workflow."""

    CSS_PATH = "styles.tcss"
    TITLE = "Dominican Eaters"
    SUB_TITLE = "Workflow launcher"
    BINDINGS = [
        ("ctrl+c", "cancel_run", "Cancel run"),
        ("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._process: asyncio.subprocess.Process | None = None
        self._cancel_requested = False
        self._active_workflow = Workflow.CONFIG_VALIDATE

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="body"):
            yield Label("Choose what you want to do", classes="section-title")
            yield Select(WORKFLOW_OPTIONS, value=Workflow.CONFIG_VALIDATE, id="workflow")
            with Vertical(classes="field", id="source-field"):
                yield Label("Configuration file", id="source-label")
                yield Input(
                    value="config/default.yaml", placeholder="Path to JSON or YAML", id="source"
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
            with Horizontal(classes="field-row hidden", id="backend-fields"):
                with Vertical(classes="field"):
                    yield Label("Backend")
                    yield Select(
                        (("Whisper", "whisper"), ("Parakeet", "parakeet"), ("Canary", "canary")),
                        value="whisper",
                        allow_blank=False,
                        id="backend",
                    )
                with Vertical(classes="field"):
                    yield Label("Model (optional)")
                    yield Input(placeholder="Use backend default", id="model")
            with Horizontal(classes="field-row hidden", id="runtime-fields"):
                with Vertical(classes="field"):
                    yield Label("Device")
                    yield Select(
                        (("Auto", "auto"), ("CPU", "cpu"), ("CUDA", "cuda")),
                        value="auto",
                        allow_blank=False,
                        id="device",
                    )
                with Vertical(classes="field"):
                    yield Label("Precision")
                    yield Select(
                        (("Auto", "auto"), ("FP16", "fp16"), ("FP32", "fp32"), ("BF16", "bf16")),
                        value="auto",
                        allow_blank=False,
                        id="precision",
                    )
            with Vertical(classes="field hidden", id="worker-field"):
                yield Label("Worker Python (required for Parakeet and Canary)")
                yield Input(
                    placeholder="/absolute/path/to/.venv-nemo/bin/python", id="worker-python"
                )
            with Horizontal(classes="field-row hidden", id="benchmark-number-fields"):
                with Vertical(classes="field"):
                    yield Label("Warmup runs")
                    yield Input(value="1", type="integer", id="warmup-runs")
                with Vertical(classes="field"):
                    yield Label("Request timeout (seconds)")
                    yield Input(value="300", type="number", id="request-timeout")
            with Horizontal(classes="field-row hidden", id="audio-policy-fields"):
                with Vertical(classes="field"):
                    yield Label("Short-audio policy")
                    yield Select(
                        (("Reject", "reject"), ("Allow", "allow")),
                        value="reject",
                        allow_blank=False,
                        id="short-audio-policy",
                    )
                with Vertical(classes="field"):
                    yield Label("Minimum audio duration (seconds)")
                    yield Input(value="0.1", type="number", id="minimum-audio-seconds")
            with Horizontal(classes="options"):
                yield Checkbox("Force reprocessing", id="force", classes="hidden")
                yield Checkbox("Verify hashes", id="verify", classes="hidden")
                yield Checkbox("Request timestamps", id="timestamps", classes="hidden")
            yield Label("Command preview", classes="section-title")
            yield Static("", id="preview")
            yield Static("Ready", id="status")
            with Horizontal(id="actions"):
                yield Button("Run", id="run", variant="primary")
                yield Button("Cancel", id="cancel", variant="warning", disabled=True)
                yield Button("Clear output", id="clear")
            yield RichLog(id="output-log", highlight=True, markup=False, wrap=True)
        yield Footer()

    def on_mount(self) -> None:
        self._sync_fields()
        self.query_one("#source", Input).focus()

    @on(Select.Changed, "#workflow")
    def workflow_changed(self) -> None:
        self._sync_fields()

    @on(Select.Changed, "#backend")
    def backend_changed(self) -> None:
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
            self.query_one("#output-log", RichLog).clear()

    @work(exclusive=True)
    async def run_selected_workflow(self) -> None:
        try:
            args = build_cli_args(self._request())
        except CommandValidationError as error:
            self._set_status(str(error), error=True)
            return
        command = (sys.executable, "-m", "dominican_eaters", *args)
        log = self.query_one("#output-log", RichLog)
        log.write(f"$ {shlex.join(command)}")
        self._set_running(True)
        self._cancel_requested = False
        try:
            self._process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            stdout = self._process.stdout
            if stdout is None:
                raise RuntimeError("Could not capture workflow output")
            while line := await stdout.readline():
                log.write(line.decode("utf-8", errors="replace").rstrip())
            return_code = await self._process.wait()
            if self._cancel_requested:
                self._set_status("Cancelled", error=True)
            elif return_code == 0:
                self._set_status("Completed successfully")
            else:
                self._set_status(f"Exited with status {return_code}", error=True)
        except asyncio.CancelledError:
            await self._stop_process()
            self._set_status("Cancelled", error=True)
        except Exception as error:
            await self._stop_process()
            log.write(f"{type(error).__name__}: {error}")
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
        self._show("#backend-fields", benchmark)
        self._show("#runtime-fields", benchmark)
        self._show("#worker-field", benchmark and self._selected("#backend") != "whisper")
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
            backend=self._selected("#backend"),
            model=self.query_one("#model", Input).value,
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

    def _set_running(self, running: bool) -> None:
        self.query_one("#run", Button).disabled = running
        self.query_one("#cancel", Button).disabled = not running
        self.query_one("#workflow", Select).disabled = running
