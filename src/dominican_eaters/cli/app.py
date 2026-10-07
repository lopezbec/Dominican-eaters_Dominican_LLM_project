"""Thin CLI edge for configuration and dataset validation."""

from __future__ import annotations

import os
import sys
from collections import Counter
from json import dumps
from pathlib import Path
from typing import cast

import click

from dominican_eaters import __version__
from dominican_eaters.collection.books import (
    BookCollectionRunner,
    ConcurrentCollectionError,
    YouTubeAudiobookSearch,
    load_book_manifest,
)
from dominican_eaters.collection.lyrics import (
    CollectionResult,
    GeniusAPI,
    LedgerConflictError,
    LyricsCollectionRunner,
    LyricsCollectionService,
    LyricsRequest,
    YouTubeMusicVideoSearch,
    load_lyrics_manifest,
)
from dominican_eaters.collection.media import (
    MediaDownloadRunner,
    MediaRecord,
    YtDlpMediaDownloader,
)
from dominican_eaters.collection.poems import (
    PoemCollector,
    YouTubeRecitationSearch,
    load_poem_manifest,
)
from dominican_eaters.collection.providers import ScrapeTubeSearch
from dominican_eaters.config import ConfigError, load_config
from dominican_eaters.data import (
    ConcurrentWriteError,
    ManifestValidationError,
    discover_stt_manifest,
    load_manifest,
    write_manifest,
)
from dominican_eaters.evaluation.asr import BenchmarkRunner, OutputCollisionError
from dominican_eaters.speech.asr import WorkerProcessError
from dominican_eaters.speech.asr.environment import preflight_asr_environment
from dominican_eaters.speech.asr.registry import (
    CURRENT_BACKEND_SPECS,
    MODEL_PRESETS,
    RUNTIME_SPECS,
    BackendName,
    ModelPreset,
)

from .backends import create_asr_backend


@click.group(invoke_without_command=True)
@click.version_option(version=__version__, prog_name="dominican-eaters")
@click.pass_context
def main(context: click.Context) -> None:
    """Collect and evaluate Dominican Spanish language data.

    Run without a subcommand in an interactive terminal to open the workflow launcher.
    """

    if context.invoked_subcommand is not None:
        return
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        click.echo(context.get_help())
        return
    _run_tui()


@main.command("tui")
def tui_command() -> None:
    """Open the interactive terminal workflow launcher."""

    _run_tui()


def _run_tui() -> None:
    from dominican_eaters.tui import TUIUnavailableError, run_tui

    try:
        run_tui()
    except TUIUnavailableError as error:
        raise click.ClickException(str(error)) from error


@main.group("config")
def config_group() -> None:
    """Validate canonical application configuration."""


@config_group.command("validate")
@click.argument("config_path", type=click.Path(path_type=Path, dir_okay=False))
@click.option("--data-root", type=click.Path(path_type=Path, file_okay=False))
@click.option("--artifacts-root", type=click.Path(path_type=Path, file_okay=False))
def validate_config(
    config_path: Path,
    data_root: Path | None,
    artifacts_root: Path | None,
) -> None:
    """Load CONFIG_PATH and print the effective filesystem roots."""

    try:
        config = load_config(
            config_path,
            data_root=data_root,
            artifacts_root=artifacts_root,
        )
    except ConfigError as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"schema_version={config.schema_version}")
    click.echo(f"data_root={config.data_root}")
    click.echo(f"artifacts_root={config.artifacts_root}")


@main.group("collect")
def collect_group() -> None:
    """Validate and run content collection workflows."""


@collect_group.group("books")
def collect_books_group() -> None:
    """Collect Dominican audiobook records."""


@collect_books_group.command("preflight")
@click.argument("manifest_path", type=click.Path(path_type=Path, dir_okay=False))
def preflight_book_manifest(manifest_path: Path) -> None:
    """Validate one canonical book source manifest."""

    try:
        manifest = load_book_manifest(manifest_path)
    except ValueError as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"schema_version={manifest.schema_version}")
    click.echo(f"books={len(manifest.books)}")
    click.echo("preflight=passed")


@collect_books_group.command("run")
@click.argument("manifest_path", type=click.Path(path_type=Path, dir_okay=False))
@click.option("--output-dir", type=click.Path(path_type=Path, file_okay=False), required=True)
@click.option("--force/--no-force", default=False, show_default=True)
def run_book_collection(manifest_path: Path, output_dir: Path, force: bool) -> None:
    """Collect books with API-key-free YouTube search and a resumable checkpoint."""

    try:
        manifest = load_book_manifest(manifest_path)
        result = BookCollectionRunner(YouTubeAudiobookSearch(ScrapeTubeSearch()), force=force).run(
            manifest.books, output_dir / "books-collection.json"
        )
    except (ValueError, OSError, ConcurrentCollectionError) as error:
        raise click.ClickException(str(error)) from error
    counts = Counter(record.status.value for record in result.records)
    click.echo(f"state={result.state.value}")
    click.echo(f"records={len(result.records)}")
    for status in sorted(counts):
        click.echo(f"{status}={counts[status]}")
    error_records = [record for record in result.records if record.issue is not None]
    for record in error_records[:5]:
        if record.issue is not None:
            click.echo(
                f"error[{record.seed.book_id}]="
                f"{record.issue.stage}:{record.issue.error_type}:{record.issue.message}",
                err=True,
            )
    _echo_omitted_errors(len(error_records))
    click.echo(f"checkpoint={output_dir.resolve() / 'books-collection.json'}")
    if counts.get("error", 0):
        raise click.exceptions.Exit(1)


@collect_group.group("lyrics")
def collect_lyrics_group() -> None:
    """Collect Dominican song and lyrics records."""


@collect_lyrics_group.command("preflight")
@click.argument("manifest_path", type=click.Path(path_type=Path, dir_okay=False))
def preflight_lyrics_manifest(manifest_path: Path) -> None:
    """Validate one canonical lyrics request manifest."""

    try:
        manifest = load_lyrics_manifest(manifest_path)
    except ValueError as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"schema_version={manifest.schema_version}")
    click.echo(f"requests={len(manifest)}")
    click.echo("preflight=passed")


@collect_lyrics_group.command("run")
@click.argument("manifest_path", type=click.Path(path_type=Path, dir_okay=False))
@click.option("--output-dir", type=click.Path(path_type=Path, file_okay=False), required=True)
@click.option("--force/--no-force", default=False, show_default=True)
def run_lyrics_collection(manifest_path: Path, output_dir: Path, force: bool) -> None:
    """Collect songs using GENIUS_ACCESS_TOKEN and API-key-free YouTube search."""

    genius: GeniusAPI | None = None
    try:
        manifest = load_lyrics_manifest(manifest_path)
        genius = GeniusAPI(_required_environment("GENIUS_ACCESS_TOKEN"))
        service = LyricsCollectionService(genius, YouTubeMusicVideoSearch(ScrapeTubeSearch()))
        result = LyricsCollectionRunner(
            service,
            on_request_started=_echo_lyrics_request_started,
            on_result_saved=_echo_lyrics_result_saved,
        ).run(manifest, output_dir, force=force)
    except (ValueError, OSError, LedgerConflictError, ConcurrentWriteError) as error:
        raise click.ClickException(str(error)) from error
    finally:
        if genius is not None:
            genius.close()
    counts = Counter(item.status.value for item in result.results)
    click.echo(f"requests={len(result.manifest)}")
    click.echo(f"results={len(result.results)}")
    for status in sorted(counts):
        click.echo(f"{status}={counts[status]}")
    error_results = [item for item in result.results if item.status.value == "error"]
    for item in error_results[:5]:
        for issue in item.issues:
            click.echo(
                f"error[{item.request.request_id}]="
                f"{issue.stage.value}:{issue.code}:{issue.message}",
                err=True,
            )
    _echo_omitted_errors(len(error_results))
    click.echo(f"ledger={(output_dir / 'lyrics-collection.json').resolve()}")
    if counts.get("error", 0):
        raise click.exceptions.Exit(1)


def _echo_lyrics_request_started(
    index: int, total: int, request: LyricsRequest, attempt: int
) -> None:
    click.echo(
        f"progress={index}/{total} state=started request_id={request.request_id} "
        f"attempt={attempt} query={dumps(request.query, ensure_ascii=False)}"
    )


def _echo_lyrics_result_saved(index: int, total: int, result: CollectionResult) -> None:
    title = result.song.title if result.song is not None else result.request.query
    click.echo(
        f"progress={index}/{total} state=saved status={result.status.value} "
        f"request_id={result.request.request_id} attempt={result.attempt} "
        f"title={dumps(title, ensure_ascii=False)}"
    )


@collect_lyrics_group.command("download-audio")
@click.argument("ledger_path", type=click.Path(path_type=Path, dir_okay=False))
@click.option("--output-dir", type=click.Path(path_type=Path, file_okay=False), required=True)
@click.option("--timeout", type=click.FloatRange(min=1), default=900.0, show_default=True)
@click.option("--force/--no-force", default=False, show_default=True)
def download_lyrics_audio(ledger_path: Path, output_dir: Path, timeout: float, force: bool) -> None:
    """Download and normalize audio selected by a lyrics collection ledger."""

    runner = MediaDownloadRunner(
        YtDlpMediaDownloader(),
        timeout_seconds=timeout,
        on_progress=_echo_media_progress,
        on_log=lambda line: click.echo(f"download={line}"),
    )
    checks = runner.preflight()
    for name, available, detail in checks:
        status = "ok" if available else "missing"
        click.echo(f"environment_check[executable:{name}]={status}:{detail}")
    if not all(available for _, available, _ in checks):
        raise click.ClickException("media download environment preflight failed")
    try:
        ledger = runner.run(ledger_path, output_dir, force=force)
    except (ValueError, OSError, ConcurrentWriteError) as error:
        raise click.ClickException(str(error)) from error
    counts = Counter(record.status.value for record in ledger.records)
    click.echo(f"records={len(ledger.records)}")
    for status in sorted(counts):
        click.echo(f"{status}={counts[status]}")
    click.echo(f"ledger={(output_dir / 'media-download.json').resolve()}")
    if counts.get("failed", 0):
        raise click.exceptions.Exit(1)


def _echo_media_progress(index: int, total: int, record: MediaRecord) -> None:
    detail = f" error={record.error_code}" if record.error_code else ""
    click.echo(
        f"progress={index}/{total} state=saved status={record.status.value} "
        f"request_id={record.request_id} attempt={record.attempt}{detail}"
    )


@collect_group.group("poems")
def collect_poems_group() -> None:
    """Collect Dominican poem recitation records."""


@collect_poems_group.command("preflight")
@click.argument("manifest_path", type=click.Path(path_type=Path, dir_okay=False))
def preflight_poem_manifest(manifest_path: Path) -> None:
    """Validate one canonical poem source manifest."""

    try:
        manifest = load_poem_manifest(manifest_path)
    except ValueError as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"schema_version={manifest.schema_version}")
    click.echo(f"poems={len(manifest)}")
    click.echo("preflight=passed")


@collect_poems_group.command("run")
@click.argument("manifest_path", type=click.Path(path_type=Path, dir_okay=False))
@click.option("--output-dir", type=click.Path(path_type=Path, file_okay=False), required=True)
@click.option("--force/--no-force", default=False, show_default=True)
def run_poem_collection(manifest_path: Path, output_dir: Path, force: bool) -> None:
    """Collect poem recitations with API-key-free YouTube search."""

    try:
        manifest = load_poem_manifest(manifest_path)
        result = PoemCollector(
            provider=YouTubeRecitationSearch(ScrapeTubeSearch()),
            checkpoint_path=output_dir / "poems-collection.json",
        ).collect(manifest.poems, force=force)
    except (ValueError, OSError, ConcurrentWriteError) as error:
        raise click.ClickException(str(error)) from error
    counts = Counter(outcome.status.value for outcome in result.outcomes)
    click.echo(f"state={result.state.value}")
    click.echo(f"sources={len(result.sources)}")
    for status in sorted(counts):
        click.echo(f"{status}={counts[status]}")
    error_outcomes = [outcome for outcome in result.outcomes if outcome.error_type is not None]
    for outcome in error_outcomes[:5]:
        if outcome.error_type is not None:
            click.echo(
                f"error[{outcome.source.source_id}]="
                f"search:{outcome.error_type}:{outcome.error_message}",
                err=True,
            )
    _echo_omitted_errors(len(error_outcomes))
    click.echo(f"checkpoint={(output_dir / 'poems-collection.json').resolve()}")
    if counts.get("error", 0):
        raise click.exceptions.Exit(1)


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise ValueError(f"{name} is required in the environment")
    return value


def _echo_omitted_errors(total: int) -> None:
    if total > 5:
        click.echo(f"additional_errors={total - 5}", err=True)


@main.group("stt")
def stt_group() -> None:
    """Validate and evaluate speech-to-text datasets."""


@stt_group.group("manifest")
def stt_manifest_group() -> None:
    """Create canonical manifests from existing audio files."""


@stt_manifest_group.command("build")
@click.argument("audio_dir", type=click.Path(path_type=Path, file_okay=False))
@click.option(
    "--output-file",
    type=click.Path(path_type=Path, dir_okay=False),
    required=True,
)
@click.option("--require-references/--allow-missing-references", default=False, show_default=True)
def build_stt_manifest(audio_dir: Path, output_file: Path, require_references: bool) -> None:
    """Scan AUDIO_DIR recursively and write one sample per supported audio file."""

    try:
        manifest = discover_stt_manifest(
            audio_dir,
            require_references=require_references,
        )
        write_manifest(manifest, output_file)
    except (ManifestValidationError, OSError) as error:
        raise click.ClickException(str(error)) from error
    references = sum(bool(sample.reference_text) for sample in manifest)
    click.echo(f"dataset_root={manifest.dataset_root}")
    click.echo(f"samples={len(manifest)}")
    click.echo(f"references={references}")
    click.echo(f"missing_references={len(manifest) - references}")
    click.echo(f"manifest={output_file.expanduser().resolve()}")


@stt_group.group("models")
def stt_models_group() -> None:
    """Inspect registered speech-to-text model presets."""


@stt_models_group.command("list")
def list_stt_models() -> None:
    """List current, planned, experimental, and blocked presets."""

    for preset in MODEL_PRESETS.values():
        click.echo(f"{preset.preset_id}\t{preset.status}\t{preset.label}\t{preset.model}")


@stt_models_group.command("show")
@click.argument("preset_id", type=click.Choice(tuple(MODEL_PRESETS)))
def show_stt_model(preset_id: str) -> None:
    """Show runtime requirements and capabilities for one preset."""

    preset = MODEL_PRESETS[preset_id]
    runtime = RUNTIME_SPECS[preset.runtime_id]
    capabilities = preset.capabilities
    click.echo(f"preset={preset.preset_id}")
    click.echo(f"status={preset.status}")
    click.echo(f"label={preset.label}")
    click.echo(f"backend={preset.backend}")
    click.echo(f"model={preset.model}")
    click.echo(f"model_revision={preset.model_revision or 'unresolved'}")
    click.echo(f"runtime={runtime.runtime_id}")
    click.echo(f"execution={runtime.execution}")
    click.echo(f"worker_module={runtime.worker_module or 'inline'}")
    click.echo(f"interpreter_env_var={runtime.interpreter_env_var or 'current-python'}")
    click.echo(f"supported_python={runtime.supported_python}")
    click.echo(f"default_precision={preset.precision or 'unresolved'}")
    click.echo(f"quantization={preset.quantization or 'none'}")
    click.echo(f"timestamps={'yes' if capabilities.timestamps else 'no'}")
    click.echo(f"streaming={'yes' if capabilities.streaming else 'no'}")
    click.echo(f"language_detection={'yes' if capabilities.language_detection else 'no'}")
    click.echo(f"long_form_policy={capabilities.long_form_policy}")
    click.echo(f"devices={','.join(capabilities.supported_devices)}")
    click.echo(f"precisions={','.join(capabilities.supported_precisions)}")
    click.echo(f"reason={preset.reason or 'ready'}")


@stt_group.command("preflight")
@click.argument("manifest_path", type=click.Path(path_type=Path, dir_okay=False))
@click.option(
    "--dataset-root",
    type=click.Path(path_type=Path, file_okay=False),
    help="Explicitly remap the dataset root; declared hashes are verified by default.",
)
@click.option("--verify-hashes/--no-verify-hashes", default=False, show_default=True)
@click.option(
    "--preset",
    "preset_id",
    type=click.Choice(tuple(MODEL_PRESETS)),
    help="Registered model/runtime preset to inspect.",
)
@click.option(
    "--backend",
    type=click.Choice(tuple(CURRENT_BACKEND_SPECS)),
    help="Legacy backend selection; use --preset for reproducible runs.",
)
@click.option("--model", help="Model name; defaults are backend-specific.")
@click.option(
    "--device", type=click.Choice(["auto", "cpu", "cuda"]), default="auto", show_default=True
)
@click.option(
    "--precision",
    type=click.Choice(["auto", "fp16", "fp32", "bf16"]),
    default="auto",
    show_default=True,
)
@click.option(
    "--worker-python",
    type=click.Path(path_type=Path, dir_okay=False),
    help="Python executable for the selected isolated worker.",
)
def preflight_stt_manifest(
    manifest_path: Path,
    dataset_root: Path | None,
    verify_hashes: bool,
    preset_id: str | None,
    backend: str | None,
    model: str | None,
    device: str,
    precision: str,
    worker_python: Path | None,
) -> None:
    """Validate an STT dataset and, when selected, its model environment."""

    try:
        preset, backend_name, selected_model, selected_device, selected_precision = (
            _resolve_stt_selection(
                preset_id=preset_id,
                backend=backend,
                model=model,
                language=None,
                precision=precision,
                device=device,
            )
        )
        manifest = load_manifest(
            manifest_path,
            dataset_root_override=dataset_root,
            verify_hashes_on_override=True,
        )
        manifest.preflight(verify_hashes=verify_hashes or dataset_root is not None)
        environment_report = None
        worker_report = None
        if backend_name is not None:
            environment_report = preflight_asr_environment(
                backend=backend_name,
                worker_python=worker_python,
                requested_device=selected_device,
            )
            if environment_report.ready and (preset is None or preset.runnable):
                selected_backend = create_asr_backend(
                    backend=backend_name,
                    model=selected_model,
                    language=preset.language if preset is not None else "es",
                    device=selected_device,
                    precision=selected_precision,
                    worker_python=worker_python,
                    request_timeout_seconds=15,
                    timestamps=False,
                    worker_stderr_sink=_echo_worker_stderr,
                    preset=preset,
                )
                worker_preflight = getattr(selected_backend, "preflight", None)
                if callable(worker_preflight):
                    worker_report = worker_preflight(close_after=True)
    except ManifestValidationError as error:
        raise click.ClickException(str(error)) from error
    except ValueError as error:
        raise click.ClickException(str(error)) from error
    except WorkerProcessError as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"schema_version={manifest.schema_version}")
    click.echo(f"dataset_root={manifest.dataset_root}")
    click.echo(f"samples={len(manifest)}")
    if preset is not None:
        click.echo(f"preset={preset.preset_id}")
        click.echo(f"preset_status={preset.status}")
        click.echo(f"preset_runtime={preset.runtime_id}")
        click.echo(f"preset_runnable={'yes' if preset.runnable else 'no'}")
        click.echo(f"preset_reason={preset.reason or 'ready'}")
    if environment_report is not None:
        click.echo(f"environment_backend={environment_report.backend}")
        click.echo(f"environment_model={selected_model}")
        click.echo(f"environment_python={environment_report.interpreter}")
        click.echo(f"environment_python_version={environment_report.python_version}")
        for check in environment_report.checks:
            state = "ok" if check.available else "missing"
            click.echo(f"environment_check[{check.kind}:{check.name}]={state}:{check.detail}")
        click.echo("environment_preflight=" + ("passed" if environment_report.ready else "failed"))
    if worker_report is not None:
        for worker_check in worker_report.checks:
            click.echo(
                f"worker_check[{worker_check.name}]={worker_check.status}:"
                f"{worker_check.severity}:{worker_check.message}"
            )
        for name, value in sorted(worker_report.environment.items()):
            click.echo(f"worker_environment[{name}]={value}")
        click.echo("worker_preflight=" + ("passed" if worker_report.ready else "failed"))
    failures = []
    if preset is not None and not preset.runnable:
        failures.append(_preset_unavailable_message(preset, action="complete preflight"))
    if environment_report is not None and not environment_report.ready:
        failures.append(f"{environment_report.backend} environment preflight failed")
    if worker_report is not None and not worker_report.ready:
        failures.append(f"{backend_name} worker preflight failed")
    if failures:
        raise click.ClickException("; ".join(failures))
    click.echo("preflight=passed")


@stt_group.command("benchmark")
@click.argument("manifest_path", type=click.Path(path_type=Path, dir_okay=False))
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False),
    required=True,
)
@click.option(
    "--preset",
    "preset_id",
    type=click.Choice(tuple(MODEL_PRESETS)),
    help="Registered model/runtime preset. Non-current presets cannot be benchmarked.",
)
@click.option(
    "--backend",
    type=click.Choice(tuple(CURRENT_BACKEND_SPECS)),
    help="Legacy backend selection; defaults to whisper when --preset is omitted.",
)
@click.option("--model", help="Model name; defaults are backend-specific.")
@click.option("--language", help="Language override; defaults to the preset language or es.")
@click.option(
    "--device", type=click.Choice(["auto", "cpu", "cuda"]), default="auto", show_default=True
)
@click.option(
    "--precision",
    type=click.Choice(["auto", "fp16", "fp32", "bf16"]),
    default="auto",
    show_default=True,
)
@click.option("--warmup-runs", type=click.IntRange(min=0), default=1, show_default=True)
@click.option(
    "--worker-python",
    type=click.Path(path_type=Path, dir_okay=False),
    help="Absolute Python executable for the selected isolated worker.",
)
@click.option(
    "--request-timeout",
    type=click.FloatRange(min=0, min_open=True),
    default=300.0,
    show_default=True,
)
@click.option("--timestamps/--no-timestamps", default=False, show_default=True)
@click.option(
    "--short-audio-policy",
    type=click.Choice(["reject", "allow"]),
    default="reject",
    show_default=True,
)
@click.option(
    "--minimum-audio-seconds",
    type=click.FloatRange(min=0, min_open=True),
    default=0.1,
    show_default=True,
)
@click.option("--verify-hashes/--no-verify-hashes", default=False, show_default=True)
@click.pass_context
def benchmark_stt(
    context: click.Context,
    manifest_path: Path,
    output_dir: Path,
    preset_id: str | None,
    backend: str | None,
    model: str | None,
    language: str | None,
    device: str,
    precision: str,
    warmup_runs: int,
    worker_python: Path | None,
    request_timeout: float,
    timestamps: bool,
    short_audio_policy: str,
    minimum_audio_seconds: float,
    verify_hashes: bool,
) -> None:
    """Run one canonical benchmark into a new OUTPUT_DIR."""

    try:
        preset, backend_name, selected_model, selected_device, selected_precision = (
            _resolve_stt_selection(
                preset_id=preset_id,
                backend=backend,
                model=model,
                language=language,
                precision=precision,
                device=device,
                require_backend=True,
            )
        )
        if preset is not None and not preset.runnable:
            raise ValueError(_preset_unavailable_message(preset, action="benchmark"))
        assert backend_name is not None
        manifest = load_manifest(manifest_path)
        selected_backend = create_asr_backend(
            backend=backend_name,
            model=selected_model,
            language=language or (preset.language if preset is not None else "es"),
            device=selected_device,
            precision=selected_precision,
            worker_python=worker_python,
            request_timeout_seconds=request_timeout,
            timestamps=timestamps,
            short_audio_policy=short_audio_policy,
            minimum_audio_seconds=minimum_audio_seconds,
            worker_stderr_sink=_echo_worker_stderr,
            preset=preset,
        )
        result = BenchmarkRunner(
            backend=selected_backend,
            warmup_runs=warmup_runs,
            verify_hashes=verify_hashes,
        ).run(manifest, output_dir)
    except (ManifestValidationError, ValueError, OutputCollisionError) as error:
        raise click.ClickException(str(error)) from error

    click.echo(f"status={result.status.value}")
    click.echo(f"backend_id={result.backend_id}")
    click.echo(f"output_dir={result.output_dir}")
    click.echo(f"expected_samples={result.expected_samples}")
    click.echo(f"succeeded_samples={result.succeeded_samples}")
    click.echo(f"failed_samples={result.failed_samples}")
    model_load = result.performance.model_load
    click.echo(
        "model_load_seconds="
        + (f"{model_load.elapsed_seconds:.6f}" if model_load is not None else "unavailable")
    )
    click.echo(
        "model_load_process_rss_peak_bytes="
        + (
            str(model_load.process_rss_peak_bytes)
            if model_load is not None and model_load.process_rss_peak_bytes is not None
            else "unavailable"
        )
    )
    click.echo(
        "model_load_process_tree_rss_peak_bytes="
        + (
            str(model_load.process_tree_rss_peak_bytes)
            if model_load is not None and model_load.process_tree_rss_peak_bytes is not None
            else "unavailable"
        )
    )
    click.echo(f"inference_seconds={result.performance.total_inference_seconds:.6f}")
    click.echo(f"audio_seconds={result.performance.total_audio_seconds:.6f}")
    click.echo(
        "real_time_factor="
        + (
            f"{result.performance.real_time_factor:.6f}"
            if result.performance.real_time_factor is not None
            else "unavailable"
        )
    )
    click.echo(f"process_rss_peak_bytes={result.performance.process_rss_peak_bytes}")
    click.echo(f"process_tree_rss_peak_bytes={result.performance.process_tree_rss_peak_bytes}")
    click.echo(f"gpu_peak_allocated_bytes={result.performance.gpu_peak_allocated_bytes}")
    click.echo(f"gpu_peak_reserved_bytes={result.performance.gpu_peak_reserved_bytes}")
    click.echo(f"result_artifact={result.output_dir / 'result.json'}")
    for failure in result.failures:
        sample = f":sample={failure.sample_id}" if failure.sample_id is not None else ""
        click.echo(f"failure={failure.stage.value}:{failure.error_type}:{failure.message}{sample}")
    if not result.successful:
        context.exit(1)


def _resolve_stt_selection(
    *,
    preset_id: str | None,
    backend: str | None,
    model: str | None,
    language: str | None,
    precision: str,
    device: str,
    require_backend: bool = False,
) -> tuple[ModelPreset | None, BackendName | None, str | None, str, str]:
    """Resolve a preset without obscuring explicit legacy option conflicts."""

    if preset_id is None:
        if backend is None and not require_backend:
            return None, None, model, device, precision
        backend_name = cast(BackendName, backend or "whisper")
        selected_model = model or CURRENT_BACKEND_SPECS[backend_name].default_model
        return None, backend_name, selected_model, device, precision

    preset = MODEL_PRESETS[preset_id]
    if backend is not None and backend != preset.backend:
        raise ValueError(f"Preset {preset_id!r} uses backend {preset.backend!r}, not {backend!r}")
    if model is not None and model != preset.model:
        raise ValueError(f"Preset {preset_id!r} fixes model {preset.model!r}, not {model!r}")
    if language is not None and language != preset.language:
        raise ValueError(
            f"Preset {preset_id!r} fixes language {preset.language!r}, not {language!r}"
        )
    if device != "auto" and device not in preset.capabilities.supported_devices:
        raise ValueError(f"Preset {preset_id!r} does not support device {device!r}")
    if precision != "auto" and precision not in preset.capabilities.supported_precisions:
        raise ValueError(f"Preset {preset_id!r} does not support precision {precision!r}")
    selected_precision = preset.precision if precision == "auto" and preset.precision else precision
    selected_device = device
    if device == "auto":
        if selected_precision in {"fp16", "bf16"} and "cuda" in (
            preset.capabilities.supported_devices
        ):
            selected_device = "cuda"
        elif len(preset.capabilities.supported_devices) == 1:
            selected_device = preset.capabilities.supported_devices[0]
    return preset, preset.backend, preset.model, selected_device, selected_precision


def _preset_unavailable_message(preset: ModelPreset, *, action: str) -> str:
    reason = preset.reason or "the adapter has not been promoted"
    return f"Preset {preset.preset_id!r} is {preset.status} and cannot {action}: {reason}"


def _echo_worker_stderr(message: str) -> None:
    """Forward worker diagnostics immediately while preserving their line endings."""

    click.echo(message, nl=False, err=True)
