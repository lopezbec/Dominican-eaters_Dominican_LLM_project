"""Thin CLI edge for configuration and dataset validation."""

from __future__ import annotations

import os
from collections import Counter
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
    GeniusAPI,
    LedgerConflictError,
    LyricsCollectionRunner,
    LyricsCollectionService,
    YouTubeMusicVideoSearch,
    load_lyrics_manifest,
)
from dominican_eaters.collection.poems import (
    PoemCollector,
    YouTubeRecitationSearch,
    load_poem_manifest,
)
from dominican_eaters.collection.providers import YouTubeDataAPI
from dominican_eaters.config import ConfigError, load_config
from dominican_eaters.data import ConcurrentWriteError, ManifestValidationError, load_manifest
from dominican_eaters.evaluation.asr import BenchmarkRunner, OutputCollisionError

from .backends import BackendName, create_asr_backend


@click.group()
@click.version_option(version=__version__, prog_name="dominican-eaters")
def main() -> None:
    """Collect and evaluate Dominican Spanish language data."""


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
    """Collect books using YOUTUBE_API_KEY and a resumable JSON checkpoint."""

    api: YouTubeDataAPI | None = None
    try:
        manifest = load_book_manifest(manifest_path)
        api = YouTubeDataAPI(_required_environment("YOUTUBE_API_KEY"))
        result = BookCollectionRunner(YouTubeAudiobookSearch(api), force=force).run(
            manifest.books, output_dir / "books-collection.json"
        )
    except (ValueError, OSError, ConcurrentCollectionError) as error:
        raise click.ClickException(str(error)) from error
    finally:
        if api is not None:
            api.close()
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
    """Collect songs using GENIUS_ACCESS_TOKEN and YOUTUBE_API_KEY."""

    genius: GeniusAPI | None = None
    youtube: YouTubeDataAPI | None = None
    try:
        manifest = load_lyrics_manifest(manifest_path)
        genius = GeniusAPI(_required_environment("GENIUS_ACCESS_TOKEN"))
        youtube = YouTubeDataAPI(_required_environment("YOUTUBE_API_KEY"))
        service = LyricsCollectionService(genius, YouTubeMusicVideoSearch(youtube))
        result = LyricsCollectionRunner(service).run(manifest, output_dir, force=force)
    except (ValueError, OSError, LedgerConflictError, ConcurrentWriteError) as error:
        raise click.ClickException(str(error)) from error
    finally:
        if genius is not None:
            genius.close()
        if youtube is not None:
            youtube.close()
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
    """Collect poem recitations using YOUTUBE_API_KEY."""

    api: YouTubeDataAPI | None = None
    try:
        manifest = load_poem_manifest(manifest_path)
        api = YouTubeDataAPI(_required_environment("YOUTUBE_API_KEY"))
        result = PoemCollector(
            provider=YouTubeRecitationSearch(api),
            checkpoint_path=output_dir / "poems-collection.json",
        ).collect(manifest.poems, force=force)
    except (ValueError, OSError, ConcurrentWriteError) as error:
        raise click.ClickException(str(error)) from error
    finally:
        if api is not None:
            api.close()
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


@stt_group.command("preflight")
@click.argument("manifest_path", type=click.Path(path_type=Path, dir_okay=False))
@click.option(
    "--dataset-root",
    type=click.Path(path_type=Path, file_okay=False),
    help="Explicitly remap the dataset root; declared hashes are verified by default.",
)
@click.option("--verify-hashes/--no-verify-hashes", default=False, show_default=True)
def preflight_stt_manifest(
    manifest_path: Path,
    dataset_root: Path | None,
    verify_hashes: bool,
) -> None:
    """Validate one strict STT manifest and its audio files."""

    try:
        manifest = load_manifest(
            manifest_path,
            dataset_root_override=dataset_root,
            verify_hashes_on_override=True,
        )
        manifest.preflight(verify_hashes=verify_hashes or dataset_root is not None)
    except ManifestValidationError as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"schema_version={manifest.schema_version}")
    click.echo(f"dataset_root={manifest.dataset_root}")
    click.echo(f"samples={len(manifest)}")
    click.echo("preflight=passed")


@stt_group.command("benchmark")
@click.argument("manifest_path", type=click.Path(path_type=Path, dir_okay=False))
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False),
    required=True,
)
@click.option(
    "--backend",
    type=click.Choice(["whisper", "parakeet", "canary"]),
    default="whisper",
    show_default=True,
)
@click.option("--model", help="Model name; defaults are backend-specific.")
@click.option("--language", default="es", show_default=True)
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
    help="Absolute Python executable for an isolated Parakeet or Canary worker.",
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
    backend: str,
    model: str | None,
    language: str,
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
        manifest = load_manifest(manifest_path)
        selected_backend = create_asr_backend(
            backend=cast(BackendName, backend),
            model=model,
            language=language,
            device=device,
            precision=precision,
            worker_python=worker_python,
            request_timeout_seconds=request_timeout,
            timestamps=timestamps,
            short_audio_policy=short_audio_policy,
            minimum_audio_seconds=minimum_audio_seconds,
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
