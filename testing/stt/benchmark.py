"""Manifest-driven Whisper benchmark; no downloads/imports during preflight."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import math
import platform
import random
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

try:
    from .evaluate_with_local_refs import evaluate, load_manifest, resolve_repo_path
except ImportError:
    from evaluate_with_local_refs import evaluate, load_manifest, resolve_repo_path

logger = logging.getLogger(__name__)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def preflight(manifest: Path, split: Optional[str], limit: Optional[int]) -> list:
    if limit is not None and limit <= 0:
        raise ValueError("--max must be positive")
    references = load_manifest(manifest)
    selected = [sample for sample in references.values() if split is None or sample.split == split]
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise ValueError("No samples selected")
    if len({sample.split for sample in selected}) > 1:
        raise ValueError("Select one split with --split; do not pool development and evaluation")
    rows = []
    for sample in selected:
        audio = resolve_repo_path(sample.audio_file, manifest)
        if not audio.is_file():
            raise ValueError(f"Audio not found: {audio}")
        # Freeze loaded reference text into the snapshot. Evaluation must not
        # reread a mutable external text file after inference has finished.
        rows.append({**asdict(sample), "reference_text_path": "",
                     "original_reference_text_path": sample.reference_text_path,
                     "audio_path": str(audio.resolve()),
                     "audio_sha256": sha256(audio)})
    return rows


def measure_call(call: Callable, torch: object, device: str, interval: float) -> Tuple[object, dict]:
    """Wall time with CUDA synchronization and sampled current-process RSS.

    RSS excludes subprocesses. CUDA peaks include the resident model, but not
    allocations outside PyTorch. The sampler is always stopped on failure.
    """
    import psutil

    process = psutil.Process()
    stop = threading.Event()
    rss = [process.memory_info().rss]
    sampling_errors = []

    def sample() -> None:
        try:
            rss[0] = max(rss[0], process.memory_info().rss)
        except psutil.Error as exc:
            sampling_errors.append(str(exc))

    def poll() -> None:
        while not stop.wait(interval):
            sample()

    cuda = device == "cuda"
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    worker = threading.Thread(target=poll, daemon=True)
    worker.start()
    start = time.perf_counter()
    try:
        result = call()
        if cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    finally:
        stop.set()
        worker.join()
        sample()
    return result, {
        "inference_seconds": elapsed,
        "ram_peak_rss_bytes": rss[0] if not sampling_errors else None,
        "ram_sampling_error": sampling_errors or None,
        "ram_sample_interval_seconds": interval,
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated() if cuda else None,
        "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved() if cuda else None,
        "gpu_measurement_note": "PyTorch allocator only" if cuda else "CPU run; CUDA not applicable",
    }


def run(args: argparse.Namespace) -> dict:
    manifest = Path(args.manifest).resolve()
    rows = preflight(manifest, args.split, args.max)
    if args.preflight:
        logger.info("Preflight passed for %d files; no audio decoding or model inference performed", len(rows))
        return {"status": "preflight", "samples": len(rows)}
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is not empty; choose a fresh directory to prevent stale results")

    import numpy as np
    import torch
    import whisper

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing silent CPU fallback")
    if device == "cpu" and args.precision == "fp16":
        raise ValueError("FP16 requires CUDA")
    fp16 = device == "cuda" and args.precision != "fp32"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output.mkdir(parents=True, exist_ok=True)
    transcripts = output / "transcriptions" / "whisper"
    transcripts.mkdir(parents=True)
    selected_manifest = output / "manifest.json"
    write_json(selected_manifest, {"rows": rows})
    versions = {}
    for package in ("openai-whisper", "torch", "numpy", "psutil"):
        versions[package] = importlib.metadata.version(package)
    metadata = {
        "status": "running", "model": args.model, "device": device,
        "decoding_precision": "fp16" if fp16 else "fp32", "batch_size": 1,
        "language": args.language, "seed": args.seed, "temperature": 0.0,
        "condition_on_previous_text": False, "python": platform.python_version(),
        "platform": platform.platform(), "packages": versions,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
        "manifest_sha256": sha256(manifest), "selected_manifest_sha256": sha256(selected_manifest),
        "timing_scope": "in-memory 16 kHz audio through transcription; excludes file decoding, hashing, scoring, and JSON writes",
        "ram_scope": "sampled RSS of current process only; short peaks may be missed",
        "warmup_runs": args.warmup, "arguments": vars(args),
    }
    write_json(output / "run.json", metadata)
    try:
        model, loading = measure_call(lambda: whisper.load_model(args.model, device=device), torch, device, args.ram_interval)
        metadata["model_loading"] = loading
        checkpoint = Path(args.model).expanduser()
        # Whisper's named checkpoint URLs embed their SHA256; local files are hashed.
        url = getattr(whisper, "_MODELS", {}).get(args.model)
        metadata["checkpoint_sha256"] = sha256(checkpoint) if checkpoint.is_file() else (url.split("/")[-2] if url else None)
        options = {"language": args.language, "task": "transcribe", "fp16": fp16,
                   "verbose": None, "temperature": 0.0, "condition_on_previous_text": False}
        # A synthetic warmup is never included in scores or timing summaries.
        for _ in range(args.warmup):
            model.transcribe(np.zeros(16000, dtype=np.float32), **options)
        records = []
        for index, row in enumerate(rows):
            record = {"file": row["audio_file"], "status": "error"}
            try:
                audio = whisper.load_audio(row["audio_path"])
                duration = len(audio) / 16000
                if duration <= 0 or not np.isfinite(audio).all():
                    raise ValueError("Audio must be nonempty and finite")
                result, measurements = measure_call(lambda: model.transcribe(audio, **options), torch, device, args.ram_interval)
                if not isinstance(result.get("text"), str):
                    raise ValueError("Model returned no text field")
                record.update(status="ok", transcript=result["text"], language=result.get("language", args.language),
                              duration_seconds=duration, rtf=measurements["inference_seconds"] / duration,
                              **measurements)
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                logger.exception("Failed to transcribe %s", row["audio_file"])
            write_json(transcripts / f"{index:06d}.json", record)
            records.append(record)
        evaluate(output / "transcriptions", selected_manifest, ["whisper"], output / "scores", args.bootstrap, args.seed)
        successful = [r for r in records if r["status"] == "ok"]
        total_duration = sum(r["duration_seconds"] for r in successful)
        metadata.update(status="complete" if len(successful) == len(rows) else "partial",
                        samples=len(rows), succeeded=len(successful), failed=len(rows) - len(successful),
                        audio_seconds=total_duration,
                        inference_seconds=sum(r["inference_seconds"] for r in successful),
                        rtf=sum(r["inference_seconds"] for r in successful) / total_duration if total_duration else None)
    except Exception as exc:
        metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(output / "run.json", metadata)
    return metadata


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="base")
    parser.add_argument("--language", default="es")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--precision", choices=("auto", "fp16", "fp32"), default="auto")
    parser.add_argument("--split")
    parser.add_argument("--max", type=int)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--ram-interval", type=float, default=0.02)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)
    if args.bootstrap < 0 or args.warmup < 0 or not math.isfinite(args.ram_interval) or args.ram_interval <= 0:
        parser.error("bootstrap/warmup must be nonnegative; ram-interval must be positive and finite")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    result = run(parse_args(argv))
    return 0 if result["status"] in ("complete", "preflight") else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
