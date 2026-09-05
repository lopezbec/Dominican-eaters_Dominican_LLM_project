"""Process resource measurement with explicit scope and units."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import psutil

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class ResourceMeasurement:
    elapsed_seconds: float
    process_rss_peak_bytes: int | None
    process_tree_rss_peak_bytes: int | None
    rss_sampling_interval_seconds: float
    rss_sampling_error: str | None = None


def measure_call(
    operation: Callable[[], T], *, sampling_interval_seconds: float = 0.02
) -> tuple[T, ResourceMeasurement]:
    """Measure wall time plus sampled host and process-tree RSS."""

    if sampling_interval_seconds <= 0:
        raise ValueError("sampling_interval_seconds must be positive")
    process = psutil.Process()
    initial_rss = process.memory_info().rss
    process_peak = [initial_rss]
    tree_peak = [initial_rss]
    errors: list[str] = []
    stop = threading.Event()

    def sample() -> None:
        try:
            host_rss = process.memory_info().rss
            tree_rss = host_rss
            for child in process.children(recursive=True):
                try:
                    tree_rss += child.memory_info().rss
                except psutil.NoSuchProcess:
                    continue
            process_peak[0] = max(process_peak[0], host_rss)
            tree_peak[0] = max(tree_peak[0], tree_rss)
        except psutil.Error as error:
            errors.append(f"{type(error).__name__}: {error}")

    def poll() -> None:
        while not stop.wait(sampling_interval_seconds):
            sample()

    worker = threading.Thread(target=poll, name="rss-sampler", daemon=True)
    worker.start()
    started = time.perf_counter()
    try:
        value = operation()
        elapsed = time.perf_counter() - started
    finally:
        stop.set()
        worker.join()
        sample()
    return value, ResourceMeasurement(
        elapsed_seconds=elapsed,
        process_rss_peak_bytes=None if errors else process_peak[0],
        process_tree_rss_peak_bytes=None if errors else tree_peak[0],
        rss_sampling_interval_seconds=sampling_interval_seconds,
        rss_sampling_error="; ".join(errors) or None,
    )
