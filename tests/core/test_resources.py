from __future__ import annotations

import subprocess
import sys
import time

import pytest

from dominican_eaters.evaluation.resources import measure_call


def test_measure_call_returns_value_monotonic_time_and_rss() -> None:
    value, measurement = measure_call(lambda: (time.sleep(0.001), "done")[1])

    assert value == "done"
    assert measurement.elapsed_seconds >= 0.001
    assert measurement.process_rss_peak_bytes is not None
    assert measurement.process_rss_peak_bytes > 0
    assert measurement.process_tree_rss_peak_bytes is not None
    assert measurement.process_tree_rss_peak_bytes >= measurement.process_rss_peak_bytes
    assert measurement.rss_sampling_interval_seconds == 0.02
    assert measurement.rss_sampling_error is None


def test_measure_call_rejects_nonpositive_sampling_interval() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        measure_call(lambda: None, sampling_interval_seconds=0)


def test_measure_call_includes_live_descendant_rss() -> None:
    def child_operation() -> None:
        child = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import time; allocation = bytearray(24 * 1024 * 1024); time.sleep(0.15)",
            ]
        )
        assert child.wait(timeout=2) == 0

    _, measurement = measure_call(child_operation, sampling_interval_seconds=0.005)

    assert measurement.process_rss_peak_bytes is not None
    assert measurement.process_tree_rss_peak_bytes is not None
    assert (
        measurement.process_tree_rss_peak_bytes - measurement.process_rss_peak_bytes
        >= 12 * 1024 * 1024
    )
