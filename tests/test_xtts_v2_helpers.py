"""Lightweight tests for XTTS local artifact helpers."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from testing.tts.xtts_v2 import (  # noqa: E402
    resolve_local_model_artifacts,
    validate_config_json,
)


def test_resolve_local_model_artifacts_from_directory() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        model_dir = Path(temp_dir)
        config = model_dir / "config.json"
        checkpoint = model_dir / "model.pth"
        config.write_text(json.dumps({"model": "xtts"}), encoding="utf-8")
        checkpoint.write_bytes(b"checkpoint")
        (model_dir / "vocab.json").write_text("{}", encoding="utf-8")
        (model_dir / "speakers_xtts.pth").write_bytes(b"speakers")

        resolved_target, resolved_config, use_directory_target = (
            resolve_local_model_artifacts(str(model_dir))
        )
        assert resolved_target == model_dir.resolve()
        assert resolved_config == config.resolve()
        assert use_directory_target is True


def test_validate_config_json_fails_for_empty_file() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        config = Path(temp_dir) / "config.json"
        config.write_text("", encoding="utf-8")

        try:
            validate_config_json(config)
        except RuntimeError as exc:
            assert "empty" in str(exc).lower()
        else:
            raise AssertionError("Expected RuntimeError for empty config.json")


if __name__ == "__main__":
    print("Running XTTS helper tests...")
    test_resolve_local_model_artifacts_from_directory()
    test_validate_config_json_fails_for_empty_file()
    print("All XTTS helper tests passed!")
