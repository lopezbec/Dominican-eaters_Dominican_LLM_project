from __future__ import annotations

from pathlib import Path

import pytest

from dominican_eaters.data import ManifestValidationError, discover_stt_manifest


def test_discovers_audio_recursively_with_sidecar_references(tmp_path: Path) -> None:
    nested = tmp_path / "album"
    nested.mkdir()
    (nested / "song.wav").write_bytes(b"wave")
    (nested / "song.txt").write_text("Hola, mundo", encoding="utf-8")
    (tmp_path / "other.mp3").write_bytes(b"mp3")
    (tmp_path / "ignored.json").write_text("{}", encoding="utf-8")

    manifest = discover_stt_manifest(tmp_path)

    assert [sample.audio_path for sample in manifest] == ["album/song.wav", "other.mp3"]
    assert manifest.samples[0].reference_text == "Hola, mundo"
    assert manifest.samples[1].reference_text == ""
    assert all(sample.sha256 for sample in manifest)
    manifest.preflight(verify_hashes=True)


def test_discovery_can_require_sidecar_references(tmp_path: Path) -> None:
    (tmp_path / "song.wav").write_bytes(b"wave")

    with pytest.raises(ManifestValidationError, match="missing sidecar"):
        discover_stt_manifest(tmp_path, require_references=True)


def test_discovery_rejects_empty_audio_directory(tmp_path: Path) -> None:
    with pytest.raises(ManifestValidationError, match="no supported audio"):
        discover_stt_manifest(tmp_path)
