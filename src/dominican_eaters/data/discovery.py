"""Deterministic STT manifest discovery from an existing audio directory."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from .manifest import AudioSample, ManifestValidationError, STTManifest

AUDIO_SUFFIXES = frozenset({".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav"})


def discover_stt_manifest(
    audio_dir: Path,
    *,
    require_references: bool = False,
) -> STTManifest:
    """Build a stable manifest, using same-name text files as references when present."""

    root = audio_dir.expanduser().resolve()
    if not root.is_dir():
        raise ManifestValidationError(f"audio directory does not exist: {root}")
    audio_paths = sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
    )
    if not audio_paths:
        raise ManifestValidationError(f"no supported audio files found below: {root}")

    samples: list[AudioSample] = []
    missing_references: list[str] = []
    for audio_path in audio_paths:
        relative = audio_path.relative_to(root)
        reference_path = audio_path.with_suffix(".txt")
        if reference_path.is_file():
            reference_text = reference_path.read_text(encoding="utf-8").strip()
        else:
            reference_text = ""
            missing_references.append(relative.as_posix())
        samples.append(
            AudioSample(
                sample_id=relative.with_suffix("").as_posix().replace("/", "__"),
                audio_path=relative.as_posix(),
                reference_text=reference_text,
                source="directory-discovery",
                sha256=_file_sha256(audio_path),
            )
        )
    if require_references and missing_references:
        preview = ", ".join(missing_references[:5])
        remainder = len(missing_references) - 5
        suffix = f" (and {remainder} more)" if remainder > 0 else ""
        raise ManifestValidationError(f"missing sidecar .txt references: {preview}{suffix}")
    return STTManifest(dataset_root=root, samples=tuple(samples))


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
