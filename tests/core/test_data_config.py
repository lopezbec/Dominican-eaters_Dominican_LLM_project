from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from dominican_eaters.config import ConfigError, load_config
from dominican_eaters.data import (
    AudioSample,
    ManifestPreflightError,
    ManifestValidationError,
    STTManifest,
    atomic_write_json,
    load_manifest,
    write_manifest,
)


def _sample(path: str = "speaker-a/clip.wav", *, checksum: str | None = None) -> AudioSample:
    return AudioSample(
        sample_id="sample-001",
        audio_path=path,
        reference_text="Hola, mundo.",
        group_id="speaker-a",
        split="test",
        source="human-reviewed",
        sha256=checksum,
    )


def test_manifest_roundtrip_is_replayable_from_a_different_directory(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    audio = dataset_root / "speaker-a" / "clip.wav"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"audio")
    checksum = sha256(b"audio").hexdigest()
    manifest = STTManifest(
        dataset_root=dataset_root.resolve(), samples=(_sample(checksum=checksum),)
    )

    first_path = tmp_path / "inputs" / "manifest.json"
    write_manifest(manifest, first_path)
    snapshot_path = tmp_path / "runs" / "run-42" / "frozen-inputs.json"
    snapshot_path.parent.mkdir(parents=True)
    snapshot_path.write_bytes(first_path.read_bytes())

    replayed = load_manifest(snapshot_path)
    replayed.preflight(verify_hashes=True)
    assert replayed.dataset_root == dataset_root.resolve()
    assert replayed.resolve_audio("sample-001") == audio.resolve()
    assert replayed.sample_ids == ("sample-001",)


def test_relative_declared_root_is_relative_to_manifest_not_working_directory(
    tmp_path: Path,
) -> None:
    manifest_dir = tmp_path / "definition"
    audio = manifest_dir / "dataset" / "speaker-a" / "clip.wav"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"audio")
    manifest_path = manifest_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_root": "dataset",
                "samples": [
                    {
                        "sample_id": "sample-001",
                        "audio_path": "speaker-a/clip.wav",
                        "reference_text": "Hola",
                        "group_id": None,
                        "split": None,
                        "source": None,
                        "sha256": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    loaded = load_manifest(manifest_path)
    assert loaded.resolve_audio(loaded[0]) == audio.resolve()


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda raw: raw.update(extra=True), "unknown fields"),
        (lambda raw: raw.pop("dataset_root"), "missing fields"),
        (lambda raw: raw.update(schema_version=2), "schema_version"),
        (lambda raw: raw["samples"][0].update(unexpected_audio_path="x"), "unknown fields"),
        (lambda raw: raw["samples"][0].pop("reference_text"), "missing fields"),
    ],
)
def test_manifest_rejects_unknown_missing_and_wrong_version_fields(
    tmp_path: Path, mutation: object, message: str
) -> None:
    raw = {
        "schema_version": 1,
        "dataset_root": str(tmp_path),
        "samples": [{"sample_id": "id", "audio_path": "audio.wav", "reference_text": "text"}],
    }
    mutation(raw)  # type: ignore[operator]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ManifestValidationError, match=message):
        load_manifest(path)


def test_sample_identity_is_not_an_audio_path_and_paths_cannot_escape(tmp_path: Path) -> None:
    with pytest.raises(ManifestValidationError, match="distinct"):
        AudioSample(sample_id="a.wav", audio_path="a.wav", reference_text="text")
    with pytest.raises(ManifestValidationError, match="stay below"):
        AudioSample(sample_id="id", audio_path="../a.wav", reference_text="text")
    with pytest.raises(ManifestValidationError, match="relative"):
        AudioSample(sample_id="id", audio_path=str(tmp_path / "a.wav"), reference_text="text")


def test_duplicate_basenames_are_valid_but_duplicate_ids_are_not(tmp_path: Path) -> None:
    samples = (
        AudioSample(sample_id="one", audio_path="a/clip.wav", reference_text="one"),
        AudioSample(sample_id="two", audio_path="b/clip.wav", reference_text="two"),
    )
    assert len(STTManifest(dataset_root=tmp_path.resolve(), samples=samples)) == 2
    with pytest.raises(ManifestValidationError, match="duplicate sample_id"):
        STTManifest(
            dataset_root=tmp_path.resolve(),
            samples=(samples[0], AudioSample("one", "b/clip.wav", "two")),
        )


def test_explicit_root_remap_verifies_content_hashes(tmp_path: Path) -> None:
    old_root = tmp_path / "old"
    new_root = tmp_path / "new"
    relative = Path("speaker-a/clip.wav")
    (new_root / relative).parent.mkdir(parents=True)
    (new_root / relative).write_bytes(b"wrong")
    manifest = STTManifest(
        dataset_root=old_root.resolve(),
        samples=(_sample(checksum=sha256(b"right").hexdigest()),),
    )

    with pytest.raises(ManifestPreflightError, match="sha256 mismatch"):
        manifest.with_dataset_root(new_root)

    without_hash = STTManifest(dataset_root=old_root.resolve(), samples=(_sample(),))
    with pytest.raises(ManifestPreflightError, match="sha256 is required"):
        without_hash.with_dataset_root(new_root)


def test_config_paths_have_explicit_config_and_override_bases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / "configuration"
    config_dir.mkdir()
    path = config_dir / "app.yaml"
    path.write_text(
        "schema_version: 1\ndata_root: ../datasets\nartifacts_root: ../outputs\n",
        encoding="utf-8",
    )
    working = tmp_path / "working"
    working.mkdir()
    monkeypatch.chdir(working)

    selected = load_config(path)
    overridden = load_config(path, data_root="override-data")
    assert selected.data_root == (tmp_path / "datasets").resolve()
    assert selected.artifacts_root == (tmp_path / "outputs").resolve()
    assert overridden.data_root == (working / "override-data").resolve()
    assert overridden.dataset_root("stt") == (working / "override-data" / "stt").resolve()


@pytest.mark.parametrize(
    "contents, message",
    [
        ("schema_version: 1\ndata_root: data\n", "missing fields"),
        (
            "schema_version: 1\ndata_root: data\nartifacts_root: artifacts\nunexpected: true\n",
            "unknown fields",
        ),
        ("schema_version: 2\ndata_root: data\nartifacts_root: artifacts\n", "schema_version"),
        ("schema_version: 1\ndata_root: ''\nartifacts_root: artifacts\n", "data_root"),
    ],
)
def test_config_strictly_rejects_invalid_shapes(
    tmp_path: Path, contents: str, message: str
) -> None:
    path = tmp_path / "app.yaml"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load_config(path)


def test_atomic_json_rejects_nonfinite_values_without_replacing_prior_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "artifact.json"
    atomic_write_json(path, {"value": 1})

    with pytest.raises(ValueError, match="compliant"):
        atomic_write_json(path, {"value": float("nan")})

    assert json.loads(path.read_text(encoding="utf-8")) == {"value": 1}
