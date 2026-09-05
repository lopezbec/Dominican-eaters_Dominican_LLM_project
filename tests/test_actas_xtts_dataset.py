"""Lightweight tests for the actas XTTS dataset planner."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from testing.tts.actas_xtts_dataset import (  # noqa: E402
    CHUNK_MANIFEST_FILENAME,
    DocumentEntry,
    VoiceEntry,
    build_synthesis_plan,
    build_chunk_records_for_document,
    build_generated_filename,
    create_chunk_metadata,
    make_voice_id,
    should_log_progress,
    split_text_into_chunks,
    split_text_into_chunks_with_offsets,
    unique_rows_by_youtube,
    validate_workspace,
    write_csv,
    write_jsonl,
    SpreadsheetRow,
)


def test_split_text_into_chunks_respects_boundaries() -> None:
    text = (
        "Primera oración con suficiente longitud para un chunk razonable. "
        "Segunda oración también bastante larga para sumar contenido útil. "
        "Tercera oración para forzar otro bloque de salida."
    )
    chunks = split_text_into_chunks(text, max_chars=90, min_chars=30)
    assert len(chunks) >= 2
    assert all(chunk.strip() for chunk in chunks)
    assert all(len(chunk) <= 112 for chunk in chunks)


def test_split_text_into_chunks_with_offsets_preserves_original_text() -> None:
    text = "  Primera línea.\nSegunda   línea con   espacios.  "
    chunks = split_text_into_chunks_with_offsets(text, max_chars=80, min_chars=10)
    assert len(chunks) == 1
    assert chunks[0]["original_text"] == "Primera línea.\nSegunda   línea con   espacios."
    assert chunks[0]["normalized_text"] == "Primera línea. Segunda línea con espacios."
    assert chunks[0]["source_char_start_local"] == 2


def test_unique_rows_by_youtube_deduplicates_voices() -> None:
    rows = [
        SpreadsheetRow(2, "a.pdf", "pdf-a", "https://youtube.com/watch?v=abc12345678"),
        SpreadsheetRow(3, "b.pdf", "pdf-b", "https://youtube.com/watch?v=abc12345678"),
        SpreadsheetRow(4, "c.pdf", "pdf-c", "https://youtube.com/watch?v=xyz12345678"),
    ]
    unique = unique_rows_by_youtube(rows)
    assert len(unique) == 2
    assert make_voice_id(unique[0]).startswith("voice_")


def test_build_synthesis_plan_creates_cartesian_pairs() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        chunks_dir = root / "chunks"
        synth_dir = root / "synthetic_audio"

        doc_a_dir = chunks_dir / "doc-a"
        doc_b_dir = chunks_dir / "doc-b"
        doc_a_dir.mkdir(parents=True)
        doc_b_dir.mkdir(parents=True)
        (doc_a_dir / "chunk_0001.txt").write_text("texto a1", encoding="utf-8")
        (doc_a_dir / "chunk_0002.txt").write_text("texto a2", encoding="utf-8")
        (doc_b_dir / "chunk_0001.txt").write_text("texto b1", encoding="utf-8")
        write_jsonl(
            doc_a_dir / CHUNK_MANIFEST_FILENAME,
            [
                {
                    "chunk_id": "chunk_0001",
                    "chunk_index": 1,
                    "source_document_id": "doc-a",
                    "source_document_name": "a.pdf",
                    "source_document_path": "/tmp/a.pdf",
                    "source_document_type": "pdf",
                    "source_page_number": 1,
                    "source_line_start": None,
                    "source_line_end": None,
                    "source_char_start": 0,
                    "source_char_end": 8,
                    "chunk_text_original": "texto a1",
                    "chunk_text_normalized_for_tts": "texto a1",
                    "chunk_char_count": 8,
                    "chunk_word_count": 2,
                },
                {
                    "chunk_id": "chunk_0002",
                    "chunk_index": 2,
                    "source_document_id": "doc-a",
                    "source_document_name": "a.pdf",
                    "source_document_path": "/tmp/a.pdf",
                    "source_document_type": "pdf",
                    "source_page_number": 2,
                    "source_line_start": None,
                    "source_line_end": None,
                    "source_char_start": 10,
                    "source_char_end": 18,
                    "chunk_text_original": "texto a2",
                    "chunk_text_normalized_for_tts": "texto a2",
                    "chunk_char_count": 8,
                    "chunk_word_count": 2,
                },
            ],
        )
        write_jsonl(
            doc_b_dir / CHUNK_MANIFEST_FILENAME,
            [
                {
                    "chunk_id": "chunk_0001",
                    "chunk_index": 1,
                    "source_document_id": "doc-b",
                    "source_document_name": "b.pdf",
                    "source_document_path": "/tmp/b.pdf",
                    "source_document_type": "pdf",
                    "source_page_number": 3,
                    "source_line_start": None,
                    "source_line_end": None,
                    "source_char_start": 0,
                    "source_char_end": 8,
                    "chunk_text_original": "texto b1",
                    "chunk_text_normalized_for_tts": "texto b1",
                    "chunk_char_count": 8,
                    "chunk_word_count": 2,
                }
            ],
        )

        voices = [
            VoiceEntry(
                "voice_001", "https://youtu.be/1", "/tmp/voice1.wav", "a.pdf", 2
            ),
            VoiceEntry(
                "voice_002", "https://youtu.be/2", "/tmp/voice2.wav", "b.pdf", 3
            ),
        ]
        documents = [
            DocumentEntry("doc-a", "a.pdf", "/tmp/a.pdf", "/tmp/a.txt", 2, 100, 2),
            DocumentEntry("doc-b", "b.pdf", "/tmp/b.pdf", "/tmp/b.txt", 1, 80, 3),
        ]

        plan = build_synthesis_plan(voices, documents, chunks_dir, synth_dir)
        assert len(plan) == 6
        assert plan[0]["voice_id"] == "voice_001"
        assert plan[0]["document_id"] == "doc-a"
        assert plan[-1]["voice_id"] == "voice_002"
        assert plan[-1]["document_id"] == "doc-b"
        assert plan[-1]["output_wav"].endswith(".wav")
        assert plan[0]["generated_filename"].startswith("voice_001__doc-a__p0001__chunk_0001")
        assert plan[0]["chunk_text_original"] == "texto a1"
        assert plan[0]["source_page_number"] == 1


def test_build_chunk_records_for_document_tracks_pdf_page_offsets() -> None:
    document_text, chunk_rows = build_chunk_records_for_document(
        document_id="doc-a",
        document_name="a.pdf",
        source_document_path=Path("/tmp/a.pdf"),
        pages_payload=[
            {"page_number": 1, "text": "Primera página."},
            {"page_number": 2, "text": "Segunda página."},
        ],
        max_chars=80,
        min_chars=10,
    )
    assert document_text == "Primera página.\n\nSegunda página."
    assert len(chunk_rows) == 2
    assert chunk_rows[0]["source_page_number"] == 1
    assert chunk_rows[1]["source_page_number"] == 2
    assert chunk_rows[1]["source_char_start"] == len("Primera página.\n\n")


def test_create_chunk_metadata_exposes_traceability_fields() -> None:
    item = {
        "sample_id": "voice__doc__chunk_0001",
        "voice_id": "voice",
        "speaker_wav": "/tmp/voice.wav",
        "youtube_url": "https://youtu.be/example",
        "source_document_id": "doc",
        "source_document_name": "Acta.pdf",
        "source_document_path": "/tmp/Acta.pdf",
        "source_document_type": "pdf",
        "source_page_number": 4,
        "source_line_start": None,
        "source_line_end": None,
        "source_char_start": 120,
        "source_char_end": 180,
        "chunk_id": "chunk_0001",
        "chunk_index": 1,
        "chunk_text_original": "Texto original",
        "chunk_text_normalized_for_tts": "Texto original",
        "chunk_char_count": 14,
        "chunk_word_count": 2,
        "generated_filename": build_generated_filename("voice", "doc", "chunk_0001", 4),
        "output_wav": "/tmp/out.wav",
    }
    record = create_chunk_metadata(
        item=item,
        tts_model_name="xtts",
        tts_model_version="0.22.0",
        generation_status="failed",
        generation_started_at="2026-06-24T10:00:00+00:00",
        generation_finished_at="2026-06-24T10:00:02+00:00",
        generation_runtime_seconds=2.0,
        error_message="boom",
        status="error",
    )
    assert record["synthetic_audio_id"] == "voice__doc__chunk_0001"
    assert record["generated_filename"].endswith(".wav")
    assert record["generation_status"] == "failed"
    assert record["error_message"] == "boom"
    assert record["source_page_number"] == 4


def test_build_synthesis_plan_random_selection_is_reproducible() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        chunks_dir = root / "chunks"
        synth_dir = root / "synthetic_audio"
        doc_dir = chunks_dir / "doc-a"
        doc_dir.mkdir(parents=True)
        for i in range(1, 6):
            (doc_dir / f"chunk_{i:04d}.txt").write_text(f"texto {i}", encoding="utf-8")

        voices = [VoiceEntry("voice_001", "https://youtu.be/1", "/tmp/voice1.wav", "a.pdf", 2)]
        documents = [DocumentEntry("doc-a", "a.pdf", "/tmp/a.pdf", "/tmp/a.txt", 5, 100, 2)]

        plan_a = build_synthesis_plan(
            voices, documents, chunks_dir, synth_dir, selection_mode="random", chunks_per_doc=2, random_seed=99
        )
        plan_b = build_synthesis_plan(
            voices, documents, chunks_dir, synth_dir, selection_mode="random", chunks_per_doc=2, random_seed=99
        )
        assert [row["chunk_id"] for row in plan_a] == [row["chunk_id"] for row in plan_b]
        assert len(plan_a) == 2


def test_build_synthesis_plan_manual_selection_and_limit() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        chunks_dir = root / "chunks"
        synth_dir = root / "synthetic_audio"
        doc_dir = chunks_dir / "doc-a"
        doc_dir.mkdir(parents=True)
        for i in range(1, 5):
            (doc_dir / f"chunk_{i:04d}.txt").write_text(f"texto {i}", encoding="utf-8")

        voices = [VoiceEntry("voice_001", "https://youtu.be/1", "/tmp/voice1.wav", "a.pdf", 2)]
        documents = [DocumentEntry("doc-a", "a.pdf", "/tmp/a.pdf", "/tmp/a.txt", 4, 100, 2)]

        plan = build_synthesis_plan(
            voices,
            documents,
            chunks_dir,
            synth_dir,
            selection_mode="manual",
            selected_chunk_ids=["chunk_0002", "chunk_0004"],
            chunks_per_doc=1,
        )
        assert len(plan) == 1
        assert plan[0]["chunk_id"] == "chunk_0002"


def test_metadata_export_helpers_write_expected_rows() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        rows = [{"sample_id": "a", "status": "ok"}, {"sample_id": "b", "status": "skipped"}]
        jsonl_path = root / "rows.jsonl"
        csv_path = root / "rows.csv"
        write_jsonl(jsonl_path, rows)
        write_csv(csv_path, rows)
        jsonl_lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(jsonl_lines) == 2
        assert "sample_id" in csv_path.read_text(encoding="utf-8")


def test_validate_workspace_flags_parity_and_missing_outputs() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        manifests = root / "manifests"
        manifests.mkdir(parents=True)
        text_path = root / "chunk_0001.txt"
        text_path.write_text("texto válido", encoding="utf-8")
        plan = [
            {
                "sample_id": "voice__doc__chunk_0001",
                "text_path": str(text_path),
                "output_wav": str(root / "out.wav"),
            }
        ]
        results = [{"sample_id": "voice__doc__chunk_0001", "output_wav": str(root / "out.wav"), "status": "ok"}]
        (manifests / "synthesis_plan.json").write_text(json.dumps(plan), encoding="utf-8")
        (manifests / "synthesis_results.json").write_text(json.dumps(results), encoding="utf-8")

        summary = validate_workspace(
            SimpleNamespace(workspace_dir=str(root), chunk_max_chars=180)
        )
        assert summary["valid"] is False
        issues = json.loads((manifests / "validation_issues.json").read_text(encoding="utf-8"))
        assert any("missing_output_for_ok" in issue for issue in issues)


def test_should_log_progress_handles_first_last_and_intervals() -> None:
    assert should_log_progress(1, 10, 5) is True
    assert should_log_progress(5, 10, 5) is True
    assert should_log_progress(10, 10, 5) is True
    assert should_log_progress(3, 10, 5) is False


if __name__ == "__main__":
    print("Running actas XTTS dataset tests...")
    test_split_text_into_chunks_respects_boundaries()
    test_split_text_into_chunks_with_offsets_preserves_original_text()
    test_unique_rows_by_youtube_deduplicates_voices()
    test_build_synthesis_plan_creates_cartesian_pairs()
    test_build_chunk_records_for_document_tracks_pdf_page_offsets()
    test_create_chunk_metadata_exposes_traceability_fields()
    test_build_synthesis_plan_random_selection_is_reproducible()
    test_build_synthesis_plan_manual_selection_and_limit()
    test_metadata_export_helpers_write_expected_rows()
    test_validate_workspace_flags_parity_and_missing_outputs()
    test_should_log_progress_handles_first_last_and_intervals()
    print("All actas XTTS dataset tests passed!")
