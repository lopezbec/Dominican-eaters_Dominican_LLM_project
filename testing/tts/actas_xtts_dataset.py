#!/usr/bin/env python3
"""Prepare and synthesize acta-based XTTS datasets from Excel + PDFs + YouTube."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib.metadata
import json
import logging
import os
import random
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from zipfile import ZipFile
from xml.etree import ElementTree as ET

logger = logging.getLogger(__name__)

CHUNK_MANIFEST_FILENAME = "chunks_manifest.jsonl"

DEFAULT_HEADERS = {
    "document_name": "Nombre y Fecha del Documento",
    "pdf_link": "Link pdf",
    "youtube_url": "Enlace youtube",
}


@dataclass
class SpreadsheetRow:
    row_index: int
    document_name: str
    pdf_link: str
    youtube_url: str


@dataclass
class DocumentEntry:
    document_id: str
    document_name: str
    pdf_path: str
    text_path: str
    chunk_count: int
    char_count: int
    row_index: int


@dataclass
class VoiceEntry:
    voice_id: str
    youtube_url: str
    speaker_wav: str
    source_document_name: str
    row_index: int


def safe_filename(value: str, max_length: int = 180) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    sanitized = sanitized.strip("._")
    return sanitized[:max_length] or "item"


def slugify(value: str, max_length: int = 120) -> str:
    lowered = value.lower().strip()
    lowered = re.sub(r"\s+", "-", lowered)
    lowered = re.sub(r"[^a-z0-9áéíóúüñ_-]", "-", lowered)
    lowered = re.sub(r"-+", "-", lowered).strip("-")
    return lowered[:max_length] or "item"


def normalize_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def count_words(text: str) -> int:
    return len([part for part in re.split(r"\s+", text.strip()) if part])


def build_source_document_id(document_name: str, source_document_path: Optional[str] = None) -> str:
    candidate = Path(source_document_path).stem if source_document_path else Path(document_name).stem
    return slugify(candidate)


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def write_jsonl_record(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def infer_source_document_type(path_value: str) -> Optional[str]:
    suffix = Path(path_value).suffix.lower().lstrip(".")
    return suffix or None


def get_tts_model_version() -> Optional[str]:
    try:
        return importlib.metadata.version("TTS")
    except importlib.metadata.PackageNotFoundError:
        return None


def build_generated_filename(
    voice_reference_id: str,
    source_document_id: str,
    chunk_id: str,
    source_page_number: Optional[int] = None,
) -> str:
    page_part = f"__p{source_page_number:04d}" if source_page_number is not None else ""
    return safe_filename(
        f"{voice_reference_id}__{source_document_id}{page_part}__{chunk_id}.wav"
    )


def split_text_into_chunks_with_offsets(
    text: str,
    max_chars: int = 900,
    min_chars: int = 180,
) -> List[Dict[str, object]]:
    normalized = normalize_text(text)
    if not normalized:
        return []

    normalized_to_original: List[int] = []
    normalized_chars: List[str] = []
    pending_space = False

    for original_index, char in enumerate(text.replace("\r", "\n")):
        if char.isspace():
            pending_space = bool(normalized_chars)
            continue
        if pending_space:
            normalized_chars.append(" ")
            normalized_to_original.append(original_index)
            pending_space = False
        normalized_chars.append(char)
        normalized_to_original.append(original_index)

    if not normalized_chars:
        return []

    cleaned = "".join(normalized_chars).strip()
    if not cleaned:
        return []

    leading_trim = len("".join(normalized_chars)) - len("".join(normalized_chars).lstrip())
    trailing_trim = len("".join(normalized_chars)) - len("".join(normalized_chars).rstrip())
    if leading_trim:
        normalized_to_original = normalized_to_original[leading_trim:]
    if trailing_trim:
        normalized_to_original = normalized_to_original[:-trailing_trim]

    sentence_matches = list(re.finditer(r".*?(?:[\.!?;:](?:\s+|$)|$)", cleaned))
    segments = [match for match in sentence_matches if match.group(0).strip()]
    if not segments:
        segments = [re.match(r".*", cleaned)]  # type: ignore[list-item]

    chunks: List[Tuple[int, int]] = []
    current_start: Optional[int] = None
    current_end = 0

    def append_span(start: int, end: int) -> None:
        if start >= end:
            return
        chunks.append((start, end))

    for segment in segments:
        sentence_start = segment.start()
        sentence_end = segment.end()
        sentence_text = cleaned[sentence_start:sentence_end].strip()
        if not sentence_text:
            continue

        if len(sentence_text) > max_chars:
            words = list(re.finditer(r"\S+", sentence_text))
            partial_start: Optional[int] = None
            partial_end: Optional[int] = None
            for word in words:
                absolute_start = sentence_start + word.start()
                absolute_end = sentence_start + word.end()
                if partial_start is None:
                    partial_start = absolute_start
                    partial_end = absolute_end
                    continue

                candidate = cleaned[partial_start:absolute_end]
                if len(candidate) > max_chars:
                    if current_start is not None:
                        append_span(current_start, current_end)
                        current_start = None
                    append_span(partial_start, partial_end or partial_start)
                    partial_start = absolute_start
                partial_end = absolute_end

            if partial_start is not None:
                if current_start is not None:
                    append_span(current_start, current_end)
                    current_start = None
                append_span(partial_start, partial_end or partial_start)
            continue

        if current_start is None:
            current_start = sentence_start
            current_end = sentence_end
            continue

        candidate = cleaned[current_start:sentence_end]
        if len(candidate) > max_chars:
            append_span(current_start, current_end)
            current_start = sentence_start
            current_end = sentence_end
        else:
            current_end = sentence_end

    if current_start is not None:
        append_span(current_start, current_end)

    merged: List[Tuple[int, int]] = []
    for start, end in chunks:
        if merged and len(cleaned[start:end].strip()) < min_chars:
            previous_start, previous_end = merged[-1]
            candidate = cleaned[previous_start:end].strip()
            if len(candidate) <= max_chars:
                merged[-1] = (previous_start, end)
                continue
        merged.append((start, end))

    chunk_rows: List[Dict[str, object]] = []
    for start, end in merged:
        chunk_text_normalized = cleaned[start:end].strip()
        if not chunk_text_normalized:
            continue
        original_start = normalized_to_original[start]
        original_end = normalized_to_original[end - 1] + 1
        chunk_text_original = text[original_start:original_end].strip()
        chunk_rows.append(
            {
                "original_text": chunk_text_original,
                "normalized_text": chunk_text_normalized,
                "source_char_start_local": original_start,
                "source_char_end_local": original_end,
            }
        )
    return chunk_rows


def split_text_into_chunks(
    text: str,
    max_chars: int = 900,
    min_chars: int = 180,
) -> List[str]:
    return [
        str(row["normalized_text"])
        for row in split_text_into_chunks_with_offsets(
            text,
            max_chars=max_chars,
            min_chars=min_chars,
        )
    ]


def _read_xlsx_rows_with_xml(excel_path: Path) -> List[SpreadsheetRow]:
    namespace = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with ZipFile(excel_path) as archive:
        shared_strings: List[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("a:si", namespace):
                shared_strings.append(
                    "".join(
                        node.text or "" for node in item.iterfind(".//a:t", namespace)
                    )
                )

        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))

    def parse_cell_value(cell: ET.Element) -> str:
        raw = cell.find("a:v", namespace)
        if raw is None:
            return ""
        value = raw.text or ""
        if cell.attrib.get("t") == "s":
            return shared_strings[int(value)]
        return value

    def cell_column(reference: str) -> str:
        match = re.match(r"([A-Z]+)", reference)
        return match.group(1) if match else ""

    rows = sheet.findall(".//a:sheetData/a:row", namespace)
    if not rows:
        return []

    header_map: Dict[str, str] = {}
    for cell in rows[0].findall("a:c", namespace):
        header_map[cell_column(cell.attrib.get("r", ""))] = parse_cell_value(cell)

    inverse_headers = {v: k for k, v in header_map.items()}
    required = [
        DEFAULT_HEADERS["document_name"],
        DEFAULT_HEADERS["pdf_link"],
        DEFAULT_HEADERS["youtube_url"],
    ]
    missing = [name for name in required if name not in inverse_headers]
    if missing:
        raise ValueError(f"Excel is missing required columns: {missing}")

    data_rows: List[SpreadsheetRow] = []
    for row in rows[1:]:
        values_by_col: Dict[str, str] = {}
        for cell in row.findall("a:c", namespace):
            values_by_col[cell_column(cell.attrib.get("r", ""))] = parse_cell_value(
                cell
            )

        document_name = values_by_col.get(
            inverse_headers[DEFAULT_HEADERS["document_name"]], ""
        ).strip()
        pdf_link = values_by_col.get(
            inverse_headers[DEFAULT_HEADERS["pdf_link"]], ""
        ).strip()
        youtube_url = values_by_col.get(
            inverse_headers[DEFAULT_HEADERS["youtube_url"]], ""
        ).strip()
        if not document_name or not youtube_url:
            continue

        row_index = int(row.attrib.get("r", "0"))
        data_rows.append(
            SpreadsheetRow(
                row_index=row_index,
                document_name=document_name,
                pdf_link=pdf_link,
                youtube_url=youtube_url,
            )
        )

    return data_rows


def load_spreadsheet_rows(excel_path: Path) -> List[SpreadsheetRow]:
    try:
        import pandas as pd

        frame = pd.read_excel(excel_path)
        missing = [
            value for value in DEFAULT_HEADERS.values() if value not in frame.columns
        ]
        if missing:
            raise ValueError(f"Excel is missing required columns: {missing}")

        rows: List[SpreadsheetRow] = []
        for frame_index, row in frame.iterrows():
            document_name = str(
                row.get(DEFAULT_HEADERS["document_name"], "") or ""
            ).strip()
            pdf_link = str(row.get(DEFAULT_HEADERS["pdf_link"], "") or "").strip()
            youtube_url = str(row.get(DEFAULT_HEADERS["youtube_url"], "") or "").strip()
            if not document_name or not youtube_url:
                continue
            rows.append(
                SpreadsheetRow(
                    row_index=int(frame_index) + 2,
                    document_name=document_name,
                    pdf_link=pdf_link,
                    youtube_url=youtube_url,
                )
            )
        return rows
    except Exception as exc:
        logger.warning(
            "Falling back to stdlib XLSX parser for %s (%s)", excel_path, exc
        )
        return _read_xlsx_rows_with_xml(excel_path)


def build_pdf_lookup(
    pdf_dir: Optional[Path], pdf_zip: Optional[Path]
) -> Dict[str, Dict[str, str]]:
    lookup: Dict[str, Dict[str, str]] = {}
    if pdf_dir and pdf_dir.exists():
        for path in pdf_dir.glob("*.pdf"):
            lookup[path.name] = {"kind": "dir", "path": str(path.resolve())}
    if pdf_zip and pdf_zip.exists():
        with ZipFile(pdf_zip) as archive:
            for member in archive.namelist():
                if member.endswith("/"):
                    continue
                name = Path(member).name
                if name.lower().endswith(".pdf"):
                    lookup.setdefault(
                        name,
                        {
                            "kind": "zip",
                            "member": member,
                            "zip_path": str(pdf_zip.resolve()),
                        },
                    )
    return lookup


def materialize_pdf(source: Dict[str, str], destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    if source["kind"] == "dir":
        return Path(source["path"])

    zip_path = Path(source["zip_path"])
    member = source["member"]
    output_path = destination_dir / Path(member).name
    if output_path.exists():
        return output_path

    with ZipFile(zip_path) as archive:
        with archive.open(member) as source_file, output_path.open("wb") as target_file:
            target_file.write(source_file.read())
    return output_path


def extract_pdf_pages(pdf_path: Path) -> List[Dict[str, object]]:
    try:
        import pdfplumber
    except ImportError as exc:
        raise ImportError("pdfplumber is required to extract acta text") from exc

    pages_payload: List[Dict[str, object]] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            if text.strip():
                pages_payload.append(
                    {"page_number": page_number, "text": text.strip()}
                )

    return pages_payload


def build_chunk_records_for_document(
    document_id: str,
    document_name: str,
    source_document_path: Path,
    pages_payload: Sequence[Dict[str, object]],
    max_chars: int,
    min_chars: int,
) -> Tuple[str, List[Dict[str, object]]]:
    chunk_rows: List[Dict[str, object]] = []
    document_parts: List[str] = []
    page_offsets: Dict[int, int] = {}
    current_offset = 0

    for index, page_payload in enumerate(pages_payload):
        page_number = int(page_payload["page_number"])
        page_text_original = str(page_payload["text"])
        if not page_text_original.strip():
            continue
        if index > 0:
            current_offset += 2
        page_offsets[page_number] = current_offset
        document_parts.append(page_text_original)
        current_offset += len(page_text_original)

    document_text_original = "\n\n".join(document_parts)
    source_document_type = infer_source_document_type(str(source_document_path))
    chunk_counter = 0

    for page_payload in pages_payload:
        page_number = int(page_payload["page_number"])
        page_text_original = str(page_payload["text"])
        page_offset = page_offsets.get(page_number)
        if page_offset is None:
            continue

        page_chunk_rows = split_text_into_chunks_with_offsets(
            page_text_original,
            max_chars=max_chars,
            min_chars=min_chars,
        )
        for page_chunk in page_chunk_rows:
            chunk_counter += 1
            chunk_id = f"chunk_{chunk_counter:04d}"
            chunk_rows.append(
                {
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_counter,
                    "source_document_id": document_id,
                    "source_document_name": document_name,
                    "source_document_path": str(source_document_path.resolve()),
                    "source_document_type": source_document_type,
                    "source_page_number": page_number,
                    "source_line_start": None,
                    "source_line_end": None,
                    "source_char_start": page_offset
                    + int(page_chunk["source_char_start_local"]),
                    "source_char_end": page_offset
                    + int(page_chunk["source_char_end_local"]),
                    "chunk_text_original": page_chunk["original_text"],
                    "chunk_text_normalized_for_tts": page_chunk["normalized_text"],
                    "chunk_char_count": len(str(page_chunk["normalized_text"])),
                    "chunk_word_count": count_words(str(page_chunk["original_text"])),
                }
            )

    return document_text_original, chunk_rows


def load_chunk_manifest_rows(document_chunk_dir: Path) -> List[Dict[str, object]]:
    manifest_path = document_chunk_dir / CHUNK_MANIFEST_FILENAME
    if manifest_path.exists():
        rows: List[Dict[str, object]] = []
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    rows = []
    for chunk_index, chunk_path in enumerate(sorted(document_chunk_dir.glob("*.txt")), start=1):
        chunk_text = chunk_path.read_text(encoding="utf-8").strip()
        rows.append(
            {
                "chunk_id": chunk_path.stem,
                "chunk_index": chunk_index,
                "chunk_text_original": chunk_text,
                "chunk_text_normalized_for_tts": chunk_text,
                "chunk_char_count": len(chunk_text),
                "chunk_word_count": count_words(chunk_text),
                "source_page_number": None,
                "source_line_start": None,
                "source_line_end": None,
                "source_char_start": None,
                "source_char_end": None,
            }
        )
    return rows


def create_chunk_metadata(
    item: Dict[str, object],
    tts_model_name: str,
    tts_model_version: Optional[str],
    generation_status: str,
    generation_started_at: Optional[str] = None,
    generation_finished_at: Optional[str] = None,
    generation_runtime_seconds: Optional[float] = None,
    error_message: Optional[str] = None,
    status: Optional[str] = None,
    skip_reason: Optional[str] = None,
) -> Dict[str, object]:
    record = dict(item)
    record.update(
        {
            "synthetic_audio_id": item.get("sample_id"),
            "synthetic_audio_path": item.get("output_wav"),
            "generated_filename": item.get("generated_filename"),
            "generation_status": generation_status,
            "error_message": error_message,
            "tts_model_name": tts_model_name,
            "tts_model_version": tts_model_version,
            "voice_reference_id": item.get("voice_id"),
            "voice_reference_audio_path": item.get("speaker_wav"),
            "voice_reference_source_url": item.get("youtube_url"),
            "generation_started_at": generation_started_at,
            "generation_finished_at": generation_finished_at,
            "generation_runtime_seconds": generation_runtime_seconds,
        }
    )
    if status is not None:
        record["status"] = status
    if skip_reason is not None:
        record["skip_reason"] = skip_reason
    return record


def unique_rows_by_youtube(rows: Iterable[SpreadsheetRow]) -> List[SpreadsheetRow]:
    seen: Dict[str, SpreadsheetRow] = {}
    for row in rows:
        seen.setdefault(row.youtube_url, row)
    return list(seen.values())


def make_voice_id(row: SpreadsheetRow) -> str:
    digest = hashlib.sha1(row.youtube_url.encode("utf-8")).hexdigest()[:8]
    return f"voice_{row.row_index:03d}_{slugify(Path(row.document_name).stem, 48)}_{digest}"


def run_yt_dlp_download(url: str, output_wav: Path, sample_rate: int = 24000) -> None:
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "yt-dlp",
        "-f",
        "bestaudio",
        "-x",
        "--audio-format",
        "wav",
        "--postprocessor-args",
        f"ffmpeg:-ar {sample_rate} -ac 1",
        "-o",
        str(output_wav),
        "--no-warnings",
        "--quiet",
        url,
    ]
    subprocess.run(cmd, check=True)


def is_valid_http_url(value: str) -> bool:
    return value.startswith(("http://", "https://"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def should_log_progress(index: int, total: int, every: int) -> bool:
    if total <= 0:
        return False
    if index in {1, total}:
        return True
    return every > 0 and index % every == 0


def build_synthesis_plan(
    voices: Sequence[VoiceEntry],
    documents: Sequence[DocumentEntry],
    chunks_dir: Path,
    synth_dir: Path,
    selection_mode: str = "all",
    chunks_per_doc: Optional[int] = None,
    random_seed: Optional[int] = None,
    selected_chunk_ids: Optional[Sequence[str]] = None,
    manual_selection: Optional[Dict[str, List[str]]] = None,
    limit_pairs: Optional[int] = None,
) -> List[Dict[str, object]]:
    selected_chunk_ids_set = set(selected_chunk_ids or [])
    rng = random.Random(random_seed)

    def choose_chunks(document_id: str, chunk_paths: List[Path]) -> List[Path]:
        if selection_mode == "all":
            chosen = chunk_paths
        elif selection_mode == "manual":
            allowed_ids = set(manual_selection.get(document_id, []) if manual_selection else [])
            if selected_chunk_ids_set:
                allowed_ids |= selected_chunk_ids_set
            chosen = [path for path in chunk_paths if path.stem in allowed_ids]
        elif selection_mode == "random":
            if chunks_per_doc is None or chunks_per_doc <= 0:
                raise ValueError("random selection_mode requires --chunks-per-doc > 0")
            sample_size = min(chunks_per_doc, len(chunk_paths))
            chosen = sorted(rng.sample(chunk_paths, sample_size), key=lambda p: p.name)
        else:
            raise ValueError(f"Unsupported selection_mode: {selection_mode}")

        if chunks_per_doc is not None and chunks_per_doc > 0 and selection_mode in {"all", "manual"}:
            chosen = chosen[:chunks_per_doc]
        return chosen

    plan: List[Dict[str, object]] = []
    for voice in voices:
        for document in documents:
            document_chunk_dir = chunks_dir / document.document_id
            chunk_rows = load_chunk_manifest_rows(document_chunk_dir)
            chunk_rows_by_id = {str(row["chunk_id"]): row for row in chunk_rows}
            chunk_paths = sorted(
                [path for path in document_chunk_dir.glob("*.txt") if path.stem in chunk_rows_by_id]
            )
            for chunk_index, chunk_path in enumerate(
                choose_chunks(document.document_id, chunk_paths), start=1
            ):
                chunk_row = chunk_rows_by_id[chunk_path.stem]
                chunk_text = str(chunk_row.get("chunk_text_normalized_for_tts", "")).strip()
                generated_filename = build_generated_filename(
                    voice_reference_id=voice.voice_id,
                    source_document_id=document.document_id,
                    chunk_id=chunk_path.stem,
                    source_page_number=chunk_row.get("source_page_number"),
                )
                output_wav = (
                    synth_dir
                    / voice.voice_id
                    / document.document_id
                    / generated_filename
                )
                plan.append(
                    {
                        "sample_id": f"{voice.voice_id}__{document.document_id}__{chunk_path.stem}",
                        "voice_id": voice.voice_id,
                        "youtube_url": voice.youtube_url,
                        "speaker_wav": voice.speaker_wav,
                        "document_id": document.document_id,
                        "document_name": document.document_name,
                        "source_document_id": chunk_row.get("source_document_id", document.document_id),
                        "source_document_name": chunk_row.get("source_document_name", document.document_name),
                        "source_document_path": chunk_row.get("source_document_path", document.pdf_path),
                        "source_document_type": chunk_row.get("source_document_type", infer_source_document_type(document.pdf_path)),
                        "chunk_id": chunk_path.stem,
                        "chunk_index": chunk_index,
                        "text_path": str(chunk_path.resolve()),
                        "chunk_text_original": chunk_row.get("chunk_text_original", chunk_text),
                        "chunk_text_normalized_for_tts": chunk_row.get("chunk_text_normalized_for_tts", chunk_text),
                        "chunk_char_count": chunk_row.get("chunk_char_count", len(chunk_text)),
                        "chunk_word_count": chunk_row.get("chunk_word_count", count_words(chunk_text)),
                        "source_page_number": chunk_row.get("source_page_number"),
                        "source_line_start": chunk_row.get("source_line_start"),
                        "source_line_end": chunk_row.get("source_line_end"),
                        "source_char_start": chunk_row.get("source_char_start"),
                        "source_char_end": chunk_row.get("source_char_end"),
                        "selection_mode": selection_mode,
                        "generated_filename": generated_filename,
                        "output_wav": str(output_wav.resolve()),
                        "metadata_jsonl_path": str((output_wav.parent / "metadata.jsonl").resolve()),
                    }
                )
                if limit_pairs is not None and len(plan) >= limit_pairs:
                    return plan
    return plan


def prepare_dataset(args: argparse.Namespace) -> Dict[str, object]:
    excel_path = Path(args.excel_path).resolve()
    workspace_dir = Path(args.workspace_dir).resolve()
    pdf_dir = Path(args.pdf_dir).resolve() if args.pdf_dir else None
    pdf_zip = Path(args.pdf_zip).resolve() if args.pdf_zip else None

    rows = load_spreadsheet_rows(excel_path)
    if args.limit_rows is not None:
        rows = rows[: args.limit_rows]
    logger.info("Loaded %s spreadsheet rows from %s", len(rows), excel_path)

    pdf_lookup = build_pdf_lookup(pdf_dir, pdf_zip)
    if not pdf_lookup:
        raise FileNotFoundError("No PDFs were found in --pdf-dir/--pdf-zip")
    logger.info("Indexed %s PDFs from provided sources", len(pdf_lookup))

    extracted_pdfs_dir = workspace_dir / "pdfs"
    texts_dir = workspace_dir / "texts"
    chunks_dir = workspace_dir / "chunks"
    manifests_dir = workspace_dir / "manifests"
    voices_dir = workspace_dir / "voice_refs"

    documents: List[DocumentEntry] = []
    missing_pdfs: List[Dict[str, object]] = []

    for row_index, row in enumerate(rows, start=1):
        source = pdf_lookup.get(row.document_name)
        if source is None:
            missing_pdfs.append(asdict(row))
            logger.warning(
                "[%s/%s] Missing PDF for spreadsheet row %s -> %s",
                row_index,
                len(rows),
                row.row_index,
                row.document_name,
            )
            continue

        if should_log_progress(row_index, len(rows), args.log_every):
            logger.info(
                "[%s/%s] Preparing document %s", row_index, len(rows), row.document_name
            )

        materialized_pdf = materialize_pdf(source, extracted_pdfs_dir)
        pages_payload = extract_pdf_pages(materialized_pdf)
        raw_text, chunk_records = build_chunk_records_for_document(
            document_id=build_source_document_id(row.document_name, str(materialized_pdf)),
            document_name=row.document_name,
            source_document_path=materialized_pdf,
            pages_payload=pages_payload,
            max_chars=args.chunk_max_chars,
            min_chars=args.chunk_min_chars,
        )
        if not raw_text:
            logger.warning("No extractable text for %s", row.document_name)
            continue

        document_id = build_source_document_id(row.document_name, str(materialized_pdf))
        text_path = texts_dir / f"{document_id}.txt"
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text(raw_text, encoding="utf-8")

        chunk_dir = chunks_dir / document_id
        chunk_dir.mkdir(parents=True, exist_ok=True)
        for chunk_record in chunk_records:
            chunk_path = chunk_dir / f"{chunk_record['chunk_id']}.txt"
            chunk_path.write_text(
                str(chunk_record["chunk_text_normalized_for_tts"]),
                encoding="utf-8",
            )
        write_jsonl(chunk_dir / CHUNK_MANIFEST_FILENAME, chunk_records)

        logger.info(
            "Prepared %s -> chars=%s chunks=%s",
            row.document_name,
            len(raw_text),
            len(chunk_records),
        )

        documents.append(
            DocumentEntry(
                document_id=document_id,
                document_name=row.document_name,
                pdf_path=str(materialized_pdf.resolve()),
                text_path=str(text_path.resolve()),
                chunk_count=len(chunk_records),
                char_count=len(raw_text),
                row_index=row.row_index,
            )
        )

    unique_voice_rows = unique_rows_by_youtube(rows)
    if args.limit_voices is not None:
        unique_voice_rows = unique_voice_rows[: args.limit_voices]
    logger.info("Selected %s unique voice candidates", len(unique_voice_rows))

    voices: List[VoiceEntry] = []
    failed_voices: List[Dict[str, object]] = []
    for voice_index, row in enumerate(unique_voice_rows, start=1):
        voice_id = make_voice_id(row)
        speaker_wav = voices_dir / f"{voice_id}.wav"
        if should_log_progress(voice_index, len(unique_voice_rows), args.log_every):
            logger.info(
                "[%s/%s] Preparing voice candidate %s from %s",
                voice_index,
                len(unique_voice_rows),
                voice_id,
                row.youtube_url,
            )

        if not is_valid_http_url(row.youtube_url):
            failed_voices.append(
                {
                    "voice_id": voice_id,
                    "youtube_url": row.youtube_url,
                    "source_document_name": row.document_name,
                    "row_index": row.row_index,
                    "error": "invalid_youtube_url",
                }
            )
            logger.warning(
                "Skipping voice candidate %s because youtube_url is invalid: %s",
                voice_id,
                row.youtube_url,
            )
            continue

        if not args.skip_voice_download and not speaker_wav.exists():
            logger.info("Downloading speaker reference to %s", speaker_wav)
            try:
                run_yt_dlp_download(
                    row.youtube_url, speaker_wav, sample_rate=args.speaker_sample_rate
                )
            except subprocess.CalledProcessError as exc:
                failed_voices.append(
                    {
                        "voice_id": voice_id,
                        "youtube_url": row.youtube_url,
                        "source_document_name": row.document_name,
                        "row_index": row.row_index,
                        "error": str(exc),
                    }
                )
                logger.warning(
                    "Skipping voice candidate %s because reference download failed: %s",
                    voice_id,
                    exc,
                )
                continue

        if not speaker_wav.exists():
            failed_voices.append(
                {
                    "voice_id": voice_id,
                    "youtube_url": row.youtube_url,
                    "source_document_name": row.document_name,
                    "row_index": row.row_index,
                    "error": "missing_speaker_wav",
                }
            )
            logger.warning(
                "Skipping voice candidate %s because no speaker wav is available at %s",
                voice_id,
                speaker_wav,
            )
            continue

        voices.append(
            VoiceEntry(
                voice_id=voice_id,
                youtube_url=row.youtube_url,
                speaker_wav=str(speaker_wav.resolve()),
                source_document_name=row.document_name,
                row_index=row.row_index,
            )
        )

    if args.limit_docs is not None:
        documents = documents[: args.limit_docs]
        logger.info("Trimmed documents to first %s entries", len(documents))

    manual_selection: Dict[str, List[str]] = {}
    if args.manual_selection_file:
        manual_payload = json.loads(
            Path(args.manual_selection_file).read_text(encoding="utf-8")
        )
        if isinstance(manual_payload, dict):
            manual_selection = {
                str(doc_id): [str(chunk_id) for chunk_id in chunk_ids]
                for doc_id, chunk_ids in manual_payload.items()
                if isinstance(chunk_ids, list)
            }

    plan = build_synthesis_plan(
        voices=voices,
        documents=documents,
        chunks_dir=chunks_dir,
        synth_dir=workspace_dir / "synthetic_audio",
        selection_mode=args.selection_mode,
        chunks_per_doc=args.chunks_per_doc,
        random_seed=args.random_seed,
        selected_chunk_ids=args.selected_chunk_ids,
        manual_selection=manual_selection,
        limit_pairs=args.limit_pairs,
    )
    logger.info(
        "Built synthesis plan with %s voice(s), %s document(s), %s output chunk pair(s)",
        len(voices),
        len(documents),
        len(plan),
    )

    write_json(manifests_dir / "spreadsheet_rows.json", [asdict(row) for row in rows])
    write_json(manifests_dir / "documents.json", [asdict(item) for item in documents])
    write_json(manifests_dir / "voices.json", [asdict(item) for item in voices])
    write_json(manifests_dir / "failed_voices.json", failed_voices)
    write_json(manifests_dir / "synthesis_plan.json", plan)
    write_jsonl(manifests_dir / "synthesis_plan.jsonl", plan)
    write_csv(manifests_dir / "synthesis_plan.csv", plan)
    write_json(manifests_dir / "missing_pdfs.json", missing_pdfs)

    summary = {
        "workspace_dir": str(workspace_dir),
        "rows_loaded": len(rows),
        "documents_prepared": len(documents),
        "voices_prepared": len(voices),
        "voices_failed": len(failed_voices),
        "planned_outputs": len(plan),
        "selection_mode": args.selection_mode,
        "chunks_per_doc": args.chunks_per_doc,
        "missing_pdfs": len(missing_pdfs),
    }
    write_json(manifests_dir / "prepare_summary.json", summary)
    return summary


def _load_xtts_model_class():
    try:
        from .xtts_v2 import XTTSv2Model  # type: ignore
    except Exception:
        from xtts_v2 import XTTSv2Model  # type: ignore
    return XTTSv2Model


def synthesize_from_plan(args: argparse.Namespace) -> Dict[str, object]:
    workspace_dir = Path(args.workspace_dir).resolve()
    plan_path = workspace_dir / "manifests" / "synthesis_plan.json"
    if not plan_path.exists():
        raise FileNotFoundError(
            f"Plan not found: {plan_path}. Run prepare stage first."
        )

    if args.agree_license:
        os.environ["COQUI_TOS_AGREED"] = "1"
    elif os.environ.get("COQUI_TOS_AGREED") != "1":
        raise RuntimeError("XTTS-v2 requires COQUI_TOS_AGREED=1 or --agree-license")

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if args.limit_pairs is not None:
        plan = plan[: args.limit_pairs]
    if not plan:
        raise RuntimeError("Synthesis plan is empty")
    logger.info("Loaded %s synthesis items from %s", len(plan), plan_path)

    manifests_dir = workspace_dir / "manifests"
    previous_results_path = manifests_dir / "synthesis_results.json"
    previous_results: Dict[str, Dict[str, object]] = {}
    if previous_results_path.exists():
        previous_rows = json.loads(previous_results_path.read_text(encoding="utf-8"))
        previous_results = {
            str(row.get("sample_id")): row
            for row in previous_rows
            if isinstance(row, dict) and row.get("sample_id")
        }

    XTTSv2Model = _load_xtts_model_class()
    model = XTTSv2Model(model_name=args.model, device=args.device)
    tts_model_version = get_tts_model_version()
    logger.info("Initializing XTTS model for synthesis")
    model.load_model()

    current_results: List[Dict[str, object]] = []
    try:
        for item_index, item in enumerate(plan, start=1):
            text_path = Path(item["text_path"])
            output_wav = Path(item["output_wav"])
            speaker_wav = item["speaker_wav"]
            started_at = utc_now_iso()
            started_clock = dt.datetime.now(dt.timezone.utc)

            if should_log_progress(item_index, len(plan), args.log_every):
                logger.info(
                    "[%s/%s] Synthesizing voice=%s document=%s chunk=%s",
                    item_index,
                    len(plan),
                    item.get("voice_id"),
                    item.get("document_id"),
                    item.get("chunk_id"),
                )

            if args.skip_existing and output_wav.exists():
                previous = previous_results.get(str(item.get("sample_id")))
                record = dict(previous) if previous else create_chunk_metadata(
                    item=item,
                    tts_model_name=args.model,
                    tts_model_version=tts_model_version,
                    generation_status="skipped",
                    generation_started_at=started_at,
                    generation_finished_at=utc_now_iso(),
                    generation_runtime_seconds=0.0,
                    error_message=None,
                    status="skipped",
                    skip_reason="existing_output",
                )
                current_results.append(record)
                logger.info("Skipped existing output: %s", output_wav)
                continue

            previous = previous_results.get(str(item.get("sample_id")))
            if (
                args.resume_from_metadata
                and previous
                and previous.get("status") == "ok"
                and output_wav.exists()
            ):
                record = dict(previous)
                record["skip_reason"] = "resume_ok_from_metadata"
                current_results.append(record)
                continue

            text = str(item.get("chunk_text_normalized_for_tts") or "").strip()
            if not text and text_path.exists():
                text = text_path.read_text(encoding="utf-8").strip()
            if not text:
                finished_at = utc_now_iso()
                record = create_chunk_metadata(
                    item=item,
                    tts_model_name=args.model,
                    tts_model_version=tts_model_version,
                    generation_status="failed",
                    generation_started_at=started_at,
                    generation_finished_at=finished_at,
                    generation_runtime_seconds=0.0,
                    error_message="empty_text",
                    status="error",
                )
                current_results.append(record)
                logger.warning("Skipped empty text file: %s", text_path)
                continue

            try:
                model.tts_to_file(
                    text=text,
                    file_path=str(output_wav),
                    speaker_wav=speaker_wav,
                    language=args.language,
                )
                finished_clock = dt.datetime.now(dt.timezone.utc)
                record = create_chunk_metadata(
                    item=item,
                    tts_model_name=args.model,
                    tts_model_version=tts_model_version,
                    generation_status="generated",
                    generation_started_at=started_at,
                    generation_finished_at=finished_clock.replace(microsecond=0).isoformat(),
                    generation_runtime_seconds=round(
                        (finished_clock - started_clock).total_seconds(), 6
                    ),
                    error_message=None,
                    status="ok",
                )
                if should_log_progress(item_index, len(plan), args.log_every):
                    logger.info("Wrote synthetic audio: %s", output_wav)
            except Exception as exc:
                finished_clock = dt.datetime.now(dt.timezone.utc)
                record = create_chunk_metadata(
                    item=item,
                    tts_model_name=args.model,
                    tts_model_version=tts_model_version,
                    generation_status="failed",
                    generation_started_at=started_at,
                    generation_finished_at=finished_clock.replace(microsecond=0).isoformat(),
                    generation_runtime_seconds=round(
                        (finished_clock - started_clock).total_seconds(), 6
                    ),
                    error_message=str(exc),
                    status="error",
                )
                logger.error(
                    "Failed synthesis for voice=%s document=%s chunk=%s: %s",
                    item.get("voice_id"),
                    item.get("document_id"),
                    item.get("chunk_id"),
                    exc,
                )
            current_results.append(record)
    finally:
        model.cleanup()

    merged_results_map = dict(previous_results)
    for record in current_results:
        merged_results_map[str(record.get("sample_id"))] = record
    results = list(merged_results_map.values())

    metadata_by_path: Dict[str, List[Dict[str, object]]] = {}
    for record in results:
        metadata_path = record.get("metadata_jsonl_path")
        if metadata_path:
            metadata_by_path.setdefault(str(metadata_path), []).append(record)
    for metadata_path, rows in metadata_by_path.items():
        path = Path(metadata_path)
        if path.exists():
            path.unlink()
        for row in sorted(rows, key=lambda item: str(item.get("sample_id", ""))):
            write_jsonl_record(path, row)

    summary = {
        "workspace_dir": str(workspace_dir),
        "planned": len(plan),
        "ok": sum(1 for item in current_results if item["status"] == "ok"),
        "skipped": sum(1 for item in current_results if item["status"] == "skipped"),
        "errors": sum(1 for item in current_results if item["status"] == "error"),
    }
    write_json(manifests_dir / "synthesis_results.json", results)
    write_jsonl(manifests_dir / "synthesis_results.jsonl", results)
    write_csv(manifests_dir / "synthesis_results.csv", results)
    write_json(manifests_dir / "synthesis_summary.json", summary)
    return summary


def validate_workspace(args: argparse.Namespace) -> Dict[str, object]:
    workspace_dir = Path(args.workspace_dir).resolve()
    manifests_dir = workspace_dir / "manifests"
    plan_path = manifests_dir / "synthesis_plan.json"
    results_path = manifests_dir / "synthesis_results.json"
    plan: List[Dict[str, object]] = json.loads(plan_path.read_text(encoding="utf-8"))
    results: List[Dict[str, object]] = []
    if results_path.exists():
        results = json.loads(results_path.read_text(encoding="utf-8"))

    issues: List[str] = []
    for row in plan:
        text_path = Path(str(row.get("text_path", "")))
        chunk_text = text_path.read_text(encoding="utf-8").strip() if text_path.exists() else ""
        if not text_path.exists():
            issues.append(f"missing_text_path:{text_path}")
        if not chunk_text:
            issues.append(f"empty_chunk_text:{row.get('sample_id')}")
        if args.chunk_max_chars and len(chunk_text) > args.chunk_max_chars:
            issues.append(f"chunk_too_long:{row.get('sample_id')}:{len(chunk_text)}")

    plan_ids = {str(row.get("sample_id")) for row in plan}
    result_ids = {str(row.get("sample_id")) for row in results if row.get("sample_id")}
    missing_in_results = sorted(plan_ids - result_ids)
    extra_in_results = sorted(result_ids - plan_ids)
    if results and missing_in_results:
        issues.append(f"metadata_parity_missing_results:{len(missing_in_results)}")
    if extra_in_results:
        issues.append(f"metadata_parity_extra_results:{len(extra_in_results)}")

    for row in results:
        output = Path(str(row.get("output_wav", "")))
        status = row.get("status")
        if status == "ok" and not output.exists():
            issues.append(f"missing_output_for_ok:{row.get('sample_id')}")

    summary = {
        "workspace_dir": str(workspace_dir),
        "plan_rows": len(plan),
        "result_rows": len(results),
        "issues": len(issues),
        "valid": len(issues) == 0,
    }
    write_json(manifests_dir / "validation_summary.json", summary)
    write_json(manifests_dir / "validation_issues.json", issues)
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare and synthesize acta-based XTTS datasets"
    )
    parser.add_argument(
        "--stage",
        choices=["prepare", "synthesize", "validate", "all"],
        default="prepare",
        help="Stage to run. Default: prepare",
    )
    parser.add_argument(
        "--excel-path", required=True, help="Excel file with PDF + YouTube columns"
    )
    parser.add_argument(
        "--pdf-dir", default="", help="Directory containing extracted PDFs"
    )
    parser.add_argument("--pdf-zip", default="", help="ZIP containing PDFs")
    parser.add_argument(
        "--workspace-dir",
        default="outputs/actas_xtts",
        help="Workspace for extracted texts, voice refs, manifests, and outputs",
    )
    parser.add_argument(
        "--limit-rows", type=int, default=None, help="Optional spreadsheet row limit"
    )
    parser.add_argument(
        "--limit-voices", type=int, default=None, help="Optional unique voice limit"
    )
    parser.add_argument(
        "--limit-docs",
        type=int,
        default=None,
        help="Optional document limit after extraction",
    )
    parser.add_argument(
        "--limit-pairs", type=int, default=None, help="Optional synthesis pair limit"
    )
    parser.add_argument(
        "--selection-mode",
        choices=["all", "random", "manual"],
        default="random",
        help="Chunk selection mode for synthesis plan (default: random)",
    )
    parser.add_argument(
        "--chunks-per-doc",
        type=int,
        default=10,
        help="Limit selected chunks per document (default: 10)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=7,
        help="Seed for reproducible random chunk selection",
    )
    parser.add_argument(
        "--selected-chunk-ids",
        nargs="*",
        default=[],
        help="Manual chunk IDs to include (e.g., chunk_0001 chunk_0003)",
    )
    parser.add_argument(
        "--manual-selection-file",
        default="",
        help="Optional JSON file mapping document_id -> [chunk_id,...]",
    )
    parser.add_argument(
        "--chunk-max-chars",
        type=int,
        default=180,
        help="Maximum chars per chunk (default tuned for XTTS Spanish)",
    )
    parser.add_argument(
        "--chunk-min-chars",
        type=int,
        default=80,
        help="Minimum chars before merge (default tuned for XTTS Spanish)",
    )
    parser.add_argument(
        "--speaker-sample-rate",
        type=int,
        default=24000,
        help="Sample rate for downloaded speaker references",
    )
    parser.add_argument(
        "--skip-voice-download",
        action="store_true",
        help="Prepare manifests/texts without downloading YouTube audio references",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip already generated synthetic WAV files during synthesis",
    )
    parser.add_argument(
        "--resume-from-metadata",
        action="store_true",
        help="Skip rows already marked ok in synthesis_results.json when output exists",
    )
    parser.add_argument(
        "--model",
        default="tts_models/multilingual/multi-dataset/xtts_v2",
        help="XTTS model id/path",
    )
    parser.add_argument(
        "--device", default=None, help="Runtime device override (cpu/cuda)"
    )
    parser.add_argument("--language", default="es", help="Target language code")
    parser.add_argument(
        "--agree-license",
        action="store_true",
        help="Set COQUI_TOS_AGREED=1 for XTTS synthesis",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Log progress every N items during prepare/synthesize (default: 1)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    summary: Dict[str, object] = {}
    if args.stage in {"prepare", "all"}:
        summary["prepare"] = prepare_dataset(args)
        logger.info("Prepare summary: %s", summary["prepare"])

    if args.stage in {"synthesize", "all"}:
        summary["synthesize"] = synthesize_from_plan(args)
        logger.info("Synthesis summary: %s", summary["synthesize"])

    if args.stage in {"validate", "all"}:
        summary["validate"] = validate_workspace(args)
        logger.info("Validation summary: %s", summary["validate"])

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
