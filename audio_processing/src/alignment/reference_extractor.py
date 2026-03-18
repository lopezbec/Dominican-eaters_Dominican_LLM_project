import os
import json
import re
from pathlib import Path
from typing import Dict
import logging

try:
    import pandas as pd
except ImportError:
    pd = None

logger = logging.getLogger(__name__)


class ReferenceTextExtractor:
    def __init__(self, config: Dict):
        self.config = config

    def extract_text_from_excel(self, module_name: str) -> Dict:
        if module_name == "lyrics-eater":
            return self._extract_lyrics()
        elif module_name == "poems-eater":
            return self._extract_poems()
        else:
            raise ValueError(f"Unsupported module: {module_name}")

    def _extract_lyrics(self) -> Dict:
        module_config = self.config["modules"]["lyrics-eater"]
        excel_path = module_config["excel_path"]
        lyrics_column = module_config["lyrics_column"]
        url_column = module_config["url_column"]
        reference_texts_dir = module_config["reference_texts_dir"]
        reports_dir = module_config.get("reports_dir", "lyrics-eater/reports")

        os.makedirs(reference_texts_dir, exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)

        if pd is None:
            raise ImportError("pandas and openpyxl required")

        df = pd.read_excel(excel_path)

        mapping = {
            "by_video_id": {},  # video_id -> filename
            "by_seq": {},  # zero-padded downloader index -> filename
        }

        from shared.utils.text_utils import clean_reference_text

        # Keep the exact URL filtering order used by downloader/DataLoader so
        # downloader report indexes map to the same rows.
        filtered_rows = []
        for _, row in df.iterrows():
            url = row.get(url_column, "")
            if isinstance(url, str) and url.startswith("http") and "youtube.com" in url:
                filtered_rows.append(row)

        report_path = os.path.join(reports_dir, "download_report.json")
        extracted = 0
        generated_filenames = set()

        if os.path.exists(report_path):
            try:
                with open(report_path, "r", encoding="utf-8") as rf:
                    report_data = json.load(rf)

                for entry in report_data.get("results", []):
                    if not entry.get("success"):
                        continue

                    idx = entry.get("index")
                    if not isinstance(idx, int):
                        continue

                    if idx < 0 or idx >= len(filtered_rows):
                        logger.warning(
                            "Download index %s out of range for filtered lyrics rows (%s)",
                            idx,
                            len(filtered_rows),
                        )
                        continue

                    row = filtered_rows[idx]
                    lyrics = row.get(lyrics_column, "")
                    if not isinstance(lyrics, str) or not lyrics.strip():
                        logger.warning(
                            "Missing lyrics text for successful download index %s; skipping reference extraction",
                            idx,
                        )
                        continue

                    file_path = entry.get("file_path")
                    if isinstance(file_path, str) and file_path.strip():
                        base_name = os.path.splitext(os.path.basename(file_path))[0]
                    else:
                        base_name = f"lyrics-eater_{idx:03d}"

                    filename = f"{base_name}.txt"
                    output_path = os.path.join(reference_texts_dir, filename)

                    cleaned_lyrics = clean_reference_text(lyrics)
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(cleaned_lyrics)
                    generated_filenames.add(filename)

                    seq_key = f"{idx:03d}"
                    mapping["by_seq"][seq_key] = filename

                    source_url = entry.get("url") or row.get(url_column, "")
                    video_match = re.search(
                        r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", str(source_url)
                    )
                    if video_match:
                        mapping["by_video_id"][video_match.group(1)] = filename

                    extracted += 1

                logger.info(
                    "Extracted lyrics using downloader stems from report: %s",
                    report_path,
                )

                if generated_filenames:
                    removed_stale = 0
                    for txt_path in Path(reference_texts_dir).glob(
                        "lyrics-eater_*.txt"
                    ):
                        if txt_path.name not in generated_filenames:
                            try:
                                txt_path.unlink()
                                removed_stale += 1
                            except Exception as unlink_error:
                                logger.warning(
                                    "Failed to remove stale reference file %s: %s",
                                    txt_path,
                                    unlink_error,
                                )

                    if removed_stale:
                        logger.info(
                            "Removed %s stale lyrics reference files not present in current download report",
                            removed_stale,
                        )
            except Exception as e:
                logger.warning(
                    "Failed to use download report %s for stem-aligned extraction (%s). Falling back to legacy naming.",
                    report_path,
                    e,
                )

        if extracted == 0:
            # Fallback to legacy behavior if report is unavailable or unusable.
            for seq_idx, (_, row) in enumerate(df.iterrows()):
                url = row.get(url_column, "")
                lyrics = row.get(lyrics_column, "")

                if not isinstance(url, str) or not url.startswith("http"):
                    continue

                if not isinstance(lyrics, str) or not lyrics.strip():
                    continue

                # determine stable id: prefer configured id_column, else parse YouTube id from URL, else fallback to seq
                id_col = module_config.get("id_column")
                id_val = None
                if id_col and id_col in row and row.get(id_col):
                    id_val = str(row.get(id_col)).strip()
                else:
                    m = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", str(url))
                    if m:
                        id_val = m.group(1)
                    else:
                        id_val = f"{seq_idx:03d}"

                filename = f"lyrics-eater_{id_val}.txt"
                output_path = os.path.join(reference_texts_dir, filename)

                cleaned_lyrics = clean_reference_text(lyrics)

                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(cleaned_lyrics)

                # populate mapping entries
                mapping["by_video_id"][id_val] = filename
                seq_key = f"{seq_idx:03d}"
                mapping["by_seq"][seq_key] = filename

                extracted += 1

        # write mapping file to reports dir
        map_path = os.path.join(reports_dir, "lyrics_reference_map.json")
        try:
            with open(map_path, "w", encoding="utf-8") as mf:
                json.dump(mapping, mf, ensure_ascii=False, indent=2)
            logger.info(f"Wrote reference mapping to: {map_path}")
        except Exception as e:
            logger.warning(f"Failed to write reference mapping to {map_path}: {e}")

        logger.info(f"Extracted {extracted} reference texts for lyrics-eater")
        return {
            "extracted": extracted,
            "mapping_path": map_path,
            "download_report_used": os.path.exists(report_path),
        }

    def _extract_poems(self) -> Dict:
        """
        Extract poem texts from Excel file.

        For poems, the text would be in a 'texto' or 'transcripcion' column.
        If no poem text is available, extraction cannot proceed.
        """
        module_config = self.config["modules"]["poems-eater"]
        excel_path = module_config["excel_path"]
        url_column = module_config.get("url_column", "URL YouTube")
        reference_texts_dir = module_config["reference_texts_dir"]

        os.makedirs(reference_texts_dir, exist_ok=True)

        if pd is None:
            raise ImportError("pandas and openpyxl required")

        try:
            df = pd.read_excel(excel_path)

            # Check for poem text column (could be 'texto', 'transcripcion', 'poema', etc.)
            text_column = None
            for possible_column in ["texto", "transcripcion", "poema", "contenido"]:
                if possible_column in df.columns:
                    text_column = possible_column
                    break

            if not text_column:
                logger.warning(
                    f"No poem text column found in {excel_path}. "
                    f"Available columns: {df.columns.tolist()}"
                )
                return {
                    "extracted": 0,
                    "note": "No poem text column available - manual extraction required",
                }

            extracted = 0
            for idx, row in df.iterrows():
                url = row.get(url_column, "")
                poem_text = row.get(text_column, "")

                if not isinstance(url, str) or not url.startswith("http"):
                    continue

                if not isinstance(poem_text, str) or not poem_text.strip():
                    continue

                filename = f"poems-eater_{idx:03d}.txt"
                output_path = os.path.join(reference_texts_dir, filename)

                from shared.utils.text_utils import clean_reference_text

                cleaned_text = clean_reference_text(poem_text)

                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(cleaned_text)

                extracted += 1

            logger.info(f"Extracted {extracted} reference texts for poems-eater")
            return {"extracted": extracted}

        except Exception as e:
            logger.error(f"Failed to extract poems: {e}")
            return {"extracted": 0, "note": f"Extraction failed: {str(e)}"}

    def extract_text_from_txt(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
