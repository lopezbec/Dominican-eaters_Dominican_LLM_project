#!/usr/bin/env python3
"""Run poem-based TTS benchmark across XTTS, F5-Spanish, and Kokoro."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

from xtts_v2 import XTTSv2Model

logger = logging.getLogger(__name__)


def load_prompts(path: Path, limit: Optional[int]) -> List[Dict[str, str]]:
    prompts = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            prompts.append(json.loads(line))
    if limit is not None:
        prompts = prompts[:limit]
    return prompts


def split_csv_paths(raw: str) -> List[str]:
    return [p.strip() for p in raw.split(",") if p.strip()]


def run_xtts(
    prompts: List[Dict[str, str]],
    output_dir: Path,
    language: str,
    speaker_wav: Union[str, List[str]],
    model_name: str,
    device: Optional[str],
    condition_name: str,
) -> List[Dict[str, object]]:
    model = XTTSv2Model(model_name=model_name, device=device)
    model.load_model()

    condition_dir = output_dir / condition_name
    condition_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    try:
        for prompt in prompts:
            wav_path = condition_dir / f"{prompt['id']}.wav"
            status = "ok"
            error = None
            try:
                model.tts_to_file(
                    text=prompt["text"],
                    file_path=str(wav_path),
                    speaker_wav=speaker_wav,
                    language=language,
                )
            except Exception as e:
                status = "error"
                error = str(e)

            rows.append(
                {
                    "prompt_id": prompt["id"],
                    "source_file": prompt.get("source_file"),
                    "model": condition_name,
                    "status": status,
                    "error": error,
                    "text": prompt["text"],
                    "wav_path": str(wav_path),
                }
            )
    finally:
        model.cleanup()

    return rows


def run_f5(
    prompts: List[Dict[str, str]],
    output_dir: Path,
    ref_audio: str,
    ref_text: str,
    model_name: str,
    device: Optional[str],
    speed: float,
    nfe_step: int,
) -> List[Dict[str, object]]:
    from f5_spanish_tts import F5SpanishTTSModel

    model = F5SpanishTTSModel(model_name=model_name, device=device)
    model.load_model()

    condition_name = "f5_spanish"
    condition_dir = output_dir / condition_name
    condition_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for prompt in prompts:
        wav_path = condition_dir / f"{prompt['id']}.wav"
        status = "ok"
        error = None
        try:
            model.tts_to_file(
                text=prompt["text"],
                file_path=str(wav_path),
                ref_audio=ref_audio,
                ref_text=ref_text,
                speed=speed,
                nfe_step=nfe_step,
            )
        except Exception as e:
            status = "error"
            error = str(e)

        rows.append(
            {
                "prompt_id": prompt["id"],
                "source_file": prompt.get("source_file"),
                "model": condition_name,
                "status": status,
                "error": error,
                "text": prompt["text"],
                "wav_path": str(wav_path),
            }
        )

    return rows


def run_kokoro(
    prompts: List[Dict[str, str]],
    output_dir: Path,
    language: str,
    voice: str,
    speed: float,
) -> List[Dict[str, object]]:
    from kokoro_tts import KokoroTTSModel, _resolve_lang_code

    lang_code = _resolve_lang_code(language)
    model = KokoroTTSModel(lang_code=lang_code, default_voice=voice)
    model.load_model()

    condition_name = "kokoro"
    condition_dir = output_dir / condition_name
    condition_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for prompt in prompts:
        wav_path = condition_dir / f"{prompt['id']}.wav"
        status = "ok"
        error = None
        try:
            model.tts_to_file(
                text=prompt["text"],
                file_path=str(wav_path),
                voice=voice,
                speed=speed,
            )
        except Exception as e:
            status = "error"
            error = str(e)

        rows.append(
            {
                "prompt_id": prompt["id"],
                "source_file": prompt.get("source_file"),
                "model": condition_name,
                "status": status,
                "error": error,
                "text": prompt["text"],
                "wav_path": str(wav_path),
            }
        )

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run poem-driven benchmark for XTTS/F5/Kokoro"
    )
    parser.add_argument(
        "--prompts-jsonl",
        required=True,
        help="Prompt file generated by poem_prompt_builder.py",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/poems_benchmark",
        help="Directory for generated outputs and manifest",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional prompt limit for quick runs",
    )

    # XTTS
    parser.add_argument(
        "--xtts-speaker-wav",
        default="",
        help="Single reference WAV for XTTS single-reference condition",
    )
    parser.add_argument(
        "--xtts-speaker-wavs",
        default="",
        help="Comma-separated WAVs for XTTS multi-reference condition",
    )
    parser.add_argument(
        "--xtts-model",
        default="tts_models/multilingual/multi-dataset/xtts_v2",
        help="XTTS model id",
    )
    parser.add_argument(
        "--language",
        default="es",
        help="Target language code for XTTS/Kokoro (default: es)",
    )
    parser.add_argument(
        "--agree-license",
        action="store_true",
        help="Agree Coqui CPML for XTTS by setting COQUI_TOS_AGREED=1",
    )

    # F5
    parser.add_argument(
        "--run-f5", action="store_true", help="Run F5-Spanish condition"
    )
    parser.add_argument("--f5-ref-audio", default="", help="Reference audio for F5")
    parser.add_argument("--f5-ref-text", default="", help="Reference transcript for F5")
    parser.add_argument("--f5-model", default="F5TTS_v1_Base", help="F5 model config")
    parser.add_argument("--f5-speed", type=float, default=1.0, help="F5 speed")
    parser.add_argument("--f5-nfe-step", type=int, default=32, help="F5 nfe step")

    # Kokoro
    parser.add_argument(
        "--run-kokoro", action="store_true", help="Run Kokoro condition"
    )
    parser.add_argument("--kokoro-voice", default="ef_dora", help="Kokoro voice id")
    parser.add_argument("--kokoro-speed", type=float, default=1.0, help="Kokoro speed")

    # Shared runtime
    parser.add_argument(
        "--device",
        default=None,
        help="Device override for XTTS/F5 (cpu/cuda/...)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    prompts_path = Path(args.prompts_jsonl).resolve()
    if not prompts_path.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_path}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(prompts_path, args.limit)
    if not prompts:
        raise RuntimeError("No prompts loaded")

    if args.agree_license:
        os.environ["COQUI_TOS_AGREED"] = "1"

    all_rows: List[Dict[str, object]] = []

    # XTTS single reference
    if args.xtts_speaker_wav:
        all_rows.extend(
            run_xtts(
                prompts=prompts,
                output_dir=output_dir,
                language=args.language,
                speaker_wav=args.xtts_speaker_wav,
                model_name=args.xtts_model,
                device=args.device,
                condition_name="xtts_single_ref",
            )
        )

    # XTTS multi reference (comma-separated) -> pass list directly when possible
    if args.xtts_speaker_wavs:
        refs = split_csv_paths(args.xtts_speaker_wavs)
        if not refs:
            raise ValueError("--xtts-speaker-wavs provided but no valid paths parsed")
        all_rows.extend(
            run_xtts(
                prompts=prompts,
                output_dir=output_dir,
                language=args.language,
                speaker_wav=refs,
                model_name=args.xtts_model,
                device=args.device,
                condition_name="xtts_multi_ref",
            )
        )

    if args.run_f5:
        if not args.f5_ref_audio:
            raise ValueError("--run-f5 requires --f5-ref-audio")
        all_rows.extend(
            run_f5(
                prompts=prompts,
                output_dir=output_dir,
                ref_audio=args.f5_ref_audio,
                ref_text=args.f5_ref_text,
                model_name=args.f5_model,
                device=args.device,
                speed=args.f5_speed,
                nfe_step=args.f5_nfe_step,
            )
        )

    if args.run_kokoro:
        all_rows.extend(
            run_kokoro(
                prompts=prompts,
                output_dir=output_dir,
                language=args.language,
                voice=args.kokoro_voice,
                speed=args.kokoro_speed,
            )
        )

    if not all_rows:
        raise RuntimeError(
            "No model conditions executed. Provide XTTS refs and/or --run-f5/--run-kokoro"
        )

    manifest = {
        "prompts_jsonl": str(prompts_path),
        "output_dir": str(output_dir),
        "language": args.language,
        "rows": all_rows,
    }
    manifest_path = output_dir / "run_manifest_poems.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Wrote manifest: %s", manifest_path)


if __name__ == "__main__":
    main()
