"""Example wrapper for Qwen3-TTS (CPU-friendly defaults)

This script demonstrates how to call Qwen3-TTS models in a reproducible way.
It defaults to CPU and writes WAV outputs to `testing/tts/outputs/`.
"""

from pathlib import Path
import argparse
import logging
import torch
import soundfile as sf

from typing import List

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("qwen3_tts")


def main():
    parser = argparse.ArgumentParser(description="Run Qwen3-TTS example (CPU default)")
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    )
    parser.add_argument(
        "--text", type=str, default="Hola, esto es una prueba de síntesis de voz."
    )
    parser.add_argument("--speaker", type=str, default="Ryan")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    this_dir = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir) if args.out_dir else this_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    logger.info("Device: %s", device)

    # Use dtype float32 on CPU for safety
    dtype = torch.float32

    try:
        from qwen_tts import Qwen3TTSModel
    except Exception as e:
        logger.error("qwen-tts package not installed: %s", e)
        logger.info("Install with: pip install qwen-tts")
        return

    logger.info("Loading model: %s", args.model)
    model = Qwen3TTSModel.from_pretrained(
        args.model, device_map={"": device}, dtype=dtype
    )

    # single inference
    wavs, sr = model.generate_custom_voice(
        text=args.text, language="Auto", speaker=args.speaker
    )
    out_path = out_dir / "qwen_example.wav"
    sf.write(str(out_path), wavs[0], sr)
    logger.info("Wrote: %s (sr=%d)", out_path, sr)


if __name__ == "__main__":
    main()
