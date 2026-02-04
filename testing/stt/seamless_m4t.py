from pathlib import Path
import argparse
import logging
import json
import numpy as np
import torch
import torchaudio
from typing import List, Optional
from tqdm import tqdm
from transformers import AutoProcessor, SeamlessM4Tv2Model
import scipy.io.wavfile as wavfile

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("seamless_m4t")


def find_audio_files(audio_dir: Path, max_files: int) -> List[Path]:
    if not audio_dir.exists() or not audio_dir.is_dir():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
    files = sorted(
        [
            p
            for p in audio_dir.glob("*.*")
            if p.suffix.lower() in {".wav", ".m4a", ".mp3", ".flac"}
        ]
    )
    return files[:max_files]


def load_and_resample(path: Path, target_sr: int = 16000):
    waveform, sr = torchaudio.load(str(path))
    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sr, new_freq=target_sr
        )
    # Model expects shape (channels, samples) or (samples,) depending on processor; we'll pass waveform (1, N)
    return waveform, target_sr


def float32_to_int16(audio: np.ndarray) -> np.ndarray:
    # audio is float32 in [-1,1] or similar -> convert to int16
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767).astype(np.int16)


def main():
    parser = argparse.ArgumentParser(
        description="Batch generate audio with SeamlessM4T v2"
    )
    parser.add_argument(
        "--max", type=int, default=10, help="Max audio files to process"
    )
    parser.add_argument(
        "--audio-dir",
        type=str,
        default=None,
        help="Source audio dir (defaults to lyrics-eater/audio)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output dir (defaults to testing/transcriptions/seamless_m4t)",
    )
    parser.add_argument(
        "--tgt-lang",
        type=str,
        required=True,
        help="Target language code (e.g. 'rus', 'spa', 'eng')",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use: 'cuda' or 'cpu' (default auto)",
    )
    args = parser.parse_args()
    this_dir = Path(__file__).resolve().parent
    repo_root = this_dir.parent
    default_audio_dir = repo_root / "lyrics-eater" / "audio"
    default_out_dir = this_dir / "transcriptions" / "seamless_m4t"
    audio_dir = Path(args.audio_dir) if args.audio_dir else default_audio_dir
    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    files = find_audio_files(audio_dir, args.max)
    if not files:
        logger.error("No audio files found in %s", audio_dir)
        return
    logger.info("Loading processor and model (this may take a while)...")
    processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
    model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large").to(
        device
    )
    model.eval()
    saved = []
    for path in tqdm(files, desc="Processing", unit="file"):
        try:
            waveform, sr = load_and_resample(path, target_sr=16000)
            # Prepare inputs for model (use `audio=` keyword; `audios` is deprecated)
            inputs = processor(audio=waveform, return_tensors="pt")
            # move tensor inputs to device
            try:
                inputs = {k: v.to(device) for k, v in inputs.items()}
            except Exception:
                # inputs may not be a dict of tensors; fall back to .to(device)
                try:
                    inputs = inputs.to(device)
                except Exception:
                    pass
            with torch.no_grad():
                generated = model.generate(**inputs, tgt_lang=args.tgt_lang)
            # generated is a list/tensor; pick first and move to cpu numpy
            audio_np = generated[0].cpu().numpy().squeeze()
            # Normalize and save as int16
            if audio_np.dtype != np.float32:
                audio_np = audio_np.astype(np.float32)
            # Ensure in [-1,1]
            max_val = np.max(np.abs(audio_np)) if np.max(np.abs(audio_np)) > 0 else 1.0
            audio_np = audio_np / max_val
            int16 = float32_to_int16(audio_np)
            out_wav = out_dir / f"{path.stem}_to_{args.tgt_lang}.wav"
            wavfile.write(str(out_wav), rate=model.config.sampling_rate, data=int16)
            meta = {
                "source_file": str(path),
                "out_wav": str(out_wav),
                "tgt_lang": args.tgt_lang,
                "sample_rate": int(model.config.sampling_rate),
                "notes": "generated with facebook/seamless-m4t-v2-large",
            }
            json_path = out_wav.with_suffix(".json")
            json_path.write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            saved.append(str(out_wav))
        except Exception as e:
            logger.exception("Failed on %s: %s", path, e)
    logger.info("Generated %d outputs (saved in %s)", len(saved), out_dir)


if __name__ == "__main__":
    main()
