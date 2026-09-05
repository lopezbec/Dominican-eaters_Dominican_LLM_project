#!/usr/bin/env python3
"""Run causal LM evaluation with logits, logprobs, and perplexity.

This script is designed for reproducible VM-side evaluation of Hugging Face
causal language models such as Qwen/Qwen3-4B-Instruct-2507.

Perplexity is computed with the standard causal LM shift:
    logits[:, t] predict labels[:, t + 1]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_eval_common import safe_exp

try:
    from transformers import BitsAndBytesConfig
except ImportError:  # pragma: no cover - depends on transformers version
    BitsAndBytesConfig = None


LOGGER = logging.getLogger("qwen_probability_eval")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a causal LM on CSV or JSONL text samples and save logits, "
            "logprobs, perplexity, and runtime statistics."
        )
    )
    parser.add_argument("--input_file", required=True, help="Input CSV or JSONL file")
    parser.add_argument(
        "--text_column",
        default="text",
        help="Column containing the input text",
    )
    parser.add_argument(
        "--id_column",
        default="document_id",
        help="Column containing the sample ID",
    )
    parser.add_argument(
        "--model_id",
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Hugging Face model ID",
    )
    parser.add_argument(
        "--output_jsonl",
        required=True,
        help="Where to write the JSONL results",
    )
    parser.add_argument(
        "--artifacts_dir",
        default="",
        help="Optional directory for sidecar tensors and per-token logprob files",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum tokenized sequence length",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="How many final next-token predictions to save",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Execution device",
    )
    parser.add_argument(
        "--load_4bit",
        action="store_true",
        help="Load the model with bitsandbytes 4-bit quantization",
    )
    parser.add_argument(
        "--save_final_probability_vector",
        action="store_true",
        help="Save the final next-token probability vector as a .pt file",
    )
    parser.add_argument(
        "--save_final_logits",
        action="store_true",
        help="Save the final next-token logits as a .pt file",
    )
    parser.add_argument(
        "--save_all_token_probability_vectors",
        action="store_true",
        help="Debug mode: save full probability vectors for each next-token step",
    )
    parser.add_argument(
        "--max_full_vector_tokens",
        type=int,
        default=64,
        help=(
            "Only save all-token probability vectors when the number of predicted "
            "positions is at or below this limit"
        ),
    )
    parser.add_argument(
        "--skip_target_token_logprobs",
        action="store_true",
        help="Skip saving per-token log probabilities for the actual next tokens",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Pass trust_remote_code=True to Transformers loaders",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional maximum number of rows to process",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return parser.parse_args(argv)


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available on this machine.")
    return device_arg


def load_input_rows(
    input_path: Path,
    text_column: str,
    id_column: str,
    limit: int,
) -> List[Dict[str, Any]]:
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(input_path)
    elif suffix == ".jsonl":
        frame = pd.read_json(input_path, lines=True)
    elif suffix == ".json":
        frame = pd.read_json(input_path)
    else:
        raise ValueError("input_file must be a CSV or JSONL file.")

    missing_columns = [
        column for column in (text_column, id_column) if column not in frame.columns
    ]
    if missing_columns:
        raise ValueError(
            "Missing required column(s): " + ", ".join(sorted(missing_columns))
        )

    if limit > 0:
        frame = frame.head(limit)

    rows: List[Dict[str, Any]] = []
    for row_index, row in frame.iterrows():
        raw_id = row[id_column]
        raw_text = row[text_column]
        sample_id = f"row_{row_index}" if pd.isna(raw_id) else str(raw_id)
        text = "" if pd.isna(raw_text) else str(raw_text)
        rows.append({"sample_id": sample_id, "text": text, "row_index": int(row_index)})
    return rows


def build_artifacts_dir(output_jsonl: Path, artifacts_dir_arg: str) -> Path:
    if artifacts_dir_arg:
        return Path(artifacts_dir_arg).resolve()
    return output_jsonl.resolve().parent / f"{output_jsonl.stem}_artifacts"


def get_ram_usage_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def preview_text(text: str, limit: int = 120) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    return normalized[:limit]


def sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_")
    return cleaned[:120] or "sample"


def primary_model_device(model: AutoModelForCausalLM, fallback_device: str) -> torch.device:
    if fallback_device == "cuda":
        return torch.device("cuda")
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device(fallback_device)


def optional_quantization_config(device: str, load_4bit: bool) -> Optional[BitsAndBytesConfig]:
    if not load_4bit:
        return None
    if device != "cuda":
        raise RuntimeError("4-bit loading requires CUDA. CPU 4-bit loading is not supported here.")
    if BitsAndBytesConfig is None:
        raise RuntimeError(
            "This Transformers version does not expose BitsAndBytesConfig. "
            "Upgrade transformers and install bitsandbytes."
        )
    try:
        import bitsandbytes  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "bitsandbytes is not installed. Install it before using --load_4bit."
        ) from exc

    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def preferred_inference_dtype(device: str) -> torch.dtype:
    if device == "cpu":
        return torch.float32
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_model_and_tokenizer(
    model_id: str,
    device: str,
    load_4bit: bool,
    trust_remote_code: bool,
) -> tuple[AutoTokenizer, AutoModelForCausalLM, torch.device]:
    LOGGER.info("Loading tokenizer: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    LOGGER.info("Loading model: %s", model_id)
    quantization_config = optional_quantization_config(device, load_4bit)
    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["dtype"] = preferred_inference_dtype(device)

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    if quantization_config is None:
        model = model.to(device)
    model.eval()
    input_device = primary_model_device(model, device)
    return tokenizer, model, input_device


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Dict[str, Any]) -> str:
    ensure_parent(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def save_tensor(path: Path, payload: Dict[str, Any]) -> str:
    ensure_parent(path)
    torch.save(payload, path)
    return str(path)


def top_k_predictions(
    tokenizer: AutoTokenizer,
    final_log_probs: torch.Tensor,
    top_k: int,
) -> List[Dict[str, Any]]:
    k = min(top_k, final_log_probs.shape[-1])
    top_values, top_indices = torch.topk(final_log_probs, k=k, dim=-1)
    values = top_values.squeeze(0).detach().cpu().tolist()
    indices = top_indices.squeeze(0).detach().cpu().tolist()

    predictions: List[Dict[str, Any]] = []
    for rank, (token_id, log_probability) in enumerate(zip(indices, values), start=1):
        predictions.append(
            {
                "rank": rank,
                "token_id": int(token_id),
                "token": tokenizer.convert_ids_to_tokens(int(token_id)),
                "decoded_token": tokenizer.decode([int(token_id)], skip_special_tokens=False),
                "log_probability": float(log_probability),
                "probability": float(math.exp(log_probability)),
            }
        )
    return predictions


def token_logprob_rows(
    tokenizer: AutoTokenizer,
    labels: torch.Tensor,
    actual_log_probs: torch.Tensor,
) -> List[Dict[str, Any]]:
    label_ids = labels.squeeze(0).detach().cpu().tolist()
    log_probs = actual_log_probs.squeeze(0).detach().cpu().tolist()
    rows: List[Dict[str, Any]] = []
    for position, (token_id, log_probability) in enumerate(zip(label_ids, log_probs), start=1):
        rows.append(
            {
                "position": position,
                "token_id": int(token_id),
                "token": tokenizer.convert_ids_to_tokens(int(token_id)),
                "decoded_token": tokenizer.decode([int(token_id)], skip_special_tokens=False),
                "log_probability": float(log_probability),
                "probability": float(math.exp(log_probability)),
            }
        )
    return rows


def build_base_result(
    sample_id: str,
    text: str,
    model_id: str,
    max_length: int,
) -> Dict[str, Any]:
    return {
        "sample_id": sample_id,
        "model_id": model_id,
        "text_preview": preview_text(text),
        "character_count": len(text),
        "word_count": len(text.split()),
        "token_count": None,
        "max_length": max_length,
        "was_truncated": None,
        "runtime_seconds": None,
        "tokens_per_second": None,
        "ram_before_mb": None,
        "ram_after_mb": None,
        "peak_cuda_memory_allocated_mb": None,
        "negative_log_likelihood": None,
        "perplexity": None,
        "target_token_logprobs_path": None,
        "final_logits_path": None,
        "final_probability_vector_path": None,
        "all_token_probability_vectors_path": None,
        "top_k_next_token_predictions": [],
        "status": "pending",
        "error_message": None,
    }


def evaluate_one_sample(
    sample_id: str,
    text: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    input_device: torch.device,
    args: argparse.Namespace,
    artifacts_dir: Path,
) -> Dict[str, Any]:
    result = build_base_result(sample_id=sample_id, text=text, model_id=args.model_id, max_length=args.max_length)

    if not text.strip():
        raise ValueError("Input text is empty.")

    full_token_ids = tokenizer(
        text,
        add_special_tokens=True,
        truncation=False,
        return_attention_mask=False,
    )["input_ids"]
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask")
    token_count = int(input_ids.shape[1])
    was_truncated = len(full_token_ids) > token_count
    result["token_count"] = token_count
    result["was_truncated"] = was_truncated

    if token_count < 2:
        raise ValueError(
            "At least 2 tokens are required to compute next-token log probabilities and perplexity."
        )

    ram_before_mb = get_ram_usage_mb()
    result["ram_before_mb"] = round(ram_before_mb, 4)

    if input_device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(input_device)
        torch.cuda.synchronize(input_device)

    started_at = time.perf_counter()
    with torch.inference_mode():
        input_ids = input_ids.to(input_device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(input_device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            use_cache=False,
        )
        logits = outputs.logits
        shifted_logits = logits[:, :-1, :]
        shifted_labels = input_ids[:, 1:]
        shifted_log_probs = torch.log_softmax(shifted_logits, dim=-1)
        actual_log_probs = shifted_log_probs.gather(
            dim=-1,
            index=shifted_labels.unsqueeze(-1),
        ).squeeze(-1)
        negative_log_likelihood = float(-actual_log_probs.mean().item())
        perplexity = float(safe_exp(negative_log_likelihood))
        final_logits = logits[:, -1, :]
        final_log_probs = torch.log_softmax(final_logits, dim=-1)

    if input_device.type == "cuda":
        torch.cuda.synchronize(input_device)
    runtime_seconds = time.perf_counter() - started_at
    ram_after_mb = get_ram_usage_mb()

    result["runtime_seconds"] = round(runtime_seconds, 6)
    result["tokens_per_second"] = round(token_count / runtime_seconds, 6) if runtime_seconds > 0 else None
    result["ram_after_mb"] = round(ram_after_mb, 4)
    if input_device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(input_device) / (1024 * 1024)
        result["peak_cuda_memory_allocated_mb"] = round(peak_memory_mb, 4)

    result["negative_log_likelihood"] = round(negative_log_likelihood, 8)
    result["perplexity"] = round(perplexity, 8) if math.isfinite(perplexity) else perplexity
    result["top_k_next_token_predictions"] = top_k_predictions(
        tokenizer=tokenizer,
        final_log_probs=final_log_probs,
        top_k=args.top_k,
    )

    sample_file_stem = f"{sanitize_name(sample_id)}_{sanitize_name(args.model_id.split('/')[-1])}"

    if not args.skip_target_token_logprobs:
        target_rows = token_logprob_rows(
            tokenizer=tokenizer,
            labels=shifted_labels,
            actual_log_probs=actual_log_probs,
        )
        target_path = artifacts_dir / "target_token_logprobs" / f"{sample_file_stem}.json"
        result["target_token_logprobs_path"] = write_json(
            target_path,
            {
                "sample_id": sample_id,
                "model_id": args.model_id,
                "max_length": args.max_length,
                "token_count": token_count,
                "was_truncated": was_truncated,
                "rows": target_rows,
            },
        )

    if args.save_final_logits:
        final_logits_path = artifacts_dir / "final_logits" / f"{sample_file_stem}.pt"
        result["final_logits_path"] = save_tensor(
            final_logits_path,
            {
                "sample_id": sample_id,
                "model_id": args.model_id,
                "input_ids": input_ids.detach().cpu().squeeze(0),
                "final_logits": final_logits.detach().cpu().squeeze(0).float(),
            },
        )

    if args.save_final_probability_vector:
        final_probability_path = artifacts_dir / "final_probability_vectors" / f"{sample_file_stem}.pt"
        result["final_probability_vector_path"] = save_tensor(
            final_probability_path,
            {
                "sample_id": sample_id,
                "model_id": args.model_id,
                "input_ids": input_ids.detach().cpu().squeeze(0),
                "final_probabilities": torch.softmax(final_logits.detach().cpu().float(), dim=-1).squeeze(0),
            },
        )

    if args.save_all_token_probability_vectors:
        predicted_positions = int(shifted_logits.shape[1])
        if predicted_positions <= args.max_full_vector_tokens:
            all_vector_path = artifacts_dir / "all_token_probability_vectors" / f"{sample_file_stem}.pt"
            result["all_token_probability_vectors_path"] = save_tensor(
                all_vector_path,
                {
                    "sample_id": sample_id,
                    "model_id": args.model_id,
                    "input_ids": input_ids.detach().cpu().squeeze(0),
                    "probabilities": torch.softmax(shifted_logits.detach().cpu().float(), dim=-1).squeeze(0),
                },
            )
        else:
            LOGGER.warning(
                "Skipping all-token probability vector save for sample %s because %s predicted positions exceed limit %s.",
                sample_id,
                predicted_positions,
                args.max_full_vector_tokens,
            )

    result["status"] = "success"
    return result


def build_error_result(
    sample_id: str,
    text: str,
    model_id: str,
    max_length: int,
    error_message: str,
) -> Dict[str, Any]:
    result = build_base_result(sample_id=sample_id, text=text, model_id=model_id, max_length=max_length)
    result["status"] = "error"
    result["error_message"] = error_message
    return result


def write_jsonl_row(output_path: Path, row: Dict[str, Any]) -> None:
    ensure_parent(output_path)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def emit_load_failure_rows(
    output_jsonl: Path,
    rows: Iterable[Dict[str, Any]],
    model_id: str,
    max_length: int,
    message: str,
) -> None:
    for row in rows:
        write_jsonl_row(
            output_jsonl,
            build_error_result(
                sample_id=str(row["sample_id"]),
                text=str(row["text"]),
                model_id=model_id,
                max_length=max_length,
                error_message=message,
            ),
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)

    input_path = Path(args.input_file).resolve()
    output_jsonl = Path(args.output_jsonl).resolve()
    artifacts_dir = build_artifacts_dir(output_jsonl=output_jsonl, artifacts_dir_arg=args.artifacts_dir)

    if output_jsonl.exists():
        output_jsonl.unlink()

    rows = load_input_rows(
        input_path=input_path,
        text_column=args.text_column,
        id_column=args.id_column,
        limit=args.limit,
    )
    LOGGER.info("Loaded %s sample(s) from %s", len(rows), input_path)

    device = resolve_device(args.device)
    LOGGER.info("Using device: %s", device)

    try:
        tokenizer, model, input_device = load_model_and_tokenizer(
            model_id=args.model_id,
            device=device,
            load_4bit=args.load_4bit,
            trust_remote_code=args.trust_remote_code,
        )
    except Exception as exc:
        message = f"Model/tokenizer load failed: {exc}"
        LOGGER.error(message)
        emit_load_failure_rows(
            output_jsonl=output_jsonl,
            rows=rows,
            model_id=args.model_id,
            max_length=args.max_length,
            message=message,
        )
        return 1

    success_count = 0
    error_count = 0
    for row in rows:
        sample_id = str(row["sample_id"])
        text = str(row["text"])
        try:
            result = evaluate_one_sample(
                sample_id=sample_id,
                text=text,
                tokenizer=tokenizer,
                model=model,
                input_device=input_device,
                args=args,
                artifacts_dir=artifacts_dir,
            )
            success_count += 1
        except Exception as exc:
            LOGGER.exception("Sample %s failed", sample_id)
            result = build_error_result(
                sample_id=sample_id,
                text=text,
                model_id=args.model_id,
                max_length=args.max_length,
                error_message=str(exc),
            )
            error_count += 1
        write_jsonl_row(output_jsonl, result)

    LOGGER.info(
        "Finished evaluation | success=%s | error=%s | output=%s",
        success_count,
        error_count,
        output_jsonl,
    )
    return 0 if error_count == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
