#!/usr/bin/env python3
"""Score five literal poem continuations and generate deterministic HF output."""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from llm_eval_common import (append_jsonl, bits_per_byte, load_and_validate_samples,
                             mask_prompt_labels, reset_output, repo_relative,
                             repo_root, safe_rate, score_target_logits, validate_output,
                             target_logprob_rows)

LOGGER = logging.getLogger("hf_math_eval")
MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run conditional continuation math and generation with one Hugging Face causal LM.")
    parser.add_argument("--input-file", default="eval_data/llm_eval_samples.jsonl")
    parser.add_argument("--output-jsonl", default="outputs/llm_eval/hf_math_results.jsonl")
    parser.add_argument("--model-id", default=MODEL)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--save-final-probability-vector", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser.parse_args(argv)


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return requested


def base_row(sample: Dict[str, Any], args: argparse.Namespace, metadata: Dict[str, Any]) -> Dict[str, Any]:
    row = {
        "model_id": args.model_id, "model_revision": metadata.get("revision", args.revision),
        "sample_id": sample["id"], "source_path": sample["source_path"], "task_type": sample["task_type"],
        "status": "error", "prompt_token_count": None, "target_token_count": None, "total_token_count": None,
        "generated_token_count": None, "prompt_was_truncated": False, "target_was_truncated": False,
        "loss": None, "avg_token_logprob": None, "negative_log_likelihood": None,
        "mean_negative_log_likelihood": None, "conditional_perplexity": None, "perplexity_proxy": None,
        "expected_byte_count": len(sample["expected"].encode("utf-8")), "bits_per_byte": None,
        "scoring_duration_seconds": None, "generation_duration_seconds": None,
        "scoring_tokens_per_second": None, "generation_tokens_per_second": None, "generated_text": None,
        "target_token_logprobs_path": None, "final_probability_vector_path": None,
        "device": metadata.get("device"), "dtype": metadata.get("dtype"),
        "quantization": metadata.get("quantization"), "torch_version": torch.__version__,
        "transformers_version": transformers.__version__, "error_message": None,
    }
    return row


def load_backend(args: argparse.Namespace, device: str) -> Tuple[Any, Any, Dict[str, Any]]:
    if device == "cpu" and args.model_id == MODEL:
        raise RuntimeError("Refusing unreasonable CPU execution for Qwen/Qwen3-4B-Instruct-2507; use CUDA")
    local_only = not args.allow_download
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision, local_files_only=local_only, trust_remote_code=args.trust_remote_code)
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else (torch.float16 if device == "cuda" else torch.float32)
    kwargs: Dict[str, Any] = {"revision": args.revision, "local_files_only": local_only, "trust_remote_code": args.trust_remote_code, "low_cpu_mem_usage": True}
    quantization = "none"
    if args.load_4bit:
        if device != "cuda":
            raise RuntimeError("--load-4bit requires CUDA")
        try:
            import bitsandbytes as bnb
            if not hasattr(bnb, "nn") or not hasattr(bnb.nn, "Linear4bit"):
                raise RuntimeError("bitsandbytes Linear4bit is unavailable")
        except Exception as exc:
            raise RuntimeError(f"bitsandbytes is not functional: {exc}") from exc
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)
        kwargs["device_map"] = "auto"
        quantization = "bitsandbytes-nf4-4bit"
    else:
        kwargs["dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **kwargs)
    if not args.load_4bit:
        model.to(device)
    model.eval()
    revision = getattr(model.config, "_commit_hash", None) or args.revision
    return tokenizer, model, {"revision": revision, "device": device, "dtype": str(dtype).replace("torch.", ""), "quantization": quantization}


def encode_continuation(tokenizer: Any, sample: Dict[str, Any], max_length: int) -> Tuple[torch.Tensor, torch.Tensor, int, bool]:
    prompt_ids = list(tokenizer(sample["prompt"], add_special_tokens=True)["input_ids"])
    target_ids = list(tokenizer(sample["expected"], add_special_tokens=False)["input_ids"])
    if not target_ids:
        raise ValueError("Expected continuation produced zero tokens")
    if len(target_ids) >= max_length:
        raise ValueError(f"Expected continuation has {len(target_ids)} tokens and cannot fit max_length={max_length} without truncation")
    allowed_prompt = max_length - len(target_ids)
    truncated = len(prompt_ids) > allowed_prompt
    if truncated:
        prompt_ids = prompt_ids[-allowed_prompt:]
    input_ids = torch.tensor([prompt_ids + target_ids], dtype=torch.long)
    return input_ids, mask_prompt_labels(input_ids, len(prompt_ids)), len(prompt_ids), truncated


def evaluate_sample(sample: Dict[str, Any], args: argparse.Namespace, tokenizer: Any, model: Any, metadata: Dict[str, Any]) -> Dict[str, Any]:
    row = base_row(sample, args, metadata)
    input_ids, labels, prompt_count, truncated = encode_continuation(tokenizer, sample, args.max_length)
    input_device = next(model.parameters()).device
    input_ids, labels = input_ids.to(input_device), labels.to(input_device)
    if input_device.type == "cuda": torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        logits = model(input_ids=input_ids, use_cache=False, return_dict=True).logits
        metrics = score_target_logits(logits, labels)
    if input_device.type == "cuda": torch.cuda.synchronize()
    scoring_seconds = time.perf_counter() - started
    target_ids = input_ids[0, prompt_count:].detach().cpu().tolist()
    values = metrics.pop("target_logprobs").detach().cpu().tolist()
    metrics.pop("log_probs_dtype")
    artifact = Path(args.output_jsonl).resolve().parent / "hf_target_token_logprobs" / f"{sample['id']}.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(target_logprob_rows(target_ids, values), ensure_ascii=False, indent=2), encoding="utf-8")
    prompt_only = input_ids[:, :prompt_count]
    vector_path = None
    if args.save_final_probability_vector:
        with torch.inference_mode():
            final_logits = model(input_ids=prompt_only, use_cache=False, return_dict=True).logits[:, -1, :]
            probabilities = torch.softmax(final_logits, dim=-1, dtype=torch.float32).squeeze(0).cpu()
        vector = Path(args.output_jsonl).resolve().parent / "hf_final_probability_vectors" / f"{sample['id']}.pt"
        vector.parent.mkdir(parents=True, exist_ok=True)
        torch.save(probabilities, vector)
        vector_path = repo_relative(vector)
    if input_device.type == "cuda": torch.cuda.synchronize()
    generated_started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(input_ids=prompt_only, do_sample=False, max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.eos_token_id)
    if input_device.type == "cuda": torch.cuda.synchronize()
    generation_seconds = time.perf_counter() - generated_started
    new_ids = generated[0, prompt_count:]
    generated_count = int(new_ids.numel())
    row.update(metrics)
    row.update({"status": "success", "prompt_token_count": prompt_count, "target_token_count": len(target_ids),
                "total_token_count": int(input_ids.shape[1]), "generated_token_count": generated_count,
                "prompt_was_truncated": truncated, "target_was_truncated": False,
                "expected_byte_count": len(sample["expected"].encode("utf-8")),
                "bits_per_byte": bits_per_byte(metrics["negative_log_likelihood"], sample["expected"]),
                "scoring_duration_seconds": scoring_seconds, "generation_duration_seconds": generation_seconds,
                "scoring_tokens_per_second": safe_rate(len(target_ids), scoring_seconds),
                "generation_tokens_per_second": safe_rate(generated_count, generation_seconds),
                "generated_text": tokenizer.decode(new_ids, skip_special_tokens=True),
                "target_token_logprobs_path": repo_relative(artifact), "final_probability_vector_path": vector_path})
    return row


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s")
    random.seed(args.seed); torch.manual_seed(args.seed)
    try:
        samples = load_and_validate_samples(Path(args.input_file).resolve(), args.limit)
        output = Path(args.output_jsonl).resolve()
        output.relative_to(repo_root())
        validate_output(output, args.overwrite)
        device = resolve_device(args.device)
    except Exception as exc:
        LOGGER.error("Preflight failed: %s", exc); return 1
    metadata: Dict[str, Any] = {"device": device, "dtype": None, "quantization": None, "revision": args.revision}
    try:
        tokenizer, model, metadata = load_backend(args, device)
    except Exception as exc:
        message = f"Model/tokenizer load failed: {exc}"; LOGGER.error("%s", message)
        reset_output(output)
        for sample in samples:
            row = base_row(sample, args, metadata); row["error_message"] = message; append_jsonl(output, row)
        return 1
    reset_output(output)
    errors = 0
    for sample in samples:
        try:
            row = evaluate_sample(sample, args, tokenizer, model, metadata)
        except Exception as exc:
            LOGGER.exception("Sample %s failed", sample["id"]); row = base_row(sample, args, metadata); row["error_message"] = str(exc); errors += 1
        append_jsonl(output, row)
    return 2 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
