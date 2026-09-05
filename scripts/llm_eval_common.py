#!/usr/bin/env python3
"""Shared validation, math, and JSONL helpers for the LLM smoke evaluation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch

REQUIRED_SAMPLE_FIELDS = {
    "id", "source_path", "source_prompt_line_start", "source_prompt_line_end",
    "source_expected_line_start", "source_expected_line_end", "prompt", "expected",
    "task_type",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def repo_relative(path: Path) -> str:
    return path.resolve().relative_to(repo_root()).as_posix()


def load_and_validate_samples(path: Path, limit: int = 0) -> List[Dict[str, Any]]:
    """Load JSONL and prove every excerpt is literal source content."""
    samples: List[Dict[str, Any]] = []
    seen = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"Cannot read input file {path}: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        try:
            sample = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on input line {line_number}: {exc}") from exc
        if not isinstance(sample, dict):
            raise ValueError(f"Input line {line_number} must be a JSON object")
        missing = sorted(REQUIRED_SAMPLE_FIELDS - sample.keys())
        if missing:
            raise ValueError(f"Input line {line_number} missing fields: {', '.join(missing)}")
        sample_id = sample["id"]
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError(f"Input line {line_number} has an empty id")
        if sample_id in seen:
            raise ValueError(f"Duplicate sample id: {sample_id}")
        seen.add(sample_id)
        if sample["task_type"] != "continuation":
            raise ValueError(f"Unsupported task_type for {sample_id}: {sample['task_type']}")
        for field in ("prompt", "expected"):
            if not isinstance(sample[field], str) or not sample[field]:
                raise ValueError(f"Sample {sample_id} has empty {field}")
        source = repo_root() / str(sample["source_path"])
        if not source.is_file():
            raise ValueError(f"Source path does not exist for {sample_id}: {sample['source_path']}")
        source_lines = source.read_text(encoding="utf-8").splitlines()
        ranges = {}
        for prefix, text_field in (("prompt", "prompt"), ("expected", "expected")):
            start = sample[f"source_{prefix}_line_start"]
            end = sample[f"source_{prefix}_line_end"]
            if not isinstance(start, int) or not isinstance(end, int) or start < 1 or end < start or end > len(source_lines):
                raise ValueError(f"Invalid {prefix} line range for {sample_id}: {start}-{end}")
            literal = "\n".join(source_lines[start - 1:end]) + ("\n" if prefix == "prompt" else "")
            if literal != sample[text_field]:
                raise ValueError(f"{prefix} content mismatch for {sample_id} at lines {start}-{end}")
            ranges[prefix] = (start, end)
        if ranges["expected"][0] != ranges["prompt"][1] + 1:
            raise ValueError(f"Prompt and expected ranges are not adjacent for {sample_id}")
        contiguous = "\n".join(source_lines[ranges["prompt"][0] - 1:ranges["expected"][1]])
        if sample["prompt"] + sample["expected"] != contiguous:
            raise ValueError(f"Reconstructed continuation does not match contiguous source for {sample_id}")
        samples.append(sample)
    return samples[:limit] if limit > 0 else samples


def safe_exp(value: float) -> float:
    return math.exp(value) if value <= 80 else float("inf")


def score_target_logits(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, Any]:
    """Score non-masked labels using causal shift and FP32 log-softmax."""
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    mask = shifted_labels.ne(-100)
    count = int(mask.sum().item())
    if count == 0:
        raise ValueError("No target tokens remain after causal shift")
    log_probs = torch.log_softmax(shifted_logits, dim=-1, dtype=torch.float32)
    safe_labels = shifted_labels.masked_fill(~mask, 0)
    selected = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)[mask]
    total = float(selected.sum().item())
    average = total / count
    loss = -average
    nll = -total
    return {
        "target_logprobs": selected,
        "target_token_count": count,
        "avg_token_logprob": average,
        "loss": loss,
        "negative_log_likelihood": nll,
        "mean_negative_log_likelihood": loss,
        "conditional_perplexity": safe_exp(loss),
        "perplexity_proxy": safe_exp(loss),
        "log_probs_dtype": log_probs.dtype,
    }


def bits_per_byte(negative_log_likelihood: float, expected: str) -> float:
    byte_count = len(expected.encode("utf-8"))
    if byte_count == 0:
        raise ValueError("Expected text has zero UTF-8 bytes")
    return negative_log_likelihood / (math.log(2) * byte_count)


def ns_to_seconds(value: Any) -> float:
    number = float(value or 0)
    return max(number, 0.0) / 1_000_000_000


def safe_rate(count: Any, seconds: float) -> float:
    return float(count or 0) / seconds if seconds > 0 else 0.0


def validate_output(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace it: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def reset_output(path: Path) -> None:
    if path.exists():
        path.unlink()


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def mask_prompt_labels(input_ids: torch.Tensor, prompt_token_count: int) -> torch.Tensor:
    labels = input_ids.clone()
    labels[:, :prompt_token_count] = -100
    return labels


def target_logprob_rows(token_ids: Sequence[int], values: Sequence[float]) -> List[Dict[str, Any]]:
    return [{"position": index, "token_id": int(token_id), "log_probability": float(value)} for index, (token_id, value) in enumerate(zip(token_ids, values), 1)]
