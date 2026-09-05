"""Offline tests for the minimal LLM evaluation pipeline."""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from llm_eval_common import (bits_per_byte, load_and_validate_samples, mask_prompt_labels,
                             ns_to_seconds, safe_rate, score_target_logits)
from run_hf_math_eval import main as hf_main
from run_ollama_runtime_eval import main as ollama_main, parse_response


class DatasetValidationTests(unittest.TestCase):
    def test_real_jsonl_is_valid_and_has_five_unique_samples(self) -> None:
        samples = load_and_validate_samples(ROOT / "eval_data/llm_eval_samples.jsonl")
        self.assertEqual(5, len(samples))
        self.assertEqual(5, len({sample["id"] for sample in samples}))

    def test_prompt_expected_boundary_is_literal_and_contiguous(self) -> None:
        samples = load_and_validate_samples(ROOT / "eval_data/llm_eval_samples.jsonl")
        for sample in samples:
            self.assertTrue(sample["prompt"].endswith("\n"))
            source_lines = (ROOT / sample["source_path"]).read_text(encoding="utf-8").splitlines()
            start = sample["source_prompt_line_start"] - 1
            end = sample["source_expected_line_end"]
            self.assertEqual("\n".join(source_lines[start:end]), sample["prompt"] + sample["expected"])

    def test_missing_source_boundary_newline_is_rejected(self) -> None:
        row = json.loads((ROOT / "eval_data/llm_eval_samples.jsonl").read_text(encoding="utf-8").splitlines()[0])
        row["prompt"] = row["prompt"].rstrip("\n")
        with self.assertRaisesRegex(ValueError, "prompt content mismatch"):
            load_and_validate_samples(self._write(row))

    def _write(self, row: dict) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "samples.jsonl"
        path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
        return path

    def test_missing_field_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing fields"):
            load_and_validate_samples(self._write({"id": "x"}))

    def test_duplicate_id_is_rejected(self) -> None:
        source = ROOT / "eval_data/llm_eval_samples.jsonl"
        first = source.read_text(encoding="utf-8").splitlines()[0]
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "duplicate.jsonl"
        path.write_text(first + "\n" + first + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            load_and_validate_samples(path)

    def test_nonexistent_source_is_rejected(self) -> None:
        row = json.loads((ROOT / "eval_data/llm_eval_samples.jsonl").read_text(encoding="utf-8").splitlines()[0])
        row["source_path"] = "all-poems/not-real.md"
        with self.assertRaisesRegex(ValueError, "does not exist"):
            load_and_validate_samples(self._write(row))


class MathTests(unittest.TestCase):
    def test_prompt_mask_causal_shift_fp32_and_metrics(self) -> None:
        probabilities = torch.tensor([[[0.1, 0.7, 0.2], [0.6, 0.3, 0.1], [0.2, 0.2, 0.6]]], dtype=torch.float16)
        logits = probabilities.log()
        ids = torch.tensor([[0, 1, 0]])
        labels = mask_prompt_labels(ids, 1)
        self.assertEqual([[-100, 1, 0]], labels.tolist())
        result = score_target_logits(logits, labels)
        expected_sum = math.log(0.7) + math.log(0.6)
        self.assertAlmostEqual(expected_sum / 2, result["avg_token_logprob"], places=3)
        self.assertAlmostEqual(-expected_sum, result["negative_log_likelihood"], places=3)
        self.assertAlmostEqual(-expected_sum / 2, result["loss"], places=3)
        self.assertAlmostEqual(math.exp(result["loss"]), result["conditional_perplexity"], places=6)
        self.assertEqual(torch.float32, result["log_probs_dtype"])

    def test_bits_per_byte(self) -> None:
        self.assertAlmostEqual(1.0, bits_per_byte(math.log(2) * 2, "é"))

    @patch("run_hf_math_eval.load_backend")
    def test_hf_rejects_output_outside_repo_before_model_load(self, load_backend: Mock) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "outside.jsonl"
            code = hf_main(["--input-file", str(ROOT / "eval_data/llm_eval_samples.jsonl"), "--output-jsonl", str(output), "--overwrite"])
        self.assertEqual(1, code)
        load_backend.assert_not_called()


class OllamaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sample = {"id": "x", "source_path": "all-poems/x.md", "task_type": "continuation", "prompt": "p"}
        self.payload = {"response": "r", "done": True, "done_reason": "stop", "total_duration": 2_000_000_000,
                        "load_duration": 100_000_000, "prompt_eval_count": 4, "prompt_eval_duration": 2_000_000_000,
                        "eval_count": 6, "eval_duration": 3_000_000_000}

    def test_ns_conversion_rates_and_response_parse(self) -> None:
        row = parse_response(self.sample, "qwen3:4b", self.payload, 4.0)
        self.assertEqual(2.0, ns_to_seconds(self.payload["total_duration"]))
        self.assertEqual(2.0, row["tokens_per_second"])
        self.assertEqual(2.0, row["prompt_tokens_per_second"])
        self.assertEqual("r", row["response"])

    def test_zero_and_negative_duration_rates_are_safe(self) -> None:
        self.assertEqual(0.0, safe_rate(4, 0))
        self.assertEqual(0.0, ns_to_seconds(-1))

    def test_empty_success_response_is_rejected(self) -> None:
        payload = dict(self.payload, response="")
        with self.assertRaisesRegex(ValueError, "empty generated response"):
            parse_response(self.sample, "qwen3:4b", payload, 1.0)

    @patch("run_ollama_runtime_eval.requests.get")
    @patch("run_ollama_runtime_eval.requests.post")
    def test_request_disables_thinking_and_never_sends_expected(self, post: Mock, get: Mock) -> None:
        get.return_value.raise_for_status.return_value = None
        get.return_value.json.return_value = {"models": [{"name": "qwen3:4b"}]}
        post.return_value.raise_for_status.return_value = None
        post.return_value.json.return_value = self.payload
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.jsonl"
            code = ollama_main(["--input-file", str(ROOT / "eval_data/llm_eval_samples.jsonl"), "--output-jsonl", str(output), "--limit", "1"])
        request_json = post.call_args.kwargs["json"]
        self.assertEqual(0, code)
        self.assertIs(False, request_json["think"])
        self.assertNotIn("expected", request_json)
        self.assertEqual(load_and_validate_samples(ROOT / "eval_data/llm_eval_samples.jsonl", 1)[0]["prompt"], request_json["prompt"])

    @patch("run_ollama_runtime_eval.requests.get")
    @patch("run_ollama_runtime_eval.requests.post")
    def test_existing_output_survives_until_backend_preflight(self, post: Mock, get: Mock) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.jsonl"
            output.write_text("previous-valid-result\n", encoding="utf-8")
            def observe_preflight(*args: object, **kwargs: object) -> Mock:
                self.assertEqual("previous-valid-result\n", output.read_text(encoding="utf-8"))
                response = Mock()
                response.raise_for_status.return_value = None
                response.json.return_value = {"models": [{"name": "qwen3:4b"}]}
                return response
            get.side_effect = observe_preflight
            post.return_value.raise_for_status.return_value = None
            post.return_value.json.return_value = self.payload
            code = ollama_main(["--input-file", str(ROOT / "eval_data/llm_eval_samples.jsonl"), "--output-jsonl", str(output), "--limit", "1", "--overwrite"])
            row = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(0, code)
        self.assertEqual("success", row["status"])

    @patch("run_ollama_runtime_eval.requests.get")
    @patch("run_ollama_runtime_eval.requests.post")
    def test_per_sample_http_error_writes_error_row(self, post: Mock, get: Mock) -> None:
        get.return_value.raise_for_status.return_value = None
        get.return_value.json.return_value = {"models": [{"name": "qwen3:4b"}]}
        post.side_effect = RuntimeError("mock failure")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.jsonl"
            code = ollama_main(["--input-file", str(ROOT / "eval_data/llm_eval_samples.jsonl"), "--output-jsonl", str(output), "--limit", "1"])
            row = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(2, code)
        self.assertEqual("error", row["status"])
        self.assertIn("mock failure", row["error_message"])


if __name__ == "__main__":
    unittest.main()
