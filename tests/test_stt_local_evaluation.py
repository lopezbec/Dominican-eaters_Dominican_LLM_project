"""Lightweight contract checks for local STT evaluation."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import itertools
from unittest.mock import patch
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from testing.stt.evaluate_with_local_refs import (  # noqa: E402
    character_error_rate,
    load_manifest,
    main,
    word_error_rate,
)
from testing.stt.scoring import alignment_counts, bootstrap_intervals, corpus_rate, normalize_text, score_text
from testing.stt.benchmark import measure_call, parse_args, preflight, run


def test_measurement_boundaries_and_cleanup() -> None:
    from types import SimpleNamespace
    import threading

    calls = []
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(
        synchronize=lambda: calls.append('sync'),
        reset_peak_memory_stats=lambda: calls.append('reset'),
        max_memory_allocated=lambda: 1000,
        max_memory_reserved=lambda: 2000,
    ))
    fake_psutil = SimpleNamespace(Process=lambda: SimpleNamespace(
        memory_info=lambda: SimpleNamespace(rss=3000)), Error=OSError)
    before = set(threading.enumerate())
    with patch.dict(sys.modules, {'psutil': fake_psutil}):
        with patch('testing.stt.benchmark.time.perf_counter', side_effect=[10, 10.25]):
            result, measurements = measure_call(lambda: 'output', fake_torch, 'cuda', 1)
        assert result == 'output'
        assert calls == ['sync', 'reset', 'sync']
        assert measurements['inference_seconds'] == 0.25
        assert measurements['gpu_peak_allocated_bytes'] == 1000
        assert measurements['gpu_peak_reserved_bytes'] == 2000
        assert measurements['ram_peak_rss_bytes'] == 3000
        _, cpu = measure_call(lambda: None, fake_torch, 'cpu', 1)
        assert cpu['gpu_peak_allocated_bytes'] is None
        def fail():
            raise RuntimeError('expected failure')
        try:
            measure_call(fail, fake_torch, 'cpu', 1)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Inference failure must propagate')
    assert set(threading.enumerate()) == before


def test_alignment_counts_and_normalization() -> None:
    assert normalize_text("¡ÑO! cafe\u0301 (hola)") == "ño café hola"
    assert score_text("hola mi gente", "hola gente")['word_deletions'] == 1
    assert score_text("casa", "caso")['word_substitutions'] == 1
    assert score_text("hola", "hola mi gente")['word_insertions'] == 2
    assert word_error_rate("hola", "adiós mi gente") == 3
    assert word_error_rate("hola", "") == 1
    assert word_error_rate("", "hola") is None
    # Exhaustively compare small inputs to an independent scalar DP oracle.
    sequences = [p for n in range(4) for p in itertools.product('ab', repeat=n)]
    for ref in sequences:
        for hyp in sequences:
            matrix = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
            for i in range(len(ref) + 1):
                matrix[i][0] = i
            for j in range(len(hyp) + 1):
                matrix[0][j] = j
            for i, a in enumerate(ref, 1):
                for j, b in enumerate(hyp, 1):
                    matrix[i][j] = min(matrix[i-1][j]+1, matrix[i][j-1]+1,
                                       matrix[i-1][j-1]+(a != b))
            counts = alignment_counts(ref, hyp)
            assert counts['errors'] == matrix[-1][-1]
            hits = len(ref) - counts['substitutions'] - counts['deletions']
            assert hits + counts['substitutions'] + counts['insertions'] == len(hyp)


def test_corpus_and_grouped_bootstrap() -> None:
    rows = [{**score_text('uno', ''), 'group_id': 'a'},
            {**score_text('uno dos tres cuatro', 'uno dos tres cuatro'), 'group_id': 'b'}]
    assert corpus_rate(rows, 'word') == 0.2
    first = bootstrap_intervals(rows, 200, 42)
    assert first == bootstrap_intervals(rows, 200, 42)
    assert first['wer'] == [0.0, 1.0]
    rows[1]['group_id'] = 'a'
    assert bootstrap_intervals(rows, 200, 42)['wer'] is None
    silence = [{**score_text('', 'hola'), 'group_id': 'silence'}]
    assert corpus_rate(silence, 'word') is None


def test_empty_failed_missing_and_duplicate_outputs() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest = root / 'manifest.json'
        manifest.write_text(json.dumps([{'audio_file': name, 'reference_text': 'hola'}
                                        for name in ['empty.wav', 'failed.wav', 'missing.wav']]))
        model_dir = root / 'transcriptions' / 'whisper'
        model_dir.mkdir(parents=True)
        (model_dir / 'empty.json').write_text(json.dumps({'file': 'empty.wav', 'transcript': ''}))
        (model_dir / 'failed.json').write_text(json.dumps({'file': 'failed.wav', 'status': 'error', 'error': 'OOM'}))
        args = ['--manifest', str(manifest), '--transcriptions-dir', str(model_dir.parent),
                '--models', 'whisper', '--output-dir', str(root / 'scores')]
        main(args)
        report = json.loads((root / 'scores/asr_leaderboard.json').read_text())
        assert report['models'][0]['wer_corpus'] == 1
        assert report['models'][0]['unscored_samples'] == 2
        assert [r['status'] for r in report['coverage']] == ['ok', 'error', 'missing']
        (model_dir / 'duplicate.json').write_text((model_dir / 'empty.json').read_text())
        try:
            main(args)
        except ValueError as exc:
            assert 'Duplicate output' in str(exc)
        else:
            raise AssertionError('Duplicate outputs must fail')


def test_manifest_preflight_and_duplicate_validation() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest = root / 'manifest.json'
        (root / 'a.wav').write_bytes(b'preflight only checks existence and hash')
        rows = [{'audio_file': 'a.wav', 'reference_text': 'hola', 'split': 'eval'}]
        manifest.write_text(json.dumps(rows))
        args = parse_args(['--manifest', str(manifest), '--output-dir', str(root / 'output'), '--preflight'])
        assert run(args)['samples'] == 1
        assert not (root / 'output').exists()
        assert preflight(manifest, 'eval', 1)[0]['audio_path'] == str(root / 'a.wav')
        try:
            preflight(manifest, 'train', None)
        except ValueError as exc:
            assert 'No samples' in str(exc)
        else:
            raise AssertionError('An empty split must fail')
        (root / 'ref.txt').write_text('hola')
        manifest.write_text(json.dumps([{'audio_file': 'a.wav', 'reference_text_path': 'ref.txt'}]))
        frozen = preflight(manifest, None, None)[0]
        assert frozen['reference_text'] == 'hola' and frozen['reference_text_path'] == ''
        (root / 'ref.txt').unlink()
        manifest.write_text(json.dumps([frozen]))
        assert load_manifest(manifest)['a.wav'].reference_text == 'hola'
        manifest.write_text(json.dumps(rows + rows))
        try:
            load_manifest(manifest)
        except ValueError as exc:
            assert 'Duplicate audio_file' in str(exc)
        else:
            raise AssertionError('Duplicate references must fail')


def test_mocked_benchmark_end_to_end() -> None:
    """Exercise orchestration and scoring without treating a fake backend as inference."""
    from types import SimpleNamespace
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False),
                                 manual_seed=lambda seed: None, version=SimpleNamespace(cuda=None))
    fake_numpy = SimpleNamespace(random=SimpleNamespace(seed=lambda seed: None),
                                 isfinite=lambda audio: SimpleNamespace(all=lambda: True))
    fake_model = SimpleNamespace(transcribe=lambda audio, **kw: {'text': 'hola', 'language': 'es'})
    fake_whisper = SimpleNamespace(load_model=lambda name, device: fake_model,
                                  load_audio=lambda path: [0] * 16000)
    def measured(call, *args):
        return call(), {'inference_seconds': 0.25}
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / 'a.wav').write_bytes(b'fake audio; not a real inference test')
        manifest = root / 'manifest.json'
        manifest.write_text(json.dumps([{'audio_file': 'a.wav', 'reference_text': 'hola gente'}]))
        args = parse_args(['--manifest', str(manifest), '--output-dir', str(root / 'run'), '--warmup', '0'])
        with patch.dict(sys.modules, {'torch': fake_torch, 'numpy': fake_numpy, 'whisper': fake_whisper}), \
             patch('testing.stt.benchmark.measure_call', measured), \
             patch('testing.stt.benchmark.importlib.metadata.version', return_value='test'):
            report = run(args)
            assert report['status'] == 'complete'
            assert report['rtf'] == 0.25
            scores = json.loads((root / 'run/scores/asr_leaderboard.json').read_text())
            assert scores['models'][0]['wer_corpus'] == 0.5
            fake_model.transcribe = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('backend failed'))
            args.output_dir = str(root / 'failed')
            report = run(args)
            assert report['status'] == 'partial' and report['failed'] == 1
            assert report['rtf'] is None


def test_metric_calculation() -> None:
    assert word_error_rate("hola mundo", "hola mundo") == 0.0
    assert word_error_rate("hola mundo", "hola") == 0.5
    assert character_error_rate("casa", "caso") == 0.25


def test_manifest_supports_inline_text_and_alignment() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        manifest_path = Path(temp_dir) / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                [
                    {
                        "audio_file": "sample.m4a",
                        "reference_text": "Hola mundo",
                        "split": "eval",
                        "source": "human-transcript",
                        "alignment_label": {
                            "match_second": 12.5,
                            "matched_word": "Hola",
                        },
                    }
                ]
            ),
            encoding="utf-8",
        )

        manifest = load_manifest(manifest_path)

        sample = manifest["sample.m4a"]
        assert sample.reference_text == "Hola mundo"
        assert sample.split == "eval"
        assert sample.source == "human-transcript"
        assert sample.alignment_match_second == 12.5
        assert sample.alignment_matched_word == "Hola"


def test_local_evaluation_writes_ranked_outputs() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        references_dir = root / "references"
        transcriptions_dir = root / "transcriptions"
        whisper_dir = transcriptions_dir / "whisper"
        canary_dir = transcriptions_dir / "canary"
        output_dir = root / "results"

        references_dir.mkdir(parents=True)
        whisper_dir.mkdir(parents=True)
        canary_dir.mkdir(parents=True)

        reference_path = references_dir / "track.txt"
        reference_path.write_text("hola mundo", encoding="utf-8")

        manifest_path = root / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "rows": [
                        {
                            "audio_file": "track.m4a",
                            "reference_text_path": "references/track.txt",
                            "split": "eval",
                            "source": "manual-transcript",
                            "match_second": 3.2,
                            "matched_word": "hola",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        (whisper_dir / "track.json").write_text(
            json.dumps({"file": "track.m4a", "transcript": "hola mundo"}),
            encoding="utf-8",
        )
        (canary_dir / "track.json").write_text(
            json.dumps({"file": "track.m4a", "transcript": "hola"}),
            encoding="utf-8",
        )

        main(
            [
                "--transcriptions-dir",
                str(transcriptions_dir),
                "--manifest",
                str(manifest_path),
                "--models",
                "whisper",
                "canary",
                "--output-dir",
                str(output_dir),
            ]
        )

        leaderboard = json.loads(
            (output_dir / "asr_leaderboard.json").read_text(encoding="utf-8")
        )
        details_csv = (output_dir / "asr_scores_by_file.csv").read_text(
            encoding="utf-8"
        )

        assert leaderboard["models"][0]["model"] == "whisper"
        assert leaderboard["models"][0]["wer_median"] == 0.0
        assert leaderboard["models"][1]["model"] == "canary"
        assert leaderboard["meta"]["rows_scored"] == 2
        assert "alignment_match_second" in details_csv
        assert "alignment_matched_word" in details_csv


if __name__ == "__main__":
    print("Running local STT evaluation tests...")
    test_metric_calculation()
    test_manifest_supports_inline_text_and_alignment()
    test_local_evaluation_writes_ranked_outputs()
    test_alignment_counts_and_normalization()
    test_corpus_and_grouped_bootstrap()
    test_empty_failed_missing_and_duplicate_outputs()
    test_manifest_preflight_and_duplicate_validation()
    test_mocked_benchmark_end_to_end()
    test_measurement_boundaries_and_cleanup()
    print("All local STT evaluation tests passed!")
