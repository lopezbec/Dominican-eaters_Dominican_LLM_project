# Speech-to-text evaluation metrics for the Dominican Eaters project

Status: metric protocol and the dependency-light evaluator are implemented for Whisper, Parakeet,
and Canary. No real model benchmark results are claimed.

## 1. Objective and scope

Evaluate whether speech models preserve the content of our Dominican Spanish dataset and can operate within the server's available resources. The email reports a Tesla T4 with 16 GB GPU memory and approximately 15.62 GB RAM; these specifications must be verified on the machine before benchmarking.

Confirmed scope: Christian meant **speech-to-text (STT)**. The first milestone is a small STT baseline evaluated against human-reviewed transcripts of the exact audio segments. The output being scored is recognized text; the objective is transcription accuracy, computational cost, and uncertainty. TTS synthesis and ASR-based evaluation of generated speech are outside this milestone.

## 2. Accuracy metrics

### Word Error Rate (WER) and error counts

Align reference and predicted word sequences using minimum edit distance. Let S, D, and I be substitutions, deletions, and insertions; N is the number of reference words:

**WER = (S + D + I) / N**

Multiply by 100 to report a percentage. Lower is better; insertions can produce values above 100%. NIST OpenASR uses WER as its primary metric and defines these error categories against human references. [NIST OpenASR challenge report, section 2.4](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=934557).

For this project, report S, D, and I alongside WER to distinguish incorrect words, omitted content, and added content. For example, reference `hola mi gente` and prediction `hola gente` have one deletion and WER = 1/3. Error counts alone do not establish the cause of a failure, and insertions alone are not a validated hallucination metric.

Proposed aggregation: corpus WER is the sum of all word edits divided by the sum of all reference words. A mean of file-level WERs is a different statistic that gives short and long recordings equal weight; label it separately. Preserve per-file counts for audit and resampling.

### Character Error Rate (CER)

Apply the same edit-distance definition to characters:

**CER = (character substitutions + deletions + insertions) / reference characters**

CER complements WER by exposing smaller spelling differences. Research advocating CER reports advantages across writing systems and human evaluation in Malayalam, English, and Arabic. This supports including CER, but is not evidence that CER is superior specifically for Dominican Spanish. [D K et al., 2024](https://arxiv.org/abs/2410.07400).

Proposed convention: normalize Unicode to NFC, lowercase, replace punctuation with spaces, collapse whitespace, and preserve Spanish accents and ñ. Tokenize WER on whitespace; exclude whitespace for CER. For `casa` versus `caso`, CER is 1/4 while WER is 1. Publish these conventions with the scores; character similarity from `SequenceMatcher` is not CER.

### Reference and normalization safeguards

These are proposed project controls, not claims that the cited papers prescribe our exact preprocessing:

- Preserve raw references and predictions alongside normalized strings.
- Score only the words actually present in each audio segment. Full lyrics or book passages are not automatically valid segment references.
- Preserve dialectal words and repetitions. Do not silently rewrite Dominican speech into formal Spanish or remove parenthetical spoken content.
- Freeze a policy for numbers, abbreviations, and annotations before comparing systems. Report a separate strict score if punctuation/capitalization matters.
- An empty hypothesis against a nonempty reference is all deletions and must be scored, not skipped. A failed inference is recorded separately from a successful empty transcript.
- An empty reference has an undefined per-file rate; preserve insertion counts, flag the row, and report silence controls separately. An all-empty reference corpus has no defined WER/CER.

Whisper's research provides context for multilingual transcription evaluation and robustness, but published model results do not replace evaluation on our own domain. [Radford et al., 2022](https://arxiv.org/abs/2212.04356).

## 3. Speed and resource metrics

### Inference time and Real-Time Factor

**RTF = processing time in seconds / audio duration in seconds**

Use the input audio duration. An RTF of 0.25 means processing takes one quarter of the audio duration. Some publications report the inverse, RTFx: larger is faster. NVIDIA's ASR performance discussion uses inverse real-time factor; do not mix it with RTF. [NVIDIA NeMo performance article](https://developer.nvidia.com/blog/accelerating-leaderboard-topping-asr-models-10x-with-nvidia-nemo/).

Proposed protocol: record model loading separately; warm up before steady-state measurements; use a monotonic clock; and state whether decoding, resampling, transfers, and file writing are included. Keep inference, model loading, and metric computation timings separate. Compute aggregate RTF as total measured time divided by total audio duration, alongside file-level timing summaries. Zero-duration audio is invalid. Record batch size and repeat runs when measuring variability.

CUDA operations are asynchronous, so synchronize the measured GPU before starting and after finishing a wall-clock interval. PyTorch also distinguishes allocated tensor memory from reserved allocator memory. [PyTorch CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html).

### GPU memory and system RAM

Proposed measurements: peak GPU allocated bytes and peak reserved bytes during a defined inference window, with the loaded model included. Record loading peaks separately. Reset peak statistics between samples, and label PyTorch measurements as allocator statistics rather than total device usage. If device-level telemetry is collected, report it separately because other processes and allocations outside PyTorch affect it. [PyTorch CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html).

For RAM, collect process resident set size (RSS), distinct from machine-wide memory use. Sample during inference and report the sampling interval: sampled peaks can miss short spikes. Include child processes when applicable and document the accounting convention. Store bytes and display MiB/GiB. Unsupported measurements should be null with an explanation, never a fabricated zero. The psutil API documents process RSS and system memory separately. [psutil API reference](https://psutil.io/api/).

These measurements test feasibility on the reported VM; parameter count or quantization alone cannot guarantee that a model will fit.

## 4. Confidence intervals

Bootstrap resampling can estimate evaluation uncertainty. Speech samples may be dependent, particularly when they share a speaker; blockwise bootstrap addresses this limitation of independently resampling utterances. [Liu and Peng, 2020](https://www.isca-archive.org/interspeech_2020/liu20c_interspeech.html).

Proposed protocol: use a fixed random seed and 2,000 bootstrap replicates, recomputing corpus WER and CER from summed counts in each replicate. Report the 2.5th and 97.5th percentiles as a 95% interval. Resample independent recordings; when multiple clips derive from one recording, speaker, or source, define and resample appropriate groups instead. Publish the grouping rule, number of groups, and valid replicate count. A single group cannot provide a useful uncertainty estimate.

When comparing models, score the same samples and use paired bootstrap differences with identical resampled groups. Overlapping marginal confidence intervals do not by themselves answer whether a difference is statistically meaningful. Report missing outputs and failures, including any common-subset restriction. The interval describes uncertainty conditional on this dataset and protocol, not protection from biased references or domain mismatch.

## 5. First implementation milestone

Start with multilingual Whisper `base`, with Spanish explicitly selected, as the first real-data
baseline. Parakeet and Canary are already integrated as isolated candidates but still require the
same real server and dataset validation. No TTS synthesis is part of this milestone.

Proposed sequence:

1. Freeze a small reviewed dataset manifest with sample IDs, source/group IDs, split, text, paths, and hashes. Preserve source separation between development and evaluation.
2. Run one model in its isolated environment with fixed language and precision. Pinning model
   revisions and recording repeat seeds remain required before publishing comparative results;
   current adapters process one file per request.
3. Transcribe each audio file without supplying its reference text to the model. Audit a sample by listening and comparing the output against the reviewed reference.
4. Save raw outputs, per-sample counts, durations, timing, memory, status/errors, corpus metrics,
   and confidence intervals in the canonical JSON artifacts. A CSV exporter remains pending.
5. Test exact matches, S/D/I examples, Spanish normalization, empty outputs, aggregation, invalid audio, missing files, and deterministic bootstrap behavior. Run a real server smoke test before a full dataset benchmark.

Defer broad repository restructuring and additional model integrations until this vertical slice is validated. Later CLI integration can delegate to benchmark modules without changing the existing data-collection contracts.

## 6. Implementation status

The first implementation now targets STT only:

| Component | Implemented | Remaining validation |
| --- | --- | --- |
| `src/dominican_eaters/text_metrics.py` | Canonical word/character S/D/I, corpus rates, and Spanish-preserving normalization | Review reference policy on the real dataset |
| `src/dominican_eaters/evaluation/asr/scoring.py` | Coverage, silence controls, corpus summaries, and grouped confidence intervals | Score reviewed server references |
| `src/dominican_eaters/evaluation/resources.py` | Monotonic model/inference timing and sampled host/process-tree RSS | Validate sampling behavior on the server |
| `src/dominican_eaters/speech/asr/whisper.py` | Lazy Whisper loading, explicit device/precision, audio duration, and CUDA allocator peaks | Real Whisper inference and T4 measurements |
| `src/dominican_eaters/speech/asr/worker_protocol.py` and `subprocess_backend.py` | Strict versioned JSONL boundary and lifecycle-safe model subprocess | Real worker failure and timeout exercises on the server |
| `workers/nemo` | Lazy Parakeet/Canary loading, explicit Spanish, short-audio policy, CUDA synchronization and allocator peaks | Real NeMo/T4 inference |
| `src/dominican_eaters/evaluation/asr/runner.py` | Lifecycle, RTF, scoring, final artifacts, and interruption-safe per-sample checkpoints | Real-data smoke test |
| `dominican-eaters stt benchmark` | Unified three-backend command with trustworthy exit status and no eager model imports | Server environment smoke test |

See the root `README.md` for the canonical commands and measurement boundaries. Paired model
comparison, strict punctuation scores, model revision pinning, and additional audio-language backends
remain future extensions. Runtime measurements are stored with raw transcription records; they
are not inferred from historical transcription JSON files that lack measurements.

The source-only ZIP does not supply a benchmark dataset or measured server results. Unit and mocked integration tests are not evidence of transcription quality or hardware feasibility. No real dataset inference result has been verified in this stage.
