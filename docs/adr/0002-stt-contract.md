# ADR 0002: First vertical slice is STT evaluation

- Status: accepted for implementation on 2026-08-30

## Decision

Implement the first clean vertical slice as:

`strict manifest → injected ASR backend → explicit results → corpus scoring and uncertainty`

Samples use stable IDs independent of paths. Manifests have one schema version and a declared
dataset root. Transcription outcomes distinguish successful empty text, failed inference, and
missing output. Scoring uses one documented normalization policy, word/character edit counts,
corpus WER/CER, coverage, and grouped percentile bootstrap intervals.

## Boundaries

- Model libraries are imported only by backend adapters in their isolated environment.
- Pure contracts, manifest validation, and scoring have no Torch or Whisper dependency.
- Production transcription does not import benchmark orchestration.
- Historical result formats and metric policies are not supported by the new runtime.
