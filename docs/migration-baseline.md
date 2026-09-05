# Clean-architecture migration baseline

Recorded on 2026-08-30 before creating the canonical package.

- Git branch: `main`
- Git commit: `2efa88d7ddb9af5c3b817005192c56ac1c767f7a`
- Source-only server archive: `Dominican-eaters_Dominican_LLM_project_source_only_20260826_225232.zip`
- Archive SHA-256: `5791c5c0c17c1fefac80c60b21f6a8b87d8547671288106a9264668a413353ce`
- Archive size: approximately 300 KiB
- Tracked files at recording time: 122
- Local default Python at recording time: 3.14.7
- Bundled validation Python available to Codex: 3.12.13
- Python 3.11 was not installed locally at recording time. `.python-version` remains 3.12 for the
  development environment while the built package declares a 3.11 compatibility floor.
- No checked-in or unignored audio/reference dataset was found by the scoped repository file scan.

The working tree was already dirty. It contained the imported server snapshot, the STT research
and baseline work, TTS experiments, documentation, and modifications to existing files. This
migration must not reset or overwrite those changes. `git status --short` is the authoritative
per-file inventory until the work is split into commits.

Known checks before migration:

- The existing focused STT test script passed nine tests in the temporary environment.
- `shared/services/transcription_exporter.py` contained a pre-existing syntax error identical to
  the tracked revision.
- Mocked old-CLI child failures returned success.
- A frozen relative-path STT manifest failed replay from its new run directory.

The original ZIP is the source-code rollback artifact. Dataset rollback requires a separate
checksummed snapshot on the server before data conversion or cutover; that snapshot does not yet
exist in this checkout.
