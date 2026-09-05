# ADR 0003: Isolate incompatible ASR runtimes behind a process protocol

- Status: accepted for implementation on 2026-09-03

## Decision

Run NeMo-family models in separately packaged worker processes. The base `dominican-eaters`
package communicates with each worker through a strict, versioned JSON Lines protocol over
standard input and output. Logs use standard error. The host starts workers with an absolute
Python executable and an argument vector without a shell.

Each request carries a unique request ID and one lifecycle command: load, warmup, transcribe, or
close. Responses echo that ID and contain either a validated success payload or a structured
error. Unknown fields, versions, commands, response IDs, malformed JSON, oversized lines,
timeouts, and premature process exits are failures.

The worker process is the GPU-resource boundary. Graceful close is attempted first; timeout or
protocol failure terminates the process. A worker imports Torch and NeMo only after validating its
startup configuration. The base package remains importable without either dependency.

## Consequences

- Whisper and NeMo dependency solvers cannot contaminate each other's runtime.
- Worker stdout cannot contain progress bars or logs.
- Model-specific objects and tensors must be normalized to canonical JSON values inside the
  worker.
- The first contract handles one audio path per request so failures retain exact sample identity.
- Batching and long-form chunking require explicit future protocol revisions.
- There are no adapters for the historical benchmark JSON formats or shell scripts.
