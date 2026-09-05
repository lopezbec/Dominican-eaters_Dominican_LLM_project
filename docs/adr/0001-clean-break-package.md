# ADR 0001: Clean-break installed package

- Status: accepted by the user on 2026-08-30

## Decision

Build one canonical `src/dominican_eaters` package with one installed
`dominican-eaters` CLI. The supported package floor is Python 3.11, pending server validation.
Model backends may retain separate environments but communicate through the canonical contracts.

There are no compatibility wrappers, import aliases, deprecated command aliases, dual readers,
or dual writers. Replaced source and its local archive were deleted at cutover. Any future data
conversion must be explicit, checksummed, and kept outside the application runtime.

## Consequences

- Behavior and schemas are tested directly against the canonical contracts.
- Packaging, imports, CLI help, configuration, and pure evaluation work without ML dependencies.
- Operational rollback restores Git, environment, and data snapshots instead of supporting two
  architectures.
- Repository guidance routes changes only through the maintained package and worker boundaries.
