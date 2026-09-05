# ADR 0001: Clean-break installed package

- Status: accepted by the user on 2026-08-30

## Decision

Build one canonical `src/dominican_eaters` package with one installed
`dominican-eaters` CLI. The supported package floor is Python 3.11, pending server validation.
Model backends may retain separate environments but communicate through the canonical contracts.

There will be no compatibility wrappers, import aliases, deprecated command aliases, dual readers,
or dual writers. Replaced code is deleted at cutover. Historical data that remains useful is moved
by a checksummed one-time converter and validated before the old data snapshot is retired.

## Consequences

- New behavior and schemas are tested directly rather than constrained by old quirks.
- Packaging, imports, CLI help, configuration, and pure evaluation work without ML dependencies.
- Operational rollback restores Git, environment, and data snapshots instead of supporting two
  architectures.
- The root `AGENTS.md` compatibility rule is explicitly overridden for this migration by the user.
