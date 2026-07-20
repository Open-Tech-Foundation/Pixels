# ADR-0006: Naming and crate layout

Status: Accepted · 2026-07-20

## Context
Bare `pixels` on crates.io is taken by a well-known GPU pixel-buffer crate
(search-shadow risk acknowledged). Org convention prefixes crates with
`otf-`. Host-binding API design (e.g. any JS-facing
chainable API) is a consumer concern and is decided in the consumer's own
repository, not here.

## Decision
Repo: `opentf/pixels`. Crates: `otf-pixels` (facade), `otf-pixels-core`,
`otf-pixels-ops`, `otf-pixels-codec-*`. Host bindings are separate crates
in their consumers' repositories. The Rust terminal is
`output(format, options)` with format as data; `metadata()` is header-only.

## Consequences
+ Consistent org branding; no crates.io collision.
+ `output(format, options)` keeps the API surface small and stable as
  formats are added, and maps cleanly to any host binding.
- "pixels" discoverability is via the org/prefix, not the bare name.
