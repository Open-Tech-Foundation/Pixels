# ADR-0004: Codec ownership split

Status: Accepted · 2026-07-20

## Context
Target formats: JPEG, PNG, WebP, GIF, AVIF, TIFF, raw. From-scratch
feasibility ranking: raw → GIF → PNG → TIFF → JPEG baseline → WebP (hard,
two codecs: VP8 intra + separate lossless) → AVIF (not feasible; AV1 is a
decade of multi-team work — even rav1d is a port). Owning codecs serves the
project's from-scratch ethos and the region-decode contract; wrapping gets
full format coverage at 1.0.

## Decision
Own: raw, GIF, PNG, TIFF, JPEG baseline. Wrap existing crates: WebP, AVIF
(dav1d/rav1e family), progressive JPEG initially. All behind the same
`Decoder`/`Encoder` traits so ownership is swappable per format without API
change. AV1 from scratch is a permanent non-goal.

## Consequences
+ Full format table at 1.0; from-scratch depth where it pays.
+ Trait boundary makes later WebP/progressive-JPEG rewrites drop-in.
- Wrapped codecs bring their licenses and (for dav1d bindings) potential
  non-Rust build deps; evaluate rav1d to stay pure-Rust.
- Two codebases styles (ours + wrapped) until/unless rewrites happen.
