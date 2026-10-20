# ADR-0019: Own JPEG XL, encoder first

Status: Accepted · 2026-10-20

## Context
JPEG XL (ISO/IEC 18181) is now decoded by the major browsers, and a server
can serve it to those that send `image/jxl` in `Accept`. Every other codec
here is written in this workspace, which ships no dependencies outside it;
wrapping libjxl would break that. A JPEG XL decoder must accept every tool
any encoder may use (Modular trees, VarDCT's transforms, patches, splines,
noise, restoration filters, ANS), while an encoder may use a small subset.

## Decision
Write JPEG XL in `otf-pixels-codec-jxl`, encoder first, each phase verified
against libjxl's `djxl` (`scripts/check-jxl-interop.sh`):

1. Lossless Modular: grey or RGB (through the YCoCg RCT), alpha, 8 or 16
   bits; the clamped gradient predictor; a tree with one context per
   channel; prefix codes fitted to each context; runs of exact
   predictions as LZ77 copies; 256x256 groups behind a table of contents.
   A bare codestream, no container.
2. Compression: trees over neighbourhood properties, the weighted
   predictor, ANS, LZ77, more contexts.
3. Lossy VarDCT in XYB, the mode most images will be served in.
4. Container boxes (EXIF, ICC), then a decoder.

The field definitions were read from libjxl (BSD-3-Clause) and jxl-oxide
(MIT/Apache-2.0); the code is written here.

## Consequences
+ `Format::Jxl` output with no new dependency; files decode bit-exactly in
  libjxl at every phase.
- Until phase 2 ends, lossless files are short of libjxl's slower efforts
  (a 960x600 photo: 266 KB, against PNG's 327 KB, `cjxl -e3`'s 245 KB and
  `-e7`'s 218 KB).
- Until phase 4, `.jxl` input is not recognised.
