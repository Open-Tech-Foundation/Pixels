# ADR-0012: Extract `otf-pixels-compress`

Status: Accepted · 2026-07-21

## Context
ADR-0010 put inflate and deflate in `otf-pixels-codec-png` and named the
condition for moving them:

> They live in `otf-pixels-codec-png` for now and are exported, because TIFF's
> deflate compression (M5) uses the same code and GIF's LZW (M5) shares its
> shape. **If a third consumer appears they move to a shared crate.**

M5 is that moment. Three codecs now need compression primitives:

- PNG needs inflate/deflate and both checksums.
- TIFF needs inflate/deflate (its `Deflate` compression) *and* LZW.
- GIF needs LZW.

Leaving them where they are would mean `otf-pixels-codec-tiff` depending on
`otf-pixels-codec-png` for zlib and on `otf-pixels-codec-gif` for LZW — codecs
importing codecs, which is exactly the tangle the "move on a third consumer"
clause existed to prevent. Duplicating LZW into both crates would be worse: two
copies of a bit-level decoder that has to agree exactly, with no test that they
do.

This record executes ADR-0010's stated plan rather than revisiting it. Nothing
about the "own our compression" decision changes.

## Decision
Create `otf-pixels-compress`, holding every compression and checksum primitive
the codecs share:

- `inflate` / `deflate` (RFC 1951) and the zlib wrapper (RFC 1950), moved
  verbatim from `otf-pixels-codec-png`, including `Inflater` and `ZlibStream`.
- `Crc32` and `Adler32`, moved with them.
- LZW, new: both the GIF variant (LSB-first, variable code width, explicit
  clear and end codes) and the TIFF variant (MSB-first, early change). One
  implementation parameterized by bit order rather than two, because the two
  specifications differ in framing and not in substance, and two copies would
  drift.

`otf-pixels-codec-png` re-exports what it previously exported, so nothing
downstream breaks and the public surface is unchanged.

The crate is a *primitives* crate: it knows about bit streams and byte
buffers, and nothing about images, pixels or descriptors. It does not depend
on `otf-pixels-core`. That boundary is what keeps it testable against
reference implementations directly, which is how ADR-0010 requires it to be
validated.

## Consequences
+ Codecs depend on primitives, not on each other. The dependency graph stays a
  tree, which is what makes "each format is a plugin" (ARCHITECTURE §Layer 2)
  true rather than aspirational.
+ One LZW, exercised by two formats with different framing — which is a
  stronger test of it than either format alone.
+ The reference-interop scripts keep working unchanged, because the code moved
  without changing.
- One more crate to publish and version. Since the whole workspace versions
  together, this costs a manifest rather than a release process.
- `otf-pixels-codec-png` now re-exports items it does not define. That is a
  small papercut for anyone reading the source expecting to find them there,
  and the alternative — a breaking change to the public API for an internal
  reorganisation — is worse.
