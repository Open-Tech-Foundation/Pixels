# ADR-0014: Own WebP, lossy and lossless, decode and encode

Status: Accepted · 2026-10-06

## Context
ADR-0004 wrapped WebP (`image-webp`) on the grounds that it is two codecs
behind one container — VP8 intra for lossy, a separate lossless format — and
that owning both was not the best use of the from-scratch budget. It also
recorded the cost: the wrapped encoder writes **lossless only**, so
`EncodeOptions::quality` is ignored and there is no way to produce the small
lossy WebP that web pipelines ask for. For an engine meant to sit behind a
runtime's image API, lossy WebP output is not optional; it is one of the two
or three formats such an API is called for most.

No pure-Rust lossy WebP encoder exists to wrap instead. The alternatives are
libwebp through C bindings, which breaks the pure-Rust, no-native-build
property ADR-0013 just restored, or owning it.

Owning it is a smaller job than ADR-0004 assumed, because most of it already
exists here in another form. VP8 lossy is a key-frame-only intra codec — for a
still image there are no inter frames to support at all — built from a 4x4 DCT
and Walsh-Hadamard transform, intra prediction, a boolean arithmetic coder and
a loop filter: the same shape of work as the owned JPEG and AV1 paths, and much
smaller than the latter. VP8L lossless is LZ77 plus canonical Huffman coding
plus a handful of reversible transforms, close kin to the owned DEFLATE
(ADR-0010).

## Decision
Reverse ADR-0004's WebP clause. Implement WebP from scratch in
`otf-pixels-codec-webp`, owning every layer, and remove the `image-webp`
dependency:

- the RIFF container: `VP8 `, `VP8L` and `VP8X` images, `ALPH` alpha, and the
  `EXIF` chunk (for orientation); animation (`ANIM`/`ANMF`) decodes its first
  frame, as today, and animation pipelines stay v2;
- VP8L (RFC 9649) decode and encode;
- VP8 (RFC 6386) key-frame decode and encode, the encoder honouring
  `EncodeOptions::quality`;
- `ALPH` decode and encode, so lossy output keeps its alpha channel.

The crate keeps its public names (`WebPCodec`, `WebPDecoder`, `WebPEncoder`),
so nothing downstream of the `Decoder`/`Encoder` traits changes, and the swap
lands one layer at a time behind them: each owned layer replaces its wrapped
counterpart only once it matches libwebp, and `image-webp` goes when the last
one does.

## Consequences
+ Lossy WebP output with real quality control, and a default build that is
  pure Rust with every format owned.
+ Owning the decoder lets lossy WebP stream by macroblock row instead of
  materializing the image, which `image-webp`'s seek-based API ruled out
  (SPEC §Formats currently lists WebP as "internal buffer").
+ Verification is against libwebp, never against ourselves: lossless decode
  must match exactly; lossy decode is bit-exact by specification, so it is
  compared at the YUV planes before colour conversion, as AV1 is against
  libaom; and libwebp must read back every file we write.
- A competitive VP8 *encoder* — rate-distortion mode decisions, adaptive
  quantization, token-probability optimisation — is far harder than a
  conformant one. The first encoder will be larger than `cwebp` at equal
  quality, and that gap is measured and stated rather than hidden.
- Another few thousand lines of index-dense codec code, mitigated as before:
  spec tables generated or checked against the RFC text, and every parser
  fuzzed.
