# Changelog

All notable changes to this project will be documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning: [SemVer](https://semver.org/).

## [Unreleased]

### Fixed
- Any streaming decode resized to more than one tile column failed outright
  with "source cannot rewind" — 512x512 to 300x300 on PNG or JPEG alike, while
  512x512 to 32x32 worked, the boundary being one 128px tile. Consecutive
  requests to a forward-only source *overlap* rather than abut, because each
  output row of a resize needs input rows either side of it, and the retained
  band was discarded wholesale instead of carrying those rows over. The
  scheduler and the M1 reference evaluator now agree byte for byte across the
  shapes that used to fail.
- `DecodeCapability::Regions` no longer claims to cover scaled-decode JPEG. A
  reduced-scale decode still emits rows in order and still entropy-decodes
  every coefficient; it is a decoder configuration, not random access, and the
  scheduler would have acted on the claim.
- The MSRV job now checks without dev-dependencies. `rust-version` is a
  promise about consuming the library; the benchmark's `fast_image_resize`
  needs 1.87 and no user of `otf-pixels` ever compiles it.
- The declared MSRV of 1.85 was not actually met: three `let` chains need
  1.88. Rewritten as nested `if let`, and CI now checks 1.85 so the promise
  stays true rather than aspirational.
- A build without the `raw` feature now compiles. The exit-criteria suites and
  the crate-level example are expressed over the raw codec, so they compile out
  with it rather than failing to build.

### Changed
- SPEC's JPEG fast-path paragraph now records that it describes intent rather
  than facade behaviour, and what stands in the way: the planner analyses an
  immutable graph where selecting a reduced source would require rewriting it,
  and ops carrying pixel-valued parameters (`crop`, `composite`, `convolve`)
  do not mean the same thing against a reduced source. The decoder half is
  implemented and usable directly.
- `otf-pixels` now evaluates pipelines with the tile scheduler instead of the
  M1 whole-image evaluator. Rows still reach an encoder top to bottom.
- `DecodedSource` now streams instead of materializing the whole image on
  first pull. It advances a forward-only decoder to the requested band and
  retains just that band as a rolling window, which is what makes constant
  memory real rather than promised. A request behind the window is a reported
  error, never a silent rewind.
- `Flip` now declares `AccessPattern::Sequential`, not `Spatial`.
  `AccessPattern` describes tile *shape*; a vertical mirror reads one input
  row per output row and wants full-width strips. Its row reversal is tile
  *order*, which `input_regions` already expresses and the scheduler resolves
  at the seam (ADR-0009).
- SPEC §Guarantees 1 now states the constant-memory condition as "where the
  format *and pipeline order* allow", naming the reverse-order-over-sequential
  -source exception rather than leaving the guarantee quietly overstated.

### Fixed
- The AVIF reconstruction decoded transform blocks whose top-left corner lies
  past the frame's right or bottom edge, which a partial superblock produces.
  The spec codes no such block, so the extra symbol reads desynchronised
  everything after them — surfacing as wrong chroma in the last region of a
  frame with odd dimensions while the luma before it stayed correct. The
  transform-block loop now skips them, and `gradient_odd_lossless` (37x29)
  decodes bit-exact.
- The AV1 CDF table generator dropped the multiplication in spec entries of the
  form `{ 128 * 125, 32768, 0 }`, splitting `128 * 125` into two separate values
  and lengthening every affected binary CDF by one bogus entry. The tokenizer
  now evaluates the product. This corrupted `dc_sign` and a dozen other binary
  CDFs, which silently desynchronised coefficient decoding — the kind of bug the
  generated-and-diffed approach exists to make visible, once something actually
  read the tables.

### Added
- `otf-pixels-codec-avif` gains the intra transform-type decode: `get_tx_set`
  (which of `DCT_DCT`-only, the seven-type set 1, or the five-type set 2 a
  transform of a given size uses), the `intra_tx_type` symbol read with its
  size-and-direction CDFs mapped back through the inversion tables, and the
  chroma derivation (`Mode_To_Txfm` gated by set membership, `compute_tx_type`).
  A large or lossless transform stays `DCT_DCT` without consuming a symbol, so
  the lossless path is unaffected. Exposed as a tested unit on the crate's AV1
  surface ahead of wiring it into the tile coefficient path.
- `otf-pixels-codec-avif` generalizes the coefficient decoder to every transform
  size and type, not just the 4x4 lossless corner. `decode_coeffs` now selects
  the `eob_pt` alphabet by size, walks the size-and-type-specific scan order
  (`get_scan` over the generated `Default`/`Mrow`/`Mcol` scan tables), and
  derives the `coeff_base`/`coeff_br` contexts from the coded block's `bwl` and
  transform class (2D vs. the 1D row/column identities). The scan tables are
  generated from a vendored spec extract by `scripts/generate-av1-scan-tables.py`
  and validated to be exact permutations, with CI re-running the generator and
  diffing — the same discipline as the CDF tables. The lossless path is the
  `TX_4X4`/`DCT_DCT` case of the general decoder and stays bit-exact against
  libavif on every fixture.
- `otf-pixels-codec-avif` gains the full lossy inverse-transform machinery, the
  foundation of the lossy decode path. `inverse_transform_2d` runs the §7.13
  butterfly network — the inverse DCT (4..64), ADST (4/8/16) and identity
  transforms, with the flip and rectangular-scaling cases — over any of the 19
  transform sizes, and `dequantize` applies the real `Dc_Qlookup`/`Ac_Qlookup`
  quantiser steps (generated from the spec). The lossless reconstruct now flows
  through the same pair — qindex 0 is just its `dc == ac == 4`, `dqDenom == 1`
  corner where the Walsh–Hadamard transform divides the dequantiser back out —
  so every lossless fixture stays bit-exact against libavif while the lossy
  transforms come online.
- `otf-pixels-codec-avif` decodes the screen-content lossless path: palette and
  chroma-from-luma prediction. Palette blocks decode their colour set (merging
  the neighbour palettes into the prediction cache, then cache hits, a base
  colour, and clipped deltas) and the per-sample colour-index map (the wavefront
  scan with its colour-order context model), and predict each sample as the
  colour its index selects. Chroma-from-luma reads its signed alphas and adds the
  scaled, DC-removed reconstructed luma to the DC chroma prediction. With these,
  every lossless AVIF fixture — including the block/screen-content one — decodes
  bit-exact against libavif; only intra block copy remains unimplemented in the
  lossless path.
- `otf-pixels-codec-avif` decodes real pixels end to end: `AvifDecoder::read_row`
  reconstructs a lossless 4:4:4 AVIF still, converts the identity-matrix planes
  to RGB, and serves the rows — bit-exact against libavif on every lossless
  fixture (`tests/reference.rs`). The tile driver walks the partition tree
  (including the T-shaped and 4-way splits), reads each block's intra mode, and
  for every 4x4 transform block predicts, decodes coefficients, inverts the
  Walsh–Hadamard transform, and writes the samples back — threading the
  neighbour level, DC-sign, mode, and block-decoded context arrays that keep the
  entropy contexts aligned with the encoder. All the intra modes a natural image
  reaches are implemented: DC, Paeth, the smooth family, the slanted directional
  modes with their edge-filter and upsample machinery, and recursive
  filter-intra. Intra block copy, subsampled chroma, lossy transforms, and grids
  report `Unsupported` rather than decode wrong.
- `otf-pixels-codec-avif` gains the 4x4 coefficient decoder — the syntax that
  turns a transform block's entropy-coded symbols into a signed level array.
  It reads `all_zero`, the end-of-block position, each coefficient's base level
  walking the scan backwards, the `coeff_br` and Golomb magnitude extension for
  large levels, and the signs walking forwards, each against its
  context-selected CDF. The context derivations (`get_coeff_base_ctx`, the
  `coeff_br` and end-of-block buckets) are transcribed from the spec and unit
  tested against its worked contexts. Scoped to `TX_4X4`, which is every block
  the lossless path codes; the wider transform sizes arrive with the lossy path.
- `otf-pixels-codec-avif` gains 4x4 intra prediction for the non-directional
  modes. DC, Paeth, and the three Smooth variants are complete, as are the two
  axis-aligned directional modes (V and H, which at exactly 90 and 180 degrees
  are plain copies with no edge filtering). The predictors are pure functions
  over the assembled `AboveRow`/`LeftCol` neighbour arrays, so they are tested
  directly against the spec's formulas; building those arrays from the plane is
  the tile driver's job. The slanted directional modes, which need the
  edge-filter and upsample machinery, report `Unsupported` until that lands.
- `otf-pixels-codec-avif` gains the first reconstruction primitives: a bounds-safe
  sample `Plane` (edge-replicating clamped reads for intra prediction, the
  workspace's slice-indexing ban concentrated in one file) and the lossless 4x4
  inverse transform. The transform is the Walsh–Hadamard pair the lossless path
  uses: dequantize each level by the qindex-0 step of 4, run the row WHT whose
  `shift = 2` divides that step straight back out, then the column WHT — which
  is why a lossless decode is bit-exact. The 4x4 blocks are handled by array
  destructuring rather than indexing, keeping the hot path within the lint.
- `otf-pixels-codec-avif` gains its reference-fixture harness ahead of pixel
  reconstruction, because our own decoder cannot validate itself: a shared
  misreading of the spec round-trips perfectly and is still wrong.
  `scripts/regenerate-avif-reference.py` encodes fixtures with libavif's
  `avifenc` and takes the expected rasters from libavif's `avifdec` — an
  independent AV1 implementation. The first fixtures are lossless, the strictest
  target there is: lossless AVIF is `CodedLossless`, so every post-filter is off
  and only the 4x4 Walsh-Hadamard transform is used, and the raster must equal
  the source exactly. CI installs libavif, re-runs the generator and diffs the
  manifest, and the lossy fixtures with their tolerances join as the DCT/ADST
  paths and post-filters land.
- `otf-pixels-codec-avif` gains the AV1 default CDF tables, generated rather than
  hand-transcribed. `scripts/generate-av1-cdf-tables.py` emits `av1/cdf.rs` from
  a vendored extract of the specification's section-10 default-CDF blocks
  (`scripts/data/av1-default-cdf-tables.txt`), and CI re-runs the generator and
  `git diff --exit-code`s the result, exactly as it does the REFERENCE
  manifests — so a stale table or a hand edit is caught. All 96 tables land in
  the exact shape `SymbolDecoder::read_symbol` consumes: cumulative frequencies
  ending in `1 << 15`, then a `0` adaptation counter.
- `otf-pixels-codec-avif` gains the AV1 multi-symbol arithmetic decoder — the
  entropy engine every tile symbol passes through. It is a range decoder over
  15-bit inverse CDFs with a per-symbol floor probability, plus the CDF
  adaptation that lets the distributions track the stream. The transcription
  follows the AV1 specification's symbol-decoding process verbatim and was
  cross-checked against libaom's `entdec.c`; the two agree on the inverse-CDF
  arithmetic and the renormalization. Reads are fallible and the `SymbolMaxBits`
  accounting means the implicit zero-padding past the real bytes is entered
  deliberately rather than by reading off the end. Numeric conformance against
  real streams is deferred to the libaom differential harness, the only
  trustworthy check for an entropy coder, which lands with reconstruction.
- `otf-pixels-codec-avif` gains the AV1 front end: the bitstream layers that
  read a still picture's headers down to — and stopping at — the tile-group
  data. A most-significant-bit-first `BitReader` with the AV1 primitives
  (`f`, `uvlc`, `leb128`, `le`, `su`, `ns`, byte alignment), the OBU framing
  layer, the full sequence header (`color_config` and all), and the frame
  (uncompressed) header restricted to the intra key-frame path — size,
  quantizer, segmentation, loop-filter/CDEF/restoration parameters, transform
  mode, tile layout and film-grain parameters, each sub-block walked in order
  so `CodedLossless` is known before the filter parameters that depend on it.
  A `StillPicture` driver ties configuration OBUs and the coded frame together
  the way an AVIF still is laid out and returns both parsed headers plus a
  locator for the tile data. Every primitive is fallible: a bitstream that ends
  mid-value, an OBU size past the buffer, or a non-zero alignment pad is a
  returned error, never a panic. The front end is exposed as public API and
  fully tested against hand-assembled streams; it is wired into pixel
  reconstruction in the phase that follows. Still no pixels: `read_row` still
  reports `Unsupported`.
- `otf-pixels-codec-avif`: the first slice of an AVIF codec owned from scratch,
  container and (eventually) AV1 bitstream both — ADR-0013, reversing ADR-0004's
  decision to wrap the dav1d/rav1e family, because the wrapped route pulled a
  non-Rust build dependency the rest of the workspace does not have. This
  release lands the ISOBMFF/HEIF container: the box reader (64-bit sizes and
  parent-bound checks that reject the classic infinite-loop and
  run-past-the-end malformations), the item model (`meta`/`pitm`/`iinf`/`iloc`/
  `iref`/`idat`), and item properties (`iprp`/`ipco`/`ipma` with
  `ispe`/`av1C`/`pixi`/`colr`/`irot`/`imir`/`auxC`). `AvifDecoder` reports
  correct `metadata()` — dimensions, pixel format, alpha, grid — for any AVIF,
  and `AvifCodec`/`probe` sniff the file by its `ftyp` brand. Wired into the
  facade behind an `avif` feature, on by default. Pixel decode currently reports
  `Unsupported`: the AV1 bitstream reconstruction lands in the phases that
  follow. Still images only; image sequences (the `avis` brand) are out of scope
  for v1.
- `otf-pixels-codec-webp`: WebP decode (lossy and lossless, alpha detected
  from the container) and lossless encode, wrapping `image-webp` per ADR-0004.
  Wired into the facade behind a `webp` feature, on by default. Pure Rust; its
  only transitive dependencies are `byteorder-lite` and `quick-error`.
- **WebP encoding is lossless only**, because the wrapped encoder has no
  quality control at all — `EncodeOptions::quality` is accepted and ignored.
  For a photograph that means a much larger file than the format's headline
  feature would give, and it is recorded here rather than left to be
  discovered from a file size. It is the strongest argument for revisiting
  WebP ownership.
- Greyscale has no native WebP mode, so one channel in comes back as three.
  Asserted rather than glossed, since it differs from PNG and JPEG.
- libwebp reads all 24 emitted WebP files back to the exact pixels put in
  (`scripts/check-webp-interop.sh`). Exact, not tolerant: the encoder is
  lossless, so there is nowhere for a bug to hide.
- Progressive JPEG decode, wrapped rather than owned (ADR-0004), behind the
  `jpeg-progressive` feature of `otf-pixels` — on by default. `jpeg-decoder`
  with default features off is the only external dependency in the default
  build and brings no transitive ones; `default-features = false` drops it and
  progressive files then report `Unsupported` naming the feature.
- The handover is the part worth testing, and is tested: our header parser
  consumes bytes from a forward-only stream before it learns the frame is
  progressive, so the reader records what it has read and replays it to the
  wrapped decoder. A short, long or misordered replay would produce a wrong
  picture or a rejected file, so both progressive fixtures are compared
  against libjpeg to within 4 per sample. The tape is dropped the moment a
  baseline frame is confirmed, so an ordinary decode buffers nothing.
- `JpegDecoder::is_progressive` reports which decoder ran. Progressive frames
  are internally buffered (SPEC §Formats) and offer no reduced-scale decode,
  so shrink-on-load correctly declines them rather than promising a fast path
  the wrapped decoder cannot provide.
- Shrink-on-load: `shrink_on_load` rewrites a graph over a reduced source when
  the whole pipeline permits it, so a JPEG thumbnail decodes at 1/8 rather
  than at full size and then discards the pixels. The decision needs the
  complete graph — `from_stream` runs before `.resize(200, 150)` is ever
  called — so it is a rewrite pass rather than a decoder option, and it runs
  above the choice of evaluator so the scheduler and the reference evaluator
  keep agreeing.
- `Op::rescaled` replaces the pair of decisions an op would otherwise have to
  get right separately: whether it means the same thing against a smaller
  input, *and* whether the instance it hands back carries state bound to the
  old one. `resize` memoizes filter tables keyed to the shape it first saw, so
  reusing the instance was a resample against the wrong scale waiting to
  happen. `crop`, `convolve` and `composite` decline outright — coordinates
  and kernel sizes are in input pixels.
- `Output::write_with_stats` returns `RunStats`, including the reduction
  shrink-on-load applied, so a pipeline that expected the fast path and did
  not get it is diagnosable rather than merely slow.
- `otf-pixels-codec-jpeg`: `JpegDecoder`, baseline JPEG from scratch — Huffman
  entropy decode, a fixed-point IDCT, every chroma subsampling, restart
  intervals, greyscale and RGB-labelled files, and EXIF orientation read (but
  not applied: `auto_orient` is a pipeline decision, and a decoder that
  rotated its own output would leave no way to turn it off). Decode is
  streaming at one MCU row, so peak memory is a band and not the image.
  Progressive, arithmetic-coded, 12-bit and CMYK files are reported
  `Unsupported` rather than `Malformed`: they are valid JPEGs this codec does
  not own, and a host binding routes on that difference.
- `otf-pixels-codec-jpeg` fixtures are compared against libjpeg-turbo with a
  tolerance rather than a hash, because JPEG defines the IDCT only to an
  accuracy bound and a hash would fail a conforming decoder. Chroma is
  upsampled nearest-neighbour where libjpeg interpolates, so subsampled
  fixtures are compared across their flat interiors, where the two filters
  must agree, and left alone at chroma edges, where they legitimately differ.
- `otf-pixels-codec-jpeg`: `JpegEncoder`, writing baseline JPEG with a
  fixed-point forward DCT, the Annex K quantization and Huffman tables scaled
  by the IJG quality mapping, and 4:4:4/4:2:2/4:2:0 chroma subsampling
  (4:4:4 from quality 90 up, where subsampling rather than quantization would
  otherwise become the dominant loss). Encoding streams at one MCU row, so
  bytes reach the sink before the last row arrives. Alpha is composited
  against black rather than dropped, as the GIF encoder already does.
  Optimal Huffman tables are deliberately not derived: that needs a counting
  pass over every coefficient before the first byte can be written, which
  trades ADR-0005's streaming contract for a few percent.
- libjpeg reads all 140 emitted JPEGs — seven sizes, three subsamplings, five
  qualities, greyscale and RGB — and decodes them to within 1.6x of the loss
  libjpeg's *own* encoder produces on the same input
  (`scripts/check-jpeg-interop.sh`). The comparison is against the reference
  encoder rather than a fixed tolerance because a fixed one is wrong at both
  ends: a steep gradient in a 7x3 image loses far more to 4:2:0 than the same
  gradient across 64x48 does.
- `otf-pixels-codec-jpeg`: `JpegDecoder::with_scale`, decoding at 1/8, 1/4 or
  1/2 resolution straight from the coefficients (SPEC §Core ops, "JPEG fast
  path"). `Scale::fitting` picks the coarsest scale still at least the target
  size, never below it. The win is downstream: at 1/8 every later op sees one
  sixty-fourth of the pixels, and the full-resolution image is never
  materialized.
- The reduced transform is defined as the **exact box average** of the full
  one — each basis entry sums the cosines of the samples that output replaces,
  so all sixty-four coefficients contribute. The first implementation instead
  inverse-transformed the top-left `MxM` corner, which is the obvious reading
  and is measurably wrong: against libjpeg's scaled decode on a noise fixture
  at 1/2, truncation was ten times further from the true downsample (15.4 mean
  error against 0.57). With the box basis we match libjpeg to three decimal
  places on every fixture that has no chroma subsampling.
- `otf-pixels`: JPEG wired into sniffing, decode and encode behind a `jpeg`
  feature, on by default. `Image::from_stream` recognises a JPEG by its magic
  bytes and `output(Format::Jpeg, options)` writes one, so a decode/resize/
  re-encode pipeline works end to end.
- A `jpeg_decode` fuzz target and an in-tree mutation harness, both asserting
  only that no input panics, plus a `jpeg_roundtrip` target asserting that
  every stream this encoder produces is one this decoder accepts at the
  declared shape.
- M5 exit-criterion tests and `benches/thumbnail.rs`, the giant-tiled-TIFF
  thumbnail benchmark against libvips. It skips cleanly when `vips` is not
  installed rather than omitting the row or inventing a number.
- ADR-0012 (extract `otf-pixels-compress`), executing the "move on a third
  consumer" clause ADR-0010 wrote for exactly this moment.
- `otf-pixels-codec-tiff`: `TiffDecoder` covering baseline TIFF 6.0 — both
  byte orders, strip and tile layouts, none/LZW/Deflate/PackBits compression,
  greyscale at 1/8/16 bits, RGB, and palette. Exotic tags are skipped, not
  errors. A **tiled** file reports `DecodeCapability::Regions`, so producing a
  region decompresses only the tiles it touches; a strip file reports
  `Sequential`, because claiming otherwise would be a lie the scheduler acts on.
- `otf-pixels-codec-tiff`: `TiffEncoder`, writing baseline TIFF in strips or
  tiles, uncompressed or Deflate. libtiff reads all 240 emitted files — six
  sizes, five formats, four layouts, two compressions — back to the pixels we
  put in (`scripts/check-tiff-interop.sh`). Tiled output is what lets a
  pipeline store an intermediate it will re-read with random access.
- `otf-pixels-core`: `DecodedSource` now asks a region-capable decoder for
  regions directly instead of driving it row by row. Without that a tiled TIFF
  was declared random-access and then read sequentially anyway.
- `otf-pixels`: GIF and TIFF wired into sniffing, decode and (for GIF) encode,
  behind `gif` and `tiff` features.
- `otf-pixels-codec-gif`: `GifDecoder` covering the whole format — all frames,
  both interlace layouts, transparency and every disposal method — and
  `GifEncoder`, single-frame with median-cut quantization and Floyd-Steinberg
  dithering, which is SPEC §Formats' stated v1 scope. `Decoder` yields the
  first frame so ordinary pipelines work unchanged; `next_frame` walks the
  rest, keeping "animation pipelines are v2" honest.
- `otf-pixels-compress`: inflate, deflate, zlib, `Crc32` and `Adler32` moved
  from `otf-pixels-codec-png` (which re-exports them, so nothing downstream
  changes), plus LZW in both the GIF and TIFF dialects — one implementation
  parameterized by bit order, since the two specifications differ in framing
  and not in substance.
- ADR-0011 (SIMD by autovectorization; fixed-point arithmetic at 8 bits).
- `otf-pixels`: `resize`, `thumbnail`, `rotate`, `modulate`, `convolve`,
  `blur`, `sharpen`, `extract_channel`, `flatten` and `composite` on the
  chainable facade. `composite` joins two lazy branches of a graph.
- `benches/ops.rs`: the M4 comparative benchmark against `image` and
  `fast_image_resize`, both dev-dependencies only. Numbers published in the
  README, including the part that does not flatter us.
- M4 exit-criterion tests: every pipeline is byte-identical run to run, across
  thread counts, across tile shapes, and against the M1 oracle.
- `otf-pixels-ops`: the rest of the v1 op set (SPEC §Core ops) — `Rotate`
  (quarter turns), `Modulate` (brightness/saturation/hue), `Convolve` with
  blur, Gaussian and sharpen presets, `Composite` (Porter-Duff source-over,
  the first two-input op), `ExtractChannel` and `Flatten`. Every op with a
  non-trivial demand mapping asserts that its output does not depend on how
  the image is tiled.
- `otf-pixels-ops`: `Resize`, separable resampling with seven filters
  (nearest, box, bilinear, Catmull-Rom, Mitchell, Lanczos2, Lanczos3), `Fit`
  modes and `without_enlargement`. Weight tables are built per *image*, not
  per tile, so the output does not depend on how the image is cut up — which
  is asserted directly rather than assumed.
- ADR-0008 (crossbeam-deque for work stealing), ADR-0009 (scheduler-inserted
  materialization at order-incompatible seams) and ADR-0010 (own inflate and
  deflate).
- `otf-pixels-codec-png`: `Crc32`, `Adler32`, and a from-scratch DEFLATE
  decompressor (`inflate_to`, `zlib_decompress`) per ADR-0010. Bounded output
  makes decompression bombs a malformed-input error rather than an allocation.
- `otf-pixels-codec-png`: a from-scratch DEFLATE compressor (`deflate`,
  `zlib_compress`, `Level`) with levels 0-9. Verified in both directions
  against reference zlib — we decode its streams, and it decodes ours
  (`scripts/check-deflate-interop.sh`).
- `otf-pixels-codec-png`: `PngDecoder`, covering every v1 PNG feature — bit
  depths 1/2/4/8/16, colour types 0/2/3/4/6, `PLTE`, `tRNS`, Adam7 interlace
  and all five filter types. Ancillary chunks are skipped, not honoured.
  Non-interlaced images **stream**: peak memory is two scanlines, the 32 KiB
  inflate window and a read buffer, none of which grow with image height.
  Interlaced images buffer, exactly as SPEC §Formats already said.
- `otf-pixels-codec-png`: `Inflater` and `ZlibStream`, the incremental
  decompressors the streaming decoder is built on, plus `ChunkStream`, which
  walks PNG chunks from a forward-only source a piece at a time.
- CI (`.github/workflows/ci.yml`): test, fmt, clippy, docs, feature
  combinations, MSRV, reference interop and fuzzing. The interop job also
  regenerates the PngSuite manifest and fails on a diff, so a stale reference
  cannot quietly start agreeing with a broken decoder.
- `fuzz/`: `cargo fuzz` targets for PNG decode, inflate and encode/decode
  round-trip, plus `tests/fuzz.rs`, an in-tree deterministic mutation harness
  that runs the same no-panic property on stable in seconds.
- M3 exit-criterion tests: PNG round-trips through the engine for every
  format it can represent, ops compose over a decoded PNG exactly as over raw
  pixels, the scheduler agrees with the M1 oracle over a PNG source, and
  decoding a tall PNG reads a fraction of the file before the first row.
- `otf-pixels`: `Image::open` and `Image::from_stream`, which identify a format
  from its magic bytes and never from a file extension (SPEC §Formats).
  `Format::Png` now resolves to a real encoder behind the `png` feature.
- `otf-pixels-core`: `Prefixed`, a source that replays a buffered prefix before
  delegating. It is what lets sniffing look at the magic bytes and still hand
  the whole stream onward, without asking any source to seek (ADR-0005).
- `otf-pixels-codec-png`: `PngCodec`, the sniffing registry entry for PNG.
- `otf-pixels-codec-png`: `PngEncoder`, writing non-interlaced PNG at DEFLATE
  levels 0-9 with per-row adaptive filter selection. Verified in the encode
  direction too: libpng reads all 140 emitted files — seven pixel formats,
  five sizes, four levels — back to the pixels we put in
  (`scripts/check-png-interop.sh`). `EncodeOptions::quality` is read as
  compression effort, since PNG is lossless and has no fidelity to trade.
- `otf-pixels-codec-png`: the PngSuite conformance corpus (100 files, ~62 KB,
  vendored with its licence) checked against reference decodings from libpng
  rather than against ourselves. All 14 corruption modes are rejected, and no
  truncation or single-byte mutation of any fixture panics.
- `otf-pixels-core`: `TileCache`, a byte-budgeted LRU of graph intermediates
  keyed by `(NodeId, Region)`. Eviction bounds what the cache *retains*, never
  what a caller holds alive, so tiles need no pinning.
- `otf-pixels-core`: `ThreadPool`, a work-stealing pool over `crossbeam-deque`
  (ADR-0008). Panicking tasks are contained and reported as errors; a batch
  reports its lowest-indexed failure so errors stay deterministic.
- `otf-pixels-core`: `Plan`, the pre-execution graph analysis. Negotiates tile
  shapes per segment (ADR-0003) and marks materialization points where
  non-forward demand meets a forward-only source (ADR-0009). Pure analysis —
  it reads no pixels.
- `Producer::capability`, the upstream half of ADR-0009's seam analysis.
  `BufferSource` reports `Regions`; `DecodedSource` delegates to its decoder.
- `otf-pixels-core`: `Scheduler`, the demand-driven parallel tile evaluator,
  plus `evaluate_tiled` and `RunStats`. Output tiles are evaluated in parallel
  batches and delivered to the sink in order.
- `NodePlan::cacheable`: only nodes demanded more than once (a shared prefix,
  or one feeding a spatial op) are retained in the tile cache.
- `Output::bytes_via_reference`, running a pipeline through the M1 evaluator so
  the scheduler can be differentially tested against it.
- `Output::threads` / `Output::scheduler_options` for tuning a run.
- Builder setters on `Limits`, `PlanOptions` and `SchedulerOptions`, which are
  `#[non_exhaustive]` and were otherwise unconfigurable downstream.
- M2 exit-criterion tests and `benches/scaling.rs`, a std-only benchmark
  reporting speedup and parallel efficiency across thread counts.
- Project documentation: README, ARCHITECTURE, SPEC, ROADMAP, ADR-0001..0007.
- Cargo workspace: `otf-pixels`, `otf-pixels-core`, `otf-pixels-ops`,
  `otf-pixels-codec-raw` (ADR-0006). No external dependencies.
- `otf-pixels-core`: `PixelsError` with stable `ErrorCode`s, `PixelFormat` and
  the `Sample` dispatch seam (ADR-0002), `Region`/`ImageDescriptor`/`Limits`
  with `max_pixels` enforced before allocation, strided `Tile`/`TileMut`/
  `TileBuf` views, streaming `Source`/`Sink` traits (ADR-0005), `Decoder`/
  `Encoder`/`Codec` traits, the `Op` trait and lazy `Image` op graph, and the
  naive whole-image evaluator that M2 will be diffed against.
- `otf-pixels-codec-raw`: `RawDecoder`/`RawEncoder` with caller-supplied
  dimensions, pixel format and stride, streaming a row at a time in both
  directions. Truncated streams are malformed-input errors, never panics.
- `otf-pixels-ops`: `Crop`, `Flip` and `Flop` geometry ops, each declaring its
  demand mapping and access pattern for M2's scheduler.
- `otf-pixels`: the chainable facade — `Image::from_raw`, `from_raw_stream`,
  `crop`/`flip`/`flop`, and the `output(format, options)` terminal with
  `write(sink)` and `bytes()`. Errors raised mid-chain are captured and
  surfaced at the terminal, so pipelines read as one expression.
- M1 exit-criterion test suite: raw → crop/flip → raw round-trips, graph
  laziness (zero source bytes read before a terminal), malformed-input and
  limit handling, determinism, and concurrent evaluation.
