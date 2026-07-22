# ADR-0013: Own AVIF, container and AV1 bitstream both

Status: Accepted · 2026-07-22

## Context
ADR-0004 ranked AVIF as "not feasible; AV1 is a decade of multi-team work — even
rav1d is a port", decided to wrap the dav1d/rav1e family behind the
`Decoder`/`Encoder` traits, and recorded "AV1 from scratch is a permanent
non-goal". That was the right call on the information available then.

In practice the wrapped route did not pan out. The candidate crates each carry a
cost this project exists to avoid: dav1d bindings pull a non-Rust build
dependency, which breaks the "pure-Rust, builds anywhere" property the rest of
the workspace holds; rav1d is a line-for-line port whose ergonomics and
error model do not fit the streaming `Decoder` contract without a wrapper as
large as the parts it hides; and rav1e for encode is heavy and hard to steer
toward the small still-picture subset we need. The trait boundary ADR-0004
put in place to make ownership "swappable per format" is exactly what makes this
reversal cheap: nothing downstream of `Decoder`/`Encoder` changes.

The decisive point ADR-0004 could not weigh is that **an AVIF still image is an
AV1 key frame**. That removes inter prediction, reference-frame management,
motion vectors, compound prediction, OBMC, and warped/global motion — well over
half of AV1's decoder surface. The remaining intra-only subset is large but
finite, and it is the same shape of work the owned PNG/JPEG/TIFF codecs already
are. Owning it also buys the one thing wrapping never could: grid region decode
through the container's own tile addressing, which is the region-random-access
contract this engine is built around.

## Decision
Reverse ADR-0004's AVIF clause. Implement AVIF from scratch in a new crate
`otf-pixels-codec-avif`, owning both layers:

- the ISOBMFF/HEIF container (ISO/IEC 23008-12): boxes, the item model
  (`meta`/`iloc`/`iinf`/`iref`/`idat`), and item properties
  (`iprp`/`ipco`/`ipma`, including `ispe`/`av1C`/`pixi`/`colr`/`irot`/`imir`/
  `auxC`); and
- the AV1 bitstream, restricted to the still-picture subset — key frames only.

AV1 lives as an internal `av1/` module tree inside the AVIF crate rather than a
separate `otf-pixels-codec-av1` crate: ADR-0012's rule is that a primitive moves
out on a *third* consumer, and there is exactly one.

Scope held deliberately narrow, each excluded case returning a clean
`Unsupported` rather than a wrong image:

- AVIF image *sequences* (`avis` brand, `moov`/`trak`) are animation and stay
  v2, matching how GIF and WebP decode only their first frame.
- Inter-coded frames of any kind.
- `reduced_descriptor` / `reduce_to`: AV1 has no cheap-thumbnail corner the way
  a DCT block does, so the codec does not claim one.

The reversal is scoped to "own AVIF". ADR-0004's decision to wrap progressive
JPEG is untouched.

## Consequences
+ The default build is pure Rust with no non-Rust build dependency, restoring
  the property AVIF was the sole exception to.
+ Grid AVIFs decode region-by-region: `capability()` reports `Regions` for a
  grid layout and `Sequential` otherwise, the same honesty as tiled TIFF. This
  is the payoff that owning the container buys and wrapping could not.
+ Verification is against the reference implementations (libaom/dav1d/libavif)
  and the Argon conformance streams, not against ourselves — the standing rule
  that our own decoder cannot validate our own encoder holds here most of all.
- This is the largest single codec in the workspace by a wide margin
  (~25–40k lines), and the intra reconstruction path is the most index-dense
  code in the repo — mitigated by concentrating bounds reasoning in one
  reviewed `av1/plane.rs` and generating the CDF tables rather than
  transcribing them.
- Performance will trail dav1d for a long time; ADR-0011 chose autovectorized
  safe Rust over intrinsics, and AV1 reconstruction is far more SIMD-shaped than
  resize. The number gets benchmarked and published whatever it says.
- Encoder output will be materially larger than cavif/rav1e until the quality
  phase, and likely after. A conformant encoder is a much lower bar than a
  competitive one, and that gap is stated rather than hidden.
