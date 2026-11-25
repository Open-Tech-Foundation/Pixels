# ADR-0020: Animation as a per-frame replay of one graph

Status: Accepted · 2026-11-25

## Context
ROADMAP v2 item 5: multi-frame pipelines. Today GIF decodes every frame
(`GifDecoder::next_frame`) but hands the pipeline the first; WebP and JPEG
XL read every frame's timing but decode only the first; every encoder
writes one frame; and `OpenOptions::animated` is reserved and refused.

libvips, and so sharp, loads an animation as one tall image, the frames
stacked, with a `page-height`. That makes every op responsible for frame
boundaries: a blur must not bleed from one frame into the next, a resize
must not mix them, a rotation must turn each in place. It also holds every
frame at once, against ADR-0005's streaming model.

## Decision
1. **One graph, replayed per frame.** The ops a caller chains are built
   once, against the canvas size, and the terminal runs them once per
   frame: the decoder composites frame *k* onto its canvas, the graph is
   re-rooted on it (the rebuild that shrink-on-load already does), and the
   rows go to an animated encoder. Every frame has the canvas's size, so
   ops bound to their input size stay valid. No op learns about frames.
2. **Core traits grow default methods, nothing breaks.**
   `Decoder::advance_frame() -> Result<Option<FrameInfo>>` moves the row
   cursor to the next composited frame (index and duration in ms);
   `Encoder::set_animation(loop_count)`, `begin_frame(duration_ms)` and
   `end_frame()` bracket each frame's rows. A codec that does not
   implement them is a still codec, as now. `Limits` gains `max_frames`
   and a cap on frames × canvas pixels, so a file of a hundred thousand
   tiny frames on a large canvas is refused before it is processed.
3. **Formats: GIF, WebP and JPEG XL**, read and written. Decoders
   composite exactly as libwebp's `anim_decode`, GIF viewers (disposal)
   and libjxl (blending) do. APNG is not read: libvips and sharp do not
   read it either, and little produces it. AVIF sequences stay refused:
   decoding them needs AV1 inter coding, a ROADMAP non-goal.
4. **No silent flattening.** An animated pipeline written to a format
   that holds one image (PNG, JPEG, TIFF, AVIF, raw) is `Unsupported`
   unless the caller said what they want: `Image::first_frame()` /
   `Image::frame(i)` for one frame, or `Image::into_strip()` for every
   frame stacked top to bottom (sharp's default). Both turn the pipeline
   into a still, so ops after them and every output work as usual.
   Opening without `animated` still gives the first frame, as sharp's
   default and today's.
5. **Whole-animation ops see every frame.** `trim` cuts the box that
   holds the content of every frame; `stats` aggregates every frame. Both
   decode the animation an extra time, which is accepted: features before
   speed.
6. **Encoders follow sharp's options.** GIF: per-frame palettes, the
   previous one reused within `interPaletteMaxError` (default 3) and
   `reuse`; frames reduced to the changed rectangle, unchanged pixels
   transparent within `interFrameMaxError`; delays in centiseconds with
   the rounding error carried forward. WebP: `ANIM`/`ANMF`, changed
   sub-rectangles at even offsets, per-frame blend and dispose, keyframe
   spacing (`kmin`/`kmax`), `minSize`, `mixed`. JPEG XL: an animation
   header and one frame per input frame. `EncodeOptions` gains sharp's
   `loop` and `delay` overrides.

## Consequences
+ Memory is one frame in flight plus the decoder's canvas (and a saved
  rectangle for GIF's "restore previous"), not the whole animation.
+ Every existing op works on animations with no change and no
  frame-boundary bugs to write.
+ Tiles within a frame still run in parallel; overlapping one frame's
  decode with the previous frame's processing is left for later.
- Frames are processed in order, one after another: an animation costs
  its frame count times a still of its canvas, plus `trim`/`stats` passes.
- Unlike sharp, writing an animation to PNG or JPEG without
  `first_frame` or `into_strip` is an error, not a stacked strip; the
  sharp table in `EMBEDDING.md` records the deviation.
- An overlay given to `composite` is a still drawn on every frame;
  animated overlays are not covered.
