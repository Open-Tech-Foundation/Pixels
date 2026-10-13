# ADR-0018: Own progressive JPEG decode

Status: Accepted · 2026-10-13 · Supersedes ADR-0004 for progressive JPEG

## Context
ADR-0004 wrapped progressive JPEG decode in `jpeg-decoder`, the last codec
not written here and the last dependency outside the workspace. The wrapper
decoded the whole file when it was opened, so reading a progressive file's
size cost a full decode, and it had one resolution, so shrink-on-load never
fired. The project now aims to ship no dependencies outside the workspace.

## Decision
Decode progressive frames in `otf-pixels-codec-jpeg`. Opening reads headers up
to the first scan, as for baseline. On the first row request every scan is
decoded into a buffer of quantized coefficients — DC and AC, first and
refinement passes, end-of-band runs and restart intervals (T.81 Annex G) —
and bands are then built from it through the baseline path's inverse DCT,
upsampling and colour conversion. Tables redefined between scans apply to the
scans after them; a component's quantization table is latched at its first
scan, as libjpeg does. At most 500 scans, libjpeg-turbo's recommended limit.

Verification needs no tolerance: libjpeg writes progressive twins of baseline
fixtures, with the same quantized coefficients sent in a different order, and
our decode of each twin must equal our decode of its baseline file byte for
byte at every scale.

## Consequences
+ No dependencies outside the workspace; the `jpeg-progressive` feature is a
  deprecated alias of `jpeg`, kept so existing builds compile.
+ Opening a 12 MP progressive file and reading its metadata: 120 ms to 0.1 ms.
+ Shrink-on-load works for progressive files: a 12 MP thumbnail went from
  160 ms to 102 ms.
- A full-size progressive decode is slower than `jpeg-decoder` (220 ms
  against 132 ms for 12 MP), because it runs our baseline output path, which
  is itself slower than libjpeg-turbo. Speeding up that shared path is the
  next piece of work.
- The coefficient buffer is two bytes per coefficient, up to six bytes per
  pixel at 4:4:4, bounded only by `max_pixels`, as with libjpeg.
