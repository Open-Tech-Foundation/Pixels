# ADR-0010: Own inflate and deflate

Status: Accepted · 2026-07-20

## Context
ADR-0004 commits to owning PNG. But PNG *is* mostly DEFLATE: chunk framing,
filtering and interlacing are the small half, and the compressed stream is the
substance. Depending on `miniz_oxide` would quietly reduce "PNG from scratch"
to "PNG container parsing from scratch", which is not what the README claims.

ADR-0008 pulls in `crossbeam-deque` rather than writing a work-stealing deque,
so a precedent exists for taking general-purpose primitives as dependencies.
The tempting reading of that precedent — "compression is general-purpose, so
take a dependency" — applies the wrong test. ADR-0008 turned on two facts
together: the deque is not part of any codec we committed to owning, **and**
a lock-free deque cannot be written at all under `unsafe_code = "forbid"`.

DEFLATE fails the first condition and passes the second. It is the compression
half of a codec ADR-0004 named, and it is entirely writable in safe Rust — the
decompressor is bounds-checked table lookups and a sliding window, both of
which the borrow checker is happy to police.

## Decision
Implement inflate and deflate from scratch, with no compression dependencies.
They live in `otf-pixels-codec-png` for now and are exported, because TIFF's
deflate compression (M5) uses the same code and GIF's LZW (M5) shares its
shape. If a third consumer appears they move to a shared crate.

The general test this sets, for future decisions: take a dependency when the
thing is **not** substance of a codec we committed to owning, **or** when we
cannot implement it safely. Owning it otherwise.

## Consequences
+ "From scratch" means what the README says it means.
+ One implementation serves PNG, TIFF deflate, and informs GIF's LZW.
+ No compression dependencies at 1.0; nothing to audit but our own code.
+ Memory-safe by construction — the classic decompressor CVEs are
  out-of-bounds writes, which `forbid(unsafe)` makes unrepresentable.
- Substantially more code to get right. PngSuite and fuzzing are the guard,
  and both are M3 exit criteria rather than later additions.
- Our deflate will compress worse and slower than zlib-class implementations
  at first. Ratio and speed are tuning work; correctness is not, and the
  encoder is specified by what a conforming decoder accepts, not by ratio.
- A decompression bug is a wrong-pixels bug rather than a crash, so it can be
  subtle. The PngSuite reference comparisons exist precisely to catch that.
