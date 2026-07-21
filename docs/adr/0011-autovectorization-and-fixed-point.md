# ADR-0011: SIMD by autovectorization; fixed-point arithmetic at 8 bits

Status: Accepted · 2026-07-21

## Context
ROADMAP M4 requires SIMD for resize and the pointwise ops. Two commitments
already on the books constrain how:

- `unsafe_code = "forbid"` workspace-wide. `forbid` cannot be relaxed by an
  `#[allow]` at a call site — that is precisely what distinguishes it from
  `deny` — and every `core::arch` intrinsic is `unsafe`. Reaching intrinsics
  therefore means editing the lint, not annotating a function.
- SPEC §Guarantees 2: byte-identical output on every platform and backend,
  with scalar and SIMD paths agreeing **exactly**, CI-enforced.

`std::simd` would give portable vectors with no `unsafe`, but it is nightly
-only against a stable 1.85 MSRV. Gating the fast path behind nightly would
mean nobody on stable — which is everybody — ever runs it, while we maintain
two paths that must stay bit-identical.

The exact-equality guarantee is the sharper constraint, and it interacts with
the arithmetic representation. A SIMD kernel that reduces across lanes sums
its terms in a different order from the scalar loop. In floating point that
changes the result, so "scalar and SIMD agree exactly" becomes a property CI
must catch rather than one the design cannot violate. In integer arithmetic
addition is associative, so any vectorization order gives identical bits.

## Decision
**SIMD is reached by autovectorization, not intrinsics.** `unsafe_code =
"forbid"` stays in force in every crate, unchanged. Kernels are written to be
vectorizable — fixed-length inner loops over lane-count chunks, no early
exits, slice-based access with the bounds check hoisted out of the hot path,
accumulators that do not alias — and correctness never depends on whether the
compiler took the hint.

**Eight-bit paths use i32 fixed-point; 16-bit and float paths use f32.** Filter
coefficients are quantized once, per resize, into `i32` with a fixed fractional
scale; products accumulate in `i32` and are rounded and clamped on the way out.
This is what libvips and `fast_image_resize` both do, for the same reasons.

Consequences for how kernels are written, which is the part that matters:

- Vectorize across **output pixels**, never by reducing across lanes. Each lane
  performs the same operations in the same order as the scalar loop, so
  agreement is structural rather than incidental.
- The scalar loop is the definition of correct. The vectorizable form is the
  only form; there is no separate "scalar fallback" to drift out of sync,
  which is why the exact-equality gate is cheap to keep true.

## Consequences
What we give up: the last increment of throughput. Hand-written AVX2 typically
beats well-shaped autovectorized code by 10–30% on this class of work — pointwise
transforms and separable resize, both of which vectorize readily. That is real,
and it is the price of the safety property being unqualified rather than
"forbidden where it matters".

We also accept a dependency on the optimizer that a benchmark, not a type
system, has to police. A compiler upgrade can silently stop vectorizing a loop.
The mitigation is that `benches/` reports throughput per op, so a regression
shows up as a number rather than as a mystery — but nothing *fails* when
vectorization is lost, and that asymmetry is worth naming.

Fixing the 8-bit representation as fixed-point is a **permanent** commitment,
not an implementation detail: SPEC §Guarantees 2 promises byte-identical output
across versions, so changing the arithmetic later changes pixels and breaks that
promise. Sixteen-bit and float paths carry f32's ordering discipline instead,
which is the one place the exact-equality gate is doing real work rather than
confirming something the design already guarantees.

If a future milestone finds a kernel where autovectorization is genuinely
inadequate — a transpose, a gather-heavy resample — the answer is a superseding
ADR scoping `unsafe` to that kernel, argued on measurements. Not a quiet
`#[allow]`.
