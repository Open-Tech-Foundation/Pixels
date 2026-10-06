#!/usr/bin/env python3
"""Generate `vp8/tables.rs` from the VP8 reference decoder's constant tables.

The default coefficient probabilities, their update probabilities and the
key-frame subblock-mode probabilities are ~3000 constants. Like the AV1
tables, hand-transcribing them is a bug source with no upside, so they are
generated, and CI re-runs this and ``git diff --exit-code``s the result.

Input is a vendored extract of the C tables in the reference decoder source
attached to RFC 6386 (section 20), kept at ``scripts/data/vp8-tables.txt``.
Each table's entry count is checked against its declared shape, every
probability against 1..=255, and both quantizer lookups against being
non-decreasing, before anything is emitted.

Usage::

    python3 scripts/generate-vp8-tables.py

Writes ``crates/otf-pixels-codec-webp/src/vp8/tables.rs``.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "scripts" / "data" / "vp8-tables.txt"
OUT = ROOT / "crates" / "otf-pixels-codec-webp" / "src" / "vp8" / "tables.rs"

# name -> (Rust name, element type, shape, doc)
TABLES = {
    "dc_q_lookup": ("DC_Q_LOOKUP", "u16", [128], "DC quantizer step per index (§14.1)."),
    "ac_q_lookup": ("AC_Q_LOOKUP", "u16", [128], "AC quantizer step per index (§14.1)."),
    "kf_y_mode_probs": ("KF_Y_MODE_PROBS", "u8", [4], "Key-frame luma mode tree probabilities (§11.2)."),
    "kf_uv_mode_probs": ("KF_UV_MODE_PROBS", "u8", [3], "Key-frame chroma mode tree probabilities (§11.4)."),
    "kf_b_mode_probs": (
        "KF_B_MODE_PROBS", "u8", [10, 10, 9],
        "Key-frame subblock mode probabilities by above and left mode (§11.5).",
    ),
    "k_coeff_entropy_update_probs": (
        "COEFF_UPDATE_PROBS", "u8", [4, 8, 3, 11],
        "Probability that each coefficient probability is updated (§13.4).",
    ),
    "k_default_coeff_probs": (
        "DEFAULT_COEFF_PROBS", "u8", [4, 8, 3, 11],
        "Default coefficient token probabilities (§13.5).",
    ),
}

HEADER = """\
// Generated from the VP8 reference decoder's tables (RFC 6386 §20) by
// scripts/generate-vp8-tables.py. Do not edit by hand.
"""


def numbers(body: str) -> list[int]:
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    body = re.sub(r"//[^\n]*", "", body)
    return [int(x) for x in re.findall(r"-?\d+", body)]


def nest(values: list[int], shape: list[int]) -> str:
    if len(shape) == 1:
        return "[" + ", ".join(str(v) for v in values) + "]"
    step = len(values) // shape[0]
    inner = [nest(values[i * step:(i + 1) * step], shape[1:]) for i in range(shape[0])]
    return "[" + ", ".join(inner) + "]"


def main() -> int:
    text = SPEC.read_text()
    out = [HEADER]
    for name, (rust, kind, shape, doc) in TABLES.items():
        match = re.search(rf"\b{name}\b[^=]*=\s*(\{{.*?\}});", text, flags=re.S)
        if not match:
            raise SystemExit(f"{name}: not found")
        values = numbers(match.group(1))
        count = 1
        for dim in shape:
            count *= dim
        if len(values) != count:
            raise SystemExit(f"{name}: expected {count} entries, got {len(values)}")
        if kind == "u8" and not all(1 <= v <= 255 for v in values):
            raise SystemExit(f"{name}: a probability falls outside 1..=255")
        if kind == "u16" and any(b < a for a, b in zip(values, values[1:])):
            raise SystemExit(f"{name}: the lookup is not non-decreasing")
        ty = kind
        for dim in reversed(shape):
            ty = f"[{ty}; {dim}]"
        out.append(f"/// {doc}")
        out.append(f"pub static {rust}: {ty} = {nest(values, shape)};")
        out.append("")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(out))
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
