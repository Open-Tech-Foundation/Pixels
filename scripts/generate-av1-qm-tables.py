#!/usr/bin/env python3
"""Generate `av1/qm_tables.rs` from the AV1 specification's quantizer matrices.

`Quantizer_Matrix` is 100,320 constants (15 levels x 2 plane types x 3344).
Like the default CDFs and scan orders, hand-transcribing it is a bug source
with no upside, so it is generated and CI re-runs this and
``git diff --exit-code``s the result.

Input is a vendored extract of the ``~~~~ c`` blocks defining ``Qm_Offset`` and
``Quantizer_Matrix`` in section 9.5.3 of the AV1 spec, kept at
``scripts/data/av1-quantizer-matrix.txt``. The extract is checked two ways
before anything is emitted: the entry counts must be exact, and every matrix
size must equal what the spec's own derivation process (§9.5.2, informative)
subsamples from the 32x32, 32x16 and 16x32 fundamentals. A dropped or doubled
number fails the second check wherever it lands.

Usage::

    python3 scripts/generate-av1-qm-tables.py

Writes ``crates/otf-pixels-codec-avif/src/av1/qm_tables.rs``.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "scripts" / "data" / "av1-quantizer-matrix.txt"
OUT = ROOT / "crates" / "otf-pixels-codec-avif" / "src" / "av1" / "qm_tables.rs"

LEVELS = 15
PLANE_TYPES = 2
QM_TOTAL_SIZE = 3344

# TX_SIZES_ALL in spec order, as (width, height); the matrix covers at most
# 32x32 of a block (`tw = Min(32, w)`, `th = Min(32, h)`).
TX_SIZES = [
    (4, 4), (8, 8), (16, 16), (32, 32), (64, 64),
    (4, 8), (8, 4), (8, 16), (16, 8), (16, 32), (32, 16),
    (32, 64), (64, 32), (4, 16), (16, 4), (8, 32), (32, 8),
    (16, 64), (64, 16),
]

HEADER = """\
// Generated from the AV1 spec (§9.5.3) quantizer matrix tables by
// scripts/generate-av1-qm-tables.py. Do not edit by hand.
//
// Every size is validated at generation time against the spec's derivation
// from the 32x32, 32x16 and 16x32 fundamental matrices (§9.5.2).
"""


def numbers(text: str) -> list[int]:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    return [int(x) for x in re.findall(r"\d+", text)]


def main() -> int:
    text = SPEC.read_text()
    offsets_text = re.search(r"Qm_Offset\[\s*TX_SIZES_ALL\s*\]\s*=\s*\{([^}]*)\}", text)
    if not offsets_text:
        raise SystemExit("Qm_Offset not found")
    offsets = numbers(offsets_text.group(1))
    if len(offsets) != len(TX_SIZES):
        raise SystemExit(f"Qm_Offset: expected {len(TX_SIZES)} entries, got {len(offsets)}")

    body = text[text.index("Quantizer_Matrix[") :]
    body = body[body.index("=") + 1 :]
    values = numbers(body)
    expected = LEVELS * PLANE_TYPES * QM_TOTAL_SIZE
    if len(values) != expected:
        raise SystemExit(f"Quantizer_Matrix: expected {expected} entries, got {len(values)}")
    if max(values) > 255 or min(values) < 1:
        raise SystemExit("Quantizer_Matrix: an entry falls outside 1..=255")

    matrices = [
        [
            values[(level * PLANE_TYPES + plane) * QM_TOTAL_SIZE :][:QM_TOTAL_SIZE]
            for plane in range(PLANE_TYPES)
        ]
        for level in range(LEVELS)
    ]

    # §9.5.2: each size is subsampled from the fundamental of its shape.
    sizes = {size: offset for size, offset in zip(TX_SIZES, offsets)}
    for level in range(LEVELS):
        for plane in range(PLANE_TYPES):
            table = matrices[level][plane]
            for (width, height), offset in zip(TX_SIZES, offsets):
                w, h = min(32, width), min(32, height)
                if w == h:
                    fw, fh = 32, 32
                elif w > h:
                    fw, fh = 32, 16
                else:
                    fw, fh = 16, 32
                fundamental = sizes[(fw, fh)]
                ratio_w, ratio_h = fw // w, fh // h
                phase_w, phase_h = (ratio_w + 1) // 2 - 1, (ratio_h + 1) // 2 - 1
                for i in range(h):
                    for j in range(w):
                        derived = table[
                            fundamental + (ratio_h * i + phase_h) * fw + ratio_w * j + phase_w
                        ]
                        if table[offset + i * w + j] != derived:
                            raise SystemExit(
                                f"level {level} plane {plane} {width}x{height} [{i}][{j}]: "
                                f"table has {table[offset + i * w + j]}, derivation gives {derived}"
                            )

    lines = [HEADER]
    lines.append("/// `Qm_Offset[TX_SIZES_ALL]`: where each transform size starts in a matrix.")
    lines.append(f"pub static QM_OFFSET: [u16; {len(offsets)}] = {offsets};".replace("[0,", "[0,"))
    lines.append("")
    lines.append("/// `QM_TOTAL_SIZE`: the entries in one level's matrix for one plane type.")
    lines.append(f"pub const QM_TOTAL_SIZE: usize = {QM_TOTAL_SIZE};")
    lines.append("")
    lines.append("/// `Quantizer_Matrix[level][plane > 0][Qm_Offset[txSz] + i * tw + j]`.")
    lines.append(
        f"pub static QUANTIZER_MATRIX: [[[u8; QM_TOTAL_SIZE]; {PLANE_TYPES}]; {LEVELS}] = ["
    )
    for level in matrices:
        lines.append("    [")
        for table in level:
            lines.append("        [")
            for start in range(0, QM_TOTAL_SIZE, 16):
                row = ", ".join(str(v) for v in table[start : start + 16])
                lines.append(f"            {row},")
            lines.append("        ],")
        lines.append("    ],")
    lines.append("];")
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(values)} entries, all sizes derivation-checked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
