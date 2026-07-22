#!/usr/bin/env python3
"""Generate `av1/scan_tables.rs` from the AV1 specification's scan tables.

The AV1 coefficient scan orders (`Default_Scan_*`, `Mrow_Scan_*`, `Mcol_Scan_*`)
are ~5000 constants across 32 tables. Like the default CDFs, hand-transcribing
them is a bug source with no upside, so they are generated and CI re-runs this
and ``git diff --exit-code``s the result.

Input is a vendored extract of the ``~~~~ c`` blocks that define the scan
tables in section 9.3 of the AV1 spec, kept at
``scripts/data/av1-scan-tables.txt`` so generation is reproducible without
network access and the source stays traceable. Each table lists positions
``w * y + x`` within a ``w`` by ``h`` transform block, and is validated here to
be an exact permutation of ``0..w*h`` before it is emitted.

Usage::

    python3 scripts/generate-av1-scan-tables.py

Writes ``crates/otf-pixels-codec-avif/src/av1/scan_tables.rs``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "scripts" / "data" / "av1-scan-tables.txt"
OUT = ROOT / "crates" / "otf-pixels-codec-avif" / "src" / "av1" / "scan_tables.rs"

TABLE = re.compile(
    r"((?:Default|Mrow|Mcol)_Scan_(\d+)x(\d+))\[\s*(\d+)\s*\]\s*=\s*\{([^}]*)\}"
)

HEADER = """\
// Generated from the AV1 spec (§9.3) scan-order tables by
// scripts/generate-av1-scan-tables.py. Do not edit by hand.
//
// Each entry is a coefficient position `w * y + x` within the transform block.
// Every table is validated to be an exact permutation of `0..w*h` at
// generation time. Included by coeff.rs.
"""


def main() -> int:
    text = SPEC.read_text()
    tables: list[tuple[str, list[int]]] = []
    for match in TABLE.finditer(text):
        name, width, height, count = (
            match.group(1),
            int(match.group(2)),
            int(match.group(3)),
            int(match.group(4)),
        )
        nums = [int(x) for x in re.findall(r"\d+", match.group(5))]
        if len(nums) != count or count != width * height:
            raise SystemExit(f"{name}: expected {width * height} entries, got {len(nums)}")
        if sorted(nums) != list(range(count)):
            raise SystemExit(f"{name}: not a permutation of 0..{count}")
        tables.append((name, nums))

    if len(tables) != 32:
        raise SystemExit(f"expected 32 scan tables, found {len(tables)}")

    lines = [HEADER]
    for name, nums in tables:
        lines.append(f"pub(super) const {name.upper()}: [u16; {len(nums)}] = [")
        for i in range(0, len(nums), 16):
            lines.append("    " + ", ".join(str(x) for x in nums[i : i + 16]) + ",")
        lines.append("];")
        lines.append("")

    OUT.write_text("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
