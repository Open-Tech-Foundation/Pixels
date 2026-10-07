#!/usr/bin/env python3
"""Write the multi-page TIFF fixture with libtiff (via Pillow).

Three pages that differ in size, colour type and content, so decoding the
wrong page cannot match the reference by accident:

- page 0: 48x32 RGB gradient
- page 1: 24x40 greyscale stripes
- page 2: 16x16 palette checkerboard

Run from the repository root, then regenerate the reference manifest:

    python3 scripts/generate-tiff-multipage-fixture.py
    python3 scripts/regenerate-tiff-reference.py crates/otf-pixels-codec-tiff/tests/fixtures
"""

import sys

try:
    from PIL import Image
except ImportError:
    sys.exit("Pillow is required: pip install pillow")

OUT = "crates/otf-pixels-codec-tiff/tests/fixtures/multipage.tif"

rgb = Image.new("RGB", (48, 32))
rgb.putdata([(x * 5, y * 8, 255 - x * 5) for y in range(32) for x in range(48)])

grey = Image.new("L", (24, 40))
grey.putdata([255 if (y // 4) % 2 else 30 for y in range(40) for _ in range(24)])

palette = Image.new("P", (16, 16))
palette.putpalette([0, 0, 0, 220, 40, 40] + [0, 0, 0] * 254)
palette.putdata([((x // 4) + (y // 4)) % 2 for y in range(16) for x in range(16)])

# LZW makes Pillow write through libtiff, the reference this is checked against.
rgb.save(OUT, save_all=True, append_images=[grey, palette], compression="tiff_lzw")
print(f"wrote {OUT}")
