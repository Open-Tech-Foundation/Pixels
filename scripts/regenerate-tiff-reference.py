#!/usr/bin/env python3
"""Regenerate the TIFF reference manifest from a reference TIFF decoder.

Our decoder is checked against ground truth from libtiff (via Pillow), not
merely against itself: a shared misreading of the specification round-trips
perfectly and is still wrong.

Canonical form
--------------
Eight-bit RGBA. Every v1 TIFF photometric and depth maps into it without
ambiguity, which is what lets one hash cover greyscale, RGB, palette and
bilevel files alike.

Sixteen-bit narrowing
---------------------
As for PNG, 16-bit samples are narrowed by discarding the low byte. Pillow
reports 16-bit greyscale as mode ``I;16`` and clips rather than scales when
converting it, so the script narrows those itself; everything else it has
already narrowed on load. Using one rule on both sides is what makes the
comparison mean anything.
"""

import argparse
import glob
import os
import sys

try:
    from PIL import Image
except ImportError:
    sys.exit("Pillow is required: pip install pillow")


def fnv1a64(data: bytes) -> int:
    """FNV-1a, 64-bit. Trivial to reimplement in Rust without a dependency."""
    h = 0xCBF29CE484222325
    for byte in data:
        h = ((h ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def canonical_rgba(image: Image.Image) -> bytes:
    """Decode to canonical eight-bit RGBA, correcting Pillow where needed."""
    if image.mode.startswith("I"):
        # Mode I;16 — convert("RGBA") clips to 255 instead of scaling.
        out = bytearray()
        for sample in image.getdata():
            grey = sample >> 8
            out += bytes((grey, grey, grey, 255))
        return bytes(out)
    return image.convert("RGBA").tobytes()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixtures", help="directory of .tif files")
    parser.add_argument("--dump", help="also write full .rgba rasters here")
    args = parser.parse_args()

    if args.dump:
        os.makedirs(args.dump, exist_ok=True)

    decoded, rejected = [], []
    for path in sorted(glob.glob(os.path.join(args.fixtures, "*.tif"))):
        name = os.path.basename(path)[:-4]
        try:
            image = Image.open(path)
            image.load()
            rgba = canonical_rgba(image)
        except Exception as error:
            rejected.append((name, str(error)[:60]))
            continue
        decoded.append((name, image.size, image.mode, fnv1a64(rgba)))
        if args.dump:
            with open(os.path.join(args.dump, name + ".rgba"), "wb") as handle:
                handle.write(rgba)

    manifest = os.path.join(args.fixtures, "REFERENCE")
    with open(manifest, "w") as handle:
        handle.write("# name width height source_mode fnv1a64_of_canonical_rgba\n")
        handle.write("# Canonical form: 8-bit RGBA, 16-bit samples narrowed by\n")
        handle.write("# discarding the low byte. source_mode is Pillow's mode,\n")
        handle.write("# recorded for diagnosis only.\n")
        handle.write("# Ground truth from libtiff via Pillow; see\n")
        handle.write("# scripts/regenerate-tiff-reference.py\n")
        for name, (width, height), mode, digest in decoded:
            handle.write(f"{name} {width} {height} {mode} {digest:016x}\n")

    print(f"reference decodings: {len(decoded)}")
    for name, error in rejected:
        print(f"  reference decoder rejected {name}: {error}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
