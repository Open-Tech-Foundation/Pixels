#!/usr/bin/env python3
"""Regenerate the GIF reference manifest from a reference GIF decoder.

Our decoder is checked against ground truth from libgif (via Pillow), not
merely against itself: a shared misreading of the specification round-trips
perfectly and is still wrong.

Canonical form
--------------
Eight-bit RGBA, one entry per *frame*, each frame being the whole canvas after
that frame has been composited. That is what a viewer draws — a frame's own
rectangle is meaningless without what it was composited onto — and it is the
form that makes disposal observable at all.

Pillow's frame handling
-----------------------
`ImageSequence` applies disposal itself, so `convert("RGBA")` on each frame
gives the composited canvas. The one thing to be careful of is that Pillow
caches palette state across frames, so each frame must be converted before
seeking to the next; doing it lazily afterwards yields the last frame's
palette applied to every frame.
"""

import argparse
import glob
import os
import sys

try:
    from PIL import Image, ImageSequence
except ImportError:
    sys.exit("Pillow is required: pip install pillow")


def fnv1a64(data: bytes) -> int:
    """FNV-1a, 64-bit. Trivial to reimplement in Rust without a dependency;
    this detects decoder differences, not attacks."""
    h = 0xCBF29CE484222325
    for byte in data:
        h = ((h ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixtures", help="directory of .gif files")
    parser.add_argument("--dump", help="also write full .rgba rasters here")
    args = parser.parse_args()

    if args.dump:
        os.makedirs(args.dump, exist_ok=True)

    decoded, rejected = [], []
    for path in sorted(glob.glob(os.path.join(args.fixtures, "*.gif"))):
        name = os.path.basename(path)[:-4]
        try:
            image = Image.open(path)
            size = image.size
            hashes = []
            for index, frame in enumerate(ImageSequence.Iterator(image)):
                # Converted eagerly: Pillow reuses palette state across
                # frames, so deferring this gives every frame the last one's
                # colours.
                rgba = frame.convert("RGBA").tobytes()
                hashes.append(fnv1a64(rgba))
                if args.dump:
                    with open(
                        os.path.join(args.dump, f"{name}.{index}.rgba"), "wb"
                    ) as handle:
                        handle.write(rgba)
        except Exception as error:
            rejected.append((name, str(error)[:60]))
            continue
        decoded.append((name, size, hashes))

    manifest = os.path.join(args.fixtures, "REFERENCE")
    with open(manifest, "w") as handle:
        handle.write("# name width height fnv1a64_per_frame...\n")
        handle.write("# Canonical form: 8-bit RGBA of the whole canvas after\n")
        handle.write("# each frame is composited, with disposal applied.\n")
        handle.write("# Ground truth from libgif via Pillow; see\n")
        handle.write("# scripts/regenerate-gif-reference.py\n")
        for name, (width, height), hashes in decoded:
            digests = " ".join(f"{h:016x}" for h in hashes)
            handle.write(f"{name} {width} {height} {digests}\n")

    print(f"reference decodings: {len(decoded)}")
    for name, error in rejected:
        print(f"  reference decoder rejected {name}: {error}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
