#!/usr/bin/env python3
"""Regenerate the PngSuite reference manifest from a reference PNG decoder.

Our decoder is checked against ground truth produced by libpng (via Pillow),
not merely against itself. Storing full RGBA rasters would add ~350 KB to the
repository for data that only ever gets compared, so the manifest records a
64-bit FNV-1a hash of each reference decoding instead.

Canonical form
--------------
Eight-bit RGBA, with 16-bit samples narrowed by discarding the low byte.
Narrowing is necessary because Pillow reports some 16-bit images at full
precision (mode ``I;16``) and others already narrowed (mode ``RGBA``), so
eight bits is the only representation both sides agree on.

Discarding the low byte is not the *good* reduction — ``round(v * 255 /
65535)`` is — but it is the only one available. Pillow narrows 16-bit colour
during load and never exposes the full-precision samples, so the script
cannot apply a better rule to those files even though it could to ``I;16``.
Using one rule everywhere keeps the comparison honest; the rule itself is a
comparison artifact on both sides, since our decoder emits full 16 bits and
narrows nothing.

What is *not* external ground truth
-----------------------------------
Two things in this manifest come from the script, not from libpng, because
Pillow gets them wrong and would otherwise bake its bugs into our reference:

* 16-bit greyscale. ``Image.convert("RGBA")`` on mode ``I;16`` *clips* to 255
  instead of scaling, so a raw sample of 2304 arrives as 255. We read the
  16-bit samples from ``getdata()`` and narrow them here instead.
* ``tRNS`` on greyscale. Pillow drops the transparency key for modes ``L``
  and ``1`` entirely. We apply it here, scaling the key from the file's IHDR
  bit depth to the eight-bit sample Pillow reports.

Pillow *is* trusted for everything else, including ``tRNS`` on truecolour and
palette images, which it applies correctly.

Deliberately corrupt files (PngSuite's ``x*``) are skipped: a conforming
decoder rejects them, so they have no reference decoding by definition. Note
that Pillow accepts some of them anyway, which is another reason not to ask
it for an answer.

Pass --dump DIR to also write the full RGBA rasters somewhere outside the
repository, which is what you want when a comparison actually fails.
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
    """FNV-1a, 64-bit. Chosen because it is trivial to reimplement in Rust
    without a dependency; this detects decoder differences, not attacks."""
    h = 0xCBF29CE484222325
    for byte in data:
        h = ((h ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def narrow16(value: int) -> int:
    """Narrow a 16-bit sample to eight bits, matching what Pillow already did
    to the 16-bit files it does not report at full precision."""
    return value >> 8


def ihdr_bit_depth(path: str) -> int:
    """Bit depth from the IHDR, which Pillow does not expose directly."""
    with open(path, "rb") as handle:
        return handle.read(26)[24]


def canonical_rgba(path: str, image: Image.Image) -> bytes:
    """Decode to canonical eight-bit RGBA, correcting Pillow where needed."""
    key = image.info.get("transparency")

    if image.mode.startswith("I"):
        # 16-bit greyscale: narrow here, because convert() would clip.
        out = bytearray()
        for sample in image.getdata():
            grey = narrow16(sample)
            alpha = 0 if sample == key else 255
            out += bytes((grey, grey, grey, alpha))
        return bytes(out)

    rgba = bytearray(image.convert("RGBA").tobytes())

    if image.mode in ("L", "1") and key is not None:
        # Greyscale tRNS, which convert() discards. Pillow reports the sample
        # already scaled to eight bits, so scale the key the same way.
        maximum = (1 << ihdr_bit_depth(path)) - 1
        scaled = (key * 255 + maximum // 2) // maximum
        for index, sample in enumerate(image.getdata()):
            if sample == scaled:
                rgba[index * 4 + 3] = 0

    return bytes(rgba)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixtures", help="directory of PngSuite .png files")
    parser.add_argument("--dump", help="also write full .rgba rasters here")
    args = parser.parse_args()

    if args.dump:
        os.makedirs(args.dump, exist_ok=True)

    decoded, rejected, skipped = [], [], 0
    for path in sorted(glob.glob(os.path.join(args.fixtures, "*.png"))):
        name = os.path.basename(path)[:-4]
        if name.startswith("x"):
            skipped += 1
            continue
        try:
            image = Image.open(path)
            image.load()
            rgba = canonical_rgba(path, image)
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
        handle.write("# discarding the low byte. source_mode is Pillow's mode for\n")
        handle.write("# the file, recorded for diagnosis only.\n")
        handle.write("# Ground truth from libpng via Pillow, except 16-bit\n")
        handle.write("# greyscale and greyscale tRNS; see\n")
        handle.write("# scripts/regenerate-pngsuite-reference.py\n")
        for name, (width, height), mode, digest in decoded:
            handle.write(f"{name} {width} {height} {mode} {digest:016x}\n")

    print(f"reference decodings: {len(decoded)}")
    print(f"deliberately corrupt files skipped: {skipped}")
    for name, error in rejected:
        print(f"  reference decoder rejected {name}: {error}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
