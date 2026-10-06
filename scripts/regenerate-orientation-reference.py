#!/usr/bin/env python3
"""Regenerate the auto-orientation fixtures for the otf-pixels facade.

Every file stores the same pixels sideways in some way and declares how to
turn them upright. The expected raster is the upright image as an independent
implementation sees it — Pillow's ``ImageOps.exif_transpose`` for EXIF, and
Pillow's own rotate/transpose applied to libavif's ``irot``/``imir`` options
for AVIF — never our own orientation code, which is what is on trial.

Fixtures
--------
``exif{N}.{tif,png,webp,jpg}``  EXIF Orientation N (1-8); PNG in ``eXIf``.
``avif_{name}.avif``        avifenc --irot/--imir combinations, lossless.
``*.raw``                   the upright RGB8 raster, row-major, no header.

TIFF, PNG and WebP are lossless, so the comparison is byte-exact. JPEG is lossy; its
raster comes from libjpeg through Pillow, and its picture is flat 8x8 blocks at
quality 100 without subsampling so the decoders cannot drift apart by more
than rounding. AVIF is ``avifenc -l``, which round-trips the source exactly.

Requires Pillow and avifenc (libavif).
"""

import argparse
import os
import subprocess
import sys
import tempfile

try:
    from PIL import Image, ImageOps
except ImportError:
    sys.exit("Pillow is required: pip install pillow")

# A small picture with no symmetry at all: every pixel distinct, and width and
# height different, so each of the eight orientations gives a different
# raster and a wrong one cannot pass by coincidence.
WIDTH, HEIGHT = 7, 5

# For JPEG, the same idea at block granularity: 3x2 flat 8x8 blocks.
BLOCK_COLOURS = [
    (230, 30, 30), (30, 200, 40), (40, 60, 220),
    (240, 220, 30), (200, 40, 210), (30, 210, 220),
]

# (fixture name, irot anticlockwise quarter turns or None, imir mode or None).
# libavif: "--irot ANGLE: 90 * ANGLE degree rotation anti-clockwise";
# "--imir AXIS: 0=top-to-bottom, 1=left-to-right".
AVIF_CASES = [
    ("irot1", 1, None),
    ("irot2", 2, None),
    ("irot3", 3, None),
    ("imir0", None, 0),
    ("imir1", None, 1),
    ("irot1_imir0", 1, 0),
    ("irot1_imir1", 1, 1),
    ("irot3_imir1", 3, 1),
]


def distinct_pixels():
    image = Image.new("RGB", (WIDTH, HEIGHT))
    image.putdata(
        [((x * 37) % 256, (y * 53) % 256, (x * 11 + y * 29 + 7) % 256)
         for y in range(HEIGHT) for x in range(WIDTH)]
    )
    return image


def blocks():
    image = Image.new("RGB", (24, 16))
    for index, colour in enumerate(BLOCK_COLOURS):
        bx, by = index % 3, index // 3
        image.paste(colour, (bx * 8, by * 8, bx * 8 + 8, by * 8 + 8))
    return image


def exif_with(orientation):
    exif = Image.Exif()
    exif[0x0112] = orientation
    return exif.tobytes()


def write_raw(path, image):
    with open(path, "wb") as f:
        f.write(image.convert("RGB").tobytes())


def upright(path):
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def avif_upright(source, irot, imir):
    # Transformative properties apply in order: rotation, then mirror.
    image = source
    if irot:
        image = image.rotate(90 * irot, expand=True)  # Pillow: anticlockwise
    if imir == 0:
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    elif imir == 1:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        default=os.path.join(
            os.path.dirname(__file__), "..", "crates", "otf-pixels", "tests",
            "fixtures", "orientation",
        ),
    )
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    source = distinct_pixels()
    block_source = blocks()

    for n in range(1, 9):
        exif = exif_with(n)
        tif = os.path.join(args.out, f"exif{n}.tif")
        source.save(tif, tiffinfo={0x0112: n})
        png = os.path.join(args.out, f"exif{n}.png")
        source.save(png, exif=exif)
        webp = os.path.join(args.out, f"exif{n}.webp")
        source.save(webp, lossless=True, exif=exif)
        jpg = os.path.join(args.out, f"exif{n}.jpg")
        block_source.save(jpg, quality=100, subsampling=0, exif=exif)
        for path in (tif, png, webp, jpg):
            write_raw(os.path.splitext(path)[0] + "." + path.rsplit(".", 1)[1] + ".raw",
                      upright(path))

    with tempfile.TemporaryDirectory() as scratch:
        png = os.path.join(scratch, "source.png")
        source.save(png)
        for name, irot, imir in AVIF_CASES:
            avif = os.path.join(args.out, f"avif_{name}.avif")
            command = ["avifenc", "-l", "-s", "10"]
            if irot is not None:
                command += ["--irot", str(irot)]
            if imir is not None:
                command += ["--imir", str(imir)]
            subprocess.run(command + [png, avif], check=True, capture_output=True)
            write_raw(avif + ".raw", avif_upright(source, irot, imir))

    print(f"wrote orientation fixtures to {os.path.normpath(args.out)}")


if __name__ == "__main__":
    main()
