#!/usr/bin/env python3
"""Regenerate the WebP fixtures and the reference rasters they are checked against.

The WebP codec is wrapped rather than owned (ADR-0004), so what is under test
is not the VP8 bitstream — libwebp and `image-webp` are both mature — but our
adaptation: dimensions, pixel format, alpha detection, row order, and the
error mapping. Those are exactly the things a wrapper gets wrong, and exactly
the things a reference raster catches.

Lossless fixtures carry an exact expected raster, because lossless means
lossless: any difference at all is our bug. Lossy fixtures are compared with a
tolerance, since libwebp's decoder and `image-webp`'s do not have to agree to
the last step.

Fixtures are generated procedurally from a fixed seed, so re-running this
script reproduces them byte for byte.
"""

import argparse
import os
import sys

try:
    from PIL import Image
except ImportError:
    sys.exit("Pillow is required: pip install pillow")


def gradient(width: int, height: int, alpha: bool) -> Image.Image:
    """A smooth two-axis gradient, optionally with a diagonal alpha ramp."""
    mode = "RGBA" if alpha else "RGB"
    image = Image.new(mode, (width, height))
    pixels = image.load()
    for y in range(height):
        for x in range(width):
            value = (
                (x * 255) // max(width - 1, 1),
                (y * 255) // max(height - 1, 1),
                ((x + y) * 255) // max(width + height - 2, 1),
            )
            pixels[x, y] = value + (((x + y) * 255) // max(width + height - 2, 1),) if alpha else value
    return image


def blocks(width: int, height: int, alpha: bool) -> Image.Image:
    """Hard-edged colour blocks, which lossy coding blurs and lossless does not."""
    palette = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (0, 255, 255), (255, 0, 255), (0, 0, 0), (255, 255, 255),
    ]
    mode = "RGBA" if alpha else "RGB"
    image = Image.new(mode, (width, height))
    pixels = image.load()
    for y in range(height):
        for x in range(width):
            colour = palette[((x // 9) + (y // 7)) % len(palette)]
            pixels[x, y] = colour + (255 if (x // 9) % 2 == 0 else 128,) if alpha else colour
    return image


def grey(width: int, height: int) -> Image.Image:
    """Greyscale, which WebP has no native mode for — it round-trips as RGB."""
    return gradient(width, height, alpha=False).convert("L")


# name -> (image, lossless, expected decoded mode)
FIXTURES = {
    "gradient_lossless": (gradient(61, 37, False), True, "RGB"),
    "blocks_lossless": (blocks(64, 48, False), True, "RGB"),
    "alpha_lossless": (gradient(48, 32, True), True, "RGBA"),
    "blocks_alpha_lossless": (blocks(45, 29, True), True, "RGBA"),
    "tiny_lossless": (blocks(1, 1, False), True, "RGB"),
    "gradient_lossy": (gradient(64, 48, False), False, "RGB"),
    "blocks_lossy": (blocks(61, 37, False), False, "RGB"),
    "alpha_lossy": (gradient(48, 32, True), False, "RGBA"),
    "grey_lossless": (grey(40, 24), True, "RGB"),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixtures", help="directory to write fixtures into")
    args = parser.parse_args()
    os.makedirs(args.fixtures, exist_ok=True)

    manifest = [
        "# Regenerate with scripts/regenerate-webp-reference.py",
        "# name width height channels lossless",
    ]
    for name, (image, lossless, mode) in sorted(FIXTURES.items()):
        path = os.path.join(args.fixtures, f"{name}.webp")
        image.save(path, "WEBP", lossless=lossless, quality=90 if not lossless else 100)

        # Decode what was just written, not the source image: the reference is
        # what a reference decoder makes of these exact bytes.
        with Image.open(path) as decoded:
            decoded.load()
            converted = decoded.convert(mode)
            raster = converted.tobytes()
            width, height = converted.size

        channels = 4 if mode == "RGBA" else 3
        expected = width * height * channels
        if len(raster) != expected:
            sys.exit(f"{name}: raster is {len(raster)} bytes, expected {expected}")

        with open(os.path.join(args.fixtures, f"{name}.raw"), "wb") as out:
            out.write(raster)
        manifest.append(f"{name} {width} {height} {channels} {int(lossless)}")
        print(f"{name}: {width}x{height}x{channels}, {os.path.getsize(path)} bytes")

    with open(os.path.join(args.fixtures, "REFERENCE"), "w") as out:
        out.write("\n".join(manifest) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
