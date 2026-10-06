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


def noise(width: int, height: int, seed: int, alpha: bool = False, smooth: bool = False) -> Image.Image:
    """Seeded pseudo-random pixels. `smooth` biases neighbours together, the
    content libwebp's predictor and colour transforms are for."""
    state = seed * 2654435761 % (1 << 32) or 1

    def rand() -> int:
        nonlocal state
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        return state

    mode = "RGBA" if alpha else "RGB"
    image = Image.new(mode, (width, height))
    pixels = image.load()
    for y in range(height):
        for x in range(width):
            if smooth:
                base = ((x * 3 + y * 2) % 200, (x * 2 + y * 5) % 180, (x + y * 3) % 220)
                value = tuple(min(255, c + rand() % 12) for c in base)
            else:
                value = (rand() % 256, rand() % 256, rand() % 256)
            if alpha:
                value = value + ((x * 17 + y * 5) % 256 if smooth else rand() % 256,)
            pixels[x, y] = value
    return image


def palette_image(width: int, height: int, colours: int, seed: int, alpha: bool = False) -> Image.Image:
    """Exactly `colours` distinct colours, which selects libwebp's colour
    indexing transform and, at 16 or fewer, pixel bundling."""
    source = noise(colours, 1, seed, alpha)
    table = list(source.getdata())
    # Distinct by construction: fold the index into one channel.
    table = [tuple([i, *t[1:]]) for i, t in enumerate(table)]
    image = Image.new(source.mode, (width, height))
    pixels = image.load()
    for y in range(height):
        for x in range(width):
            pixels[x, y] = table[((x // 3) * 7 + (y // 2) * 3 + x * y) % colours]
    return image


# Lossless decode corpus: name -> (image, method, quality). Spread over what
# libwebp's encoder chooses between — every transform, colour indexing at each
# bundling width, the colour cache, meta prefix codes on larger images — and
# the degenerate shapes. The reference is libwebp's decode of the same bytes.
LOSSLESS = {
    "noise_m0": (noise(37, 23, 1), 0, 0),
    "noise_m6": (noise(37, 23, 2), 6, 100),
    "smooth_m4": (noise(96, 64, 3, smooth=True), 4, 75),
    "smooth_m6": (noise(96, 64, 4, smooth=True), 6, 100),
    "smooth_alpha_m4": (noise(80, 50, 5, alpha=True, smooth=True), 4, 75),
    "noise_alpha_m2": (noise(40, 30, 6, alpha=True), 2, 50),
    "palette2": (palette_image(61, 19, 2, 7), 4, 75),
    "palette3": (palette_image(61, 19, 3, 8), 4, 75),
    "palette4_alpha": (palette_image(45, 21, 4, 9, alpha=True), 4, 75),
    "palette5": (palette_image(45, 21, 5, 10), 6, 100),
    "palette16": (palette_image(70, 33, 16, 11), 4, 75),
    "palette17": (palette_image(70, 33, 17, 12), 4, 75),
    "palette256": (palette_image(64, 64, 256, 13), 6, 100),
    "large_smooth_m4": (noise(512, 384, 14, smooth=True), 4, 75),
    "large_mixed_m6": (Image.composite(noise(400, 300, 15, smooth=True), noise(400, 300, 16),
                                       palette_image(400, 300, 2, 17).convert("L")), 6, 90),
    "large_alpha_m5": (noise(300, 200, 18, alpha=True, smooth=True), 5, 80),
    "column": (noise(1, 37, 19, smooth=True), 4, 75),
    "row": (noise(37, 1, 20, smooth=True), 4, 75),
    "pixel_alpha": (noise(1, 1, 21, alpha=True), 4, 75),
    "blocks_m6": (blocks(200, 150, False), 6, 100),
}


def write_lossless(fixtures: str) -> None:
    directory = os.path.join(fixtures, "lossless")
    os.makedirs(directory, exist_ok=True)
    manifest = [
        "# Regenerate with scripts/regenerate-webp-reference.py",
        "# name width height channels",
    ]
    for name, (image, method, quality) in sorted(LOSSLESS.items()):
        path = os.path.join(directory, f"{name}.webp")
        image.save(path, "WEBP", lossless=True, method=method, quality=quality, exact=True)
        with Image.open(path) as decoded:
            decoded.load()
            mode = decoded.mode
            raster = decoded.tobytes()
            width, height = decoded.size
        channels = {"RGB": 3, "RGBA": 4}[mode]
        with open(os.path.join(directory, f"{name}.raw"), "wb") as out:
            out.write(raster)
        manifest.append(f"{name} {width} {height} {channels}")
        print(f"lossless/{name}: {width}x{height}x{channels}, {os.path.getsize(path)} bytes")
    with open(os.path.join(directory, "REFERENCE"), "w") as out:
        out.write("\n".join(manifest) + "\n")


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
    write_lossless(args.fixtures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
