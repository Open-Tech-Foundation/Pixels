#!/usr/bin/env python3
"""Regenerate the JPEG fixtures and the reference rasters they are checked against.

Our decoder is checked against ground truth from libjpeg-turbo (via Pillow),
not merely against itself: a shared misreading of the specification round-trips
perfectly and is still wrong.

Why rasters and not hashes
--------------------------
The GIF and PNG suites compare hashes, because those formats define an exact
answer. JPEG does not. The standard specifies the inverse DCT only to an
accuracy bound, so two conforming decoders routinely differ by a step or two
per sample, and a hash would fail on a decoder that is entirely correct. So the
reference raster is stored in full and compared with a tolerance, which is the
only comparison the format actually licenses.

Chroma upsampling
-----------------
libjpeg upsamples 4:2:2 and 4:2:0 chroma with a triangle filter; we use
nearest-neighbour, because a triangle filter needs rows from the *next* MCU row
and our decoder is built to hold exactly one. The two agree except near a
chroma edge, so the subsampled fixtures come in two kinds: smoothly shaded
ones, compared across the whole raster, and a hard-edged one whose flat
interiors are compared tightly while its edges are left alone. Fixtures saved
at 4:4:4 involve no upsampling at all and are compared everywhere.

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


def gradient(width: int, height: int) -> Image.Image:
    """A smooth two-axis gradient: everything a DCT compresses well."""
    image = Image.new("RGB", (width, height))
    pixels = image.load()
    for y in range(height):
        for x in range(width):
            pixels[x, y] = (
                (x * 255) // max(width - 1, 1),
                (y * 255) // max(height - 1, 1),
                ((x + y) * 255) // max(width + height - 2, 1),
            )
    return image


def blocks(width: int, height: int, cell: tuple = (7, 5)) -> Image.Image:
    """Hard-edged colour blocks: the worst case for a DCT, and so the case
    that exercises the high-frequency coefficients a smooth image never
    reaches.

    `cell` sets how large each block of flat colour is. Subsampled fixtures
    need blocks wide enough to have an interior, because that interior is
    where the reference comparison can hold chroma to a tight bound — at the
    edges, a triangle upsampler and a nearest one legitimately disagree.
    """
    palette = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (0, 255, 255), (255, 0, 255), (0, 0, 0), (255, 255, 255),
    ]
    image = Image.new("RGB", (width, height))
    pixels = image.load()
    for y in range(height):
        for x in range(width):
            pixels[x, y] = palette[((x // cell[0]) + (y // cell[1])) % len(palette)]
    return image


def noise(width: int, height: int) -> Image.Image:
    """Deterministic pseudo-random noise, which leaves no coefficient zero and
    so gives the entropy decoder the longest codes it will ever see."""
    image = Image.new("RGB", (width, height))
    pixels = image.load()
    state = 0x12345678
    for y in range(height):
        for x in range(width):
            channel = []
            for _ in range(3):
                state = (state * 1103515245 + 12345) & 0xFFFFFFFF
                channel.append((state >> 16) & 0xFF)
            pixels[x, y] = tuple(channel)
    return image


# name -> (source image, mode, save options)
FIXTURES = {
    # 4:4:4, so no chroma upsampling is involved at all. Odd dimensions mean
    # neither axis is a whole number of blocks.
    "gradient444": (gradient(61, 37), "RGB", dict(quality=92, subsampling=0)),
    "blocks444": (blocks(61, 37), "RGB", dict(quality=90, subsampling=0)),
    "noise444": (noise(48, 32), "RGB", dict(quality=95, subsampling=0)),
    # Grayscale: one component, so an MCU is one block.
    "gray": (gradient(61, 37).convert("L"), "L", dict(quality=90)),
    "graynoise": (noise(40, 24).convert("L"), "L", dict(quality=85)),
    # 4:2:2 and 4:2:0: 2x1 and 2x2 luma sampling, the interleaved MCU layouts.
    "gradient422": (gradient(64, 48), "RGB", dict(quality=90, subsampling=1)),
    "gradient420": (gradient(64, 48), "RGB", dict(quality=90, subsampling=2)),
    "blocks420": (blocks(70, 42, cell=(18, 14)), "RGB", dict(quality=88, subsampling=2)),
    # Restart markers every two MCUs, which forces resynchronization and a
    # predictor reset mid-row and mid-image.
    "restart420": (
        gradient(64, 48), "RGB",
        dict(quality=90, subsampling=2, restart_marker_blocks=2),
    ),
    "restart444": (
        blocks(61, 37), "RGB",
        dict(quality=90, subsampling=0, restart_marker_blocks=3),
    ),
    # Progressive, which a wrapped decoder handles (ADR-0004). Compared like
    # any other fixture: what matters is that the seam between our header
    # parsing and the wrapped decoder produces the same picture libjpeg does.
    "progressive": (
        gradient(48, 32), "RGB",
        dict(quality=90, subsampling=0, progressive=True),
    ),
    "progressive420": (
        blocks(64, 48), "RGB",
        dict(quality=85, subsampling=2, progressive=True),
    ),
    # Smaller than one MCU in both axes: the whole image is edge padding.
    "tiny420": (blocks(3, 2), "RGB", dict(quality=80, subsampling=2)),
    "tiny444": (blocks(1, 1), "RGB", dict(quality=80, subsampling=0)),
}


# Fixtures this codec deliberately does not decode. They carry no reference
# raster — the expected outcome is a specific refusal, which `tests/malformed.rs`
# asserts — but they are still fed to the fuzz corpus, where "does not panic"
# applies to them like anything else.
UNSUPPORTED = {
    "cmyk": (gradient(48, 32).convert("CMYK"), dict(quality=90)),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixtures", help="directory to write fixtures into")
    args = parser.parse_args()
    os.makedirs(args.fixtures, exist_ok=True)

    manifest = [
        "# Regenerate with scripts/regenerate-jpeg-reference.py",
        "# name width height channels",
    ]
    for name, (image, mode, options) in sorted(FIXTURES.items()):
        path = os.path.join(args.fixtures, f"{name}.jpg")
        image.save(path, "JPEG", **options)

        # Decode what we just wrote, rather than reusing the source image: the
        # reference is what a reference decoder makes of these exact bytes,
        # not what went into the encoder.
        with Image.open(path) as decoded:
            decoded.load()
            if decoded.mode != mode:
                sys.exit(f"{name}: reopened as {decoded.mode}, expected {mode}")
            raster = decoded.tobytes()
            width, height = decoded.size

        channels = 1 if mode == "L" else 3
        expected = width * height * channels
        if len(raster) != expected:
            sys.exit(f"{name}: raster is {len(raster)} bytes, expected {expected}")

        with open(os.path.join(args.fixtures, f"{name}.raw"), "wb") as out:
            out.write(raster)

        # libjpeg's own M/8 scaled decode, which Pillow exposes as draft mode.
        # This is the reference for our reduced IDCT: without it the only
        # check on a scaled decode would be against our own full decode, which
        # cannot catch a shared misreading of what M/8 even means.
        #
        # Pillow chooses the reduction by integer division of the current size
        # by the requested one, so the request is a floor and the result a
        # ceiling. A fixture smaller than the divisor cannot be driven to that
        # reduction at all, and is skipped rather than fudged.
        for denominator in (8, 4, 2):
            if width < denominator or height < denominator:
                continue
            request = (width // denominator, height // denominator)
            expected_size = (-(-width // denominator), -(-height // denominator))
            with Image.open(path) as draft:
                draft.draft(mode, request)
                draft.load()
                if draft.size != expected_size:
                    sys.exit(
                        f"{name}: draft at 1/{denominator} gave {draft.size}, "
                        f"expected {expected_size}"
                    )
                data = draft.convert(mode).tobytes()
            suffix = f"s{8 // denominator}"
            with open(os.path.join(args.fixtures, f"{name}.{suffix}.raw"), "wb") as out:
                out.write(data)

        manifest.append(f"{name} {width} {height} {channels}")
        print(f"{name}: {width}x{height}x{channels}, {os.path.getsize(path)} bytes")

    for name, (image, options) in sorted(UNSUPPORTED.items()):
        path = os.path.join(args.fixtures, f"{name}.jpg")
        image.save(path, "JPEG", **options)
        print(f"{name}: unsupported by design, {os.path.getsize(path)} bytes")

    with open(os.path.join(args.fixtures, "REFERENCE"), "w") as out:
        out.write("\n".join(manifest) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
