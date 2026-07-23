#!/usr/bin/env python3
"""Regenerate the AVIF fixtures and the reference rasters they are checked against.

Unlike the wrapped codecs, AVIF is owned outright (ADR-0013): what is under test
is our own AV1 bitstream decoder, so the reference *must* come from an
independent implementation, never from ourselves. Fixtures are encoded with
libavif's `avifenc` (libaom) and the expected rasters are what libavif's
`avifdec` makes of those exact bytes — decode us, compare against them.

The lossless fixtures are the first reconstruction target and the strictest
check there is: lossless AVIF is `CodedLossless`, which turns off every
post-filter and uses only the 4x4 Walsh-Hadamard transform, and its raster must
equal the source exactly. Lossy fixtures are compared with a tolerance, because
a lossy decode is only required to be close, and they exercise the DCT/ADST
paths and the post-filters as those land.

Fixtures are generated procedurally from fixed content, so re-running this
reproduces them. Requires `avifenc`/`avifdec` on PATH (libavif) and Pillow.
"""

import argparse
import os
import subprocess
import sys

try:
    from PIL import Image
except ImportError:
    sys.exit("Pillow is required: pip install pillow")


def gradient(width: int, height: int) -> Image.Image:
    """A smooth two-axis RGB gradient."""
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


def blocks(width: int, height: int) -> Image.Image:
    """Hard-edged colour blocks; lossy coding blurs their edges, lossless does not."""
    palette = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (0, 255, 255), (255, 0, 255), (0, 0, 0), (255, 255, 255),
    ]
    image = Image.new("RGB", (width, height))
    pixels = image.load()
    for y in range(height):
        for x in range(width):
            pixels[x, y] = palette[((x // 9) + (y // 7)) % len(palette)]
    return image


# name -> (image, lossless, quality, yuv, tolerance)
#
# Lossless fixtures are `CodedLossless` and must round-trip exactly. The
# "nofilter" lossy fixtures are lossy AVIF encoded with every in-loop post-filter
# turned off (see `encode`): our reconstruct is filter-free, so with the filters
# off it must still match libavif's decode to the byte. They exercise the
# DCT/ADST inverse transforms, the larger transform sizes and chroma-from-luma
# that lossless never reaches. A true lossy fixture (filters on) will only join
# once the post-filters are implemented, and then with a tolerance.
FIXTURES = {
    "gradient_lossless": (gradient(64, 48), True, 100, "444", 0),
    "blocks_lossless": (blocks(64, 48), True, 100, "444", 0),
    "gradient_odd_lossless": (gradient(37, 29), True, 100, "444", 0),
    "tiny_lossless": (blocks(4, 4), True, 100, "444", 0),
    "gradient_nofilter": (gradient(64, 48), False, 30, "444", 0),
    "blocks_nofilter": (blocks(48, 40), False, 40, "444", 0),
    "gradient_odd_nofilter": (gradient(37, 29), False, 35, "444", 0),
}

# aom options that disable every in-loop post-filter, so a filter-free decoder
# reproduces the frame exactly: deblock, CDEF, loop restoration, the delta-q /
# TPL machinery that would vary the quantiser per block.
NOFILTER_AOM_OPTS = [
    "enable-cdef=0",
    "enable-restoration=0",
    "loopfilter-control=0",
    "deltaq-mode=0",
    "enable-tpl-model=0",
]


def encode(image: Image.Image, path: str, lossless: bool, quality: int, yuv: str) -> None:
    png = path + ".src.png"
    image.save(png, "PNG")
    cmd = ["avifenc", "-s", "6", "-y", yuv]
    if lossless:
        cmd.append("--lossless")
    else:
        # Lossy with an identity colour matrix (matrix_coefficients == 0, full
        # range) so the decode compares in the same RGB == (V, Y, U) space the
        # lossless fixtures use, and with every post-filter disabled.
        cmd += ["-q", str(quality), "-r", "full", "--cicp", "1/13/0"]
        for opt in NOFILTER_AOM_OPTS:
            cmd += ["-a", opt]
    cmd += [png, path]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.remove(png)


def reference_raster(path: str) -> Image.Image:
    """What libavif's decoder makes of these exact bytes, as 8-bit RGB."""
    png = path + ".ref.png"
    subprocess.run(
        ["avifdec", "-d", "8", path, png],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    with Image.open(png) as decoded:
        decoded.load()
        rgb = decoded.convert("RGB")
    os.remove(png)
    return rgb


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixtures", help="directory to write fixtures into")
    args = parser.parse_args()
    os.makedirs(args.fixtures, exist_ok=True)

    manifest = [
        "# Regenerate with scripts/regenerate-avif-reference.py",
        "# name width height channels tolerance",
    ]
    for name, (image, lossless, quality, yuv, tolerance) in sorted(FIXTURES.items()):
        path = os.path.join(args.fixtures, f"{name}.avif")
        encode(image, path, lossless, quality, yuv)

        reference = reference_raster(path)
        width, height = reference.size
        raster = reference.tobytes()

        if lossless and reference.tobytes() != image.convert("RGB").tobytes():
            sys.exit(f"{name}: lossless fixture did not round-trip through libavif")

        channels = 3
        expected = width * height * channels
        if len(raster) != expected:
            sys.exit(f"{name}: raster is {len(raster)} bytes, expected {expected}")

        with open(os.path.join(args.fixtures, f"{name}.raw"), "wb") as out:
            out.write(raster)
        manifest.append(f"{name} {width} {height} {channels} {tolerance}")
        print(f"{name}: {width}x{height}x{channels}, {os.path.getsize(path)} bytes")

    with open(os.path.join(args.fixtures, "REFERENCE"), "w") as out:
        out.write("\n".join(manifest) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
