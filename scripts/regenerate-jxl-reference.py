#!/usr/bin/env python3
"""Regenerate the JPEG XL decode fixtures and the rasters they are checked against.

The JPEG XL decoder is owned outright (ADR-0019), so the reference must come
from an independent implementation: fixtures are written by libjxl's `cjxl`
across the coding tools it uses (Modular with each effort, palette, squeeze;
VarDCT with its block types, progressive passes, an LF frame, noise, the
edge-preserving filter, upsampling, patches; JPEG recompression; alpha;
16-bit; animation), and the expected rasters are what libjxl's `djxl` makes of
those exact bytes. A lossless fixture must match to the byte; a lossy one
within the manifest's tolerance, since the two decoders round floats
differently.

Sources are generated procedurally from fixed content, so re-running this
reproduces them. Requires `cjxl` and `djxl` (libjxl-tools) on PATH, and Pillow.
"""

import os
import random
import subprocess
import sys
import tempfile

try:
    from PIL import Image, ImageDraw
except ImportError:
    sys.exit("Pillow is required: pip install pillow")

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "crates", "otf-pixels-codec-jxl", "tests", "fixtures")


def textured(width, height, seed=7):
    """Smooth gradients, edges and deterministic noise: something like a photo."""
    rng = random.Random(seed)
    image = Image.new("RGB", (width, height))
    px = image.load()
    for y in range(height):
        for x in range(width):
            n = rng.randint(-18, 18)
            base = ((x * 255) // max(width - 1, 1), (y * 255) // max(height - 1, 1), ((x * y) // 7) % 256)
            px[x, y] = tuple(max(0, min(255, c + n)) for c in base)
    draw = ImageDraw.Draw(image)
    draw.ellipse((width // 5, height // 5, width * 3 // 5, height * 4 // 5), fill=(230, 40, 50))
    draw.rectangle((width * 2 // 3, height // 8, width - 4, height // 2), fill=(20, 60, 200))
    return image


def flat(width, height):
    """Few colours in blocks, which `cjxl` codes with a palette."""
    colours = [(250, 250, 250), (10, 10, 10), (200, 30, 30), (30, 160, 60), (40, 70, 200)]
    image = Image.new("RGB", (width, height))
    px = image.load()
    for y in range(height):
        for x in range(width):
            px[x, y] = colours[((x // 11) + (y // 7) * 3) % len(colours)]
    return image


def text(width, height):
    """Repeated glyphs on white, which `cjxl --patches=1` codes as patches."""
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    for row in range(height // 14):
        draw.text((2, 2 + row * 14), "JPEG XL patches JPEG XL patches", fill=(0, 0, 0))
    return image


def with_alpha(image):
    out = image.convert("RGBA")
    alpha = Image.linear_gradient("L").resize(image.size)
    out.putalpha(alpha)
    return out


def grey16(width, height):
    image = Image.new("I;16", (width, height))
    px = image.load()
    for y in range(height):
        for x in range(width):
            px[x, y] = (x * 977 + y * 613 + (x * y) % 97 * 31) % 65536
    return image


def run(*args):
    subprocess.run(args, check=True, capture_output=True)


def pam_pixels(path):
    data = open(path, "rb").read()
    end = data.index(b"ENDHDR\n") + 7
    header = data[:end].decode()
    fields = dict(line.split(" ", 1) for line in header.splitlines()[1:-1])
    return int(fields["WIDTH"]), int(fields["HEIGHT"]), int(fields["DEPTH"]), int(fields["MAXVAL"]), data[end:]


def main():
    os.makedirs(OUT, exist_ok=True)
    tmp = tempfile.mkdtemp()
    sources = {
        "photo": textured(96, 72),
        "wide": textured(300, 140, seed=3),
        "flat": flat(70, 50),
        "text": text(240, 70),
        "alpha": with_alpha(textured(64, 48, seed=5)),
        "grey16": grey16(48, 40),
    }
    paths = {}
    for name, image in sources.items():
        paths[name] = os.path.join(tmp, name + ".png")
        image.save(paths[name])
    jpeg = os.path.join(tmp, "photo.jpg")
    sources["photo"].save(jpeg, quality=85, subsampling=2)
    frames = []
    for i in range(3):
        frame = Image.new("RGB", (64, 48), (20, 40, 60))
        ImageDraw.Draw(frame).rectangle((4 + i * 16, 8, 24 + i * 16, 36), fill=(250, 200, 30))
        frames.append(frame)
    gif = os.path.join(tmp, "anim.gif")
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=100, loop=0, disposal=1)

    # name, source, cjxl arguments, tolerance (in output steps)
    cases = [
        ("photo_lossless_e1", "photo", ["-d", "0", "-e", "1"], 0),
        ("photo_lossless_e3", "photo", ["-d", "0", "-e", "3"], 0),
        ("photo_lossless_e7", "photo", ["-d", "0", "-e", "7"], 0),
        ("photo_lossless_e9", "photo", ["-d", "0", "-e", "9"], 0),
        ("photo_lossless_squeeze", "photo", ["-d", "0", "-p"], 0),
        ("wide_lossless_groups", "wide", ["-d", "0", "-g", "0"], 0),
        ("flat_lossless_palette", "flat", ["-d", "0", "-e", "7"], 0),
        ("alpha_lossless", "alpha", ["-d", "0"], 0),
        ("grey16_lossless", "grey16", ["-d", "0", "-e", "7"], 0),
        ("photo_d1", "photo", ["-d", "1"], 1),
        ("photo_d1_e9", "photo", ["-d", "1", "-e", "9"], 1),
        ("photo_d4", "photo", ["-d", "4"], 1),
        ("wide_d2", "wide", ["-d", "2"], 1),
        ("wide_d1_progressive", "wide", ["-d", "1", "-p"], 1),
        ("wide_d1_lf_frame", "wide", ["-d", "1", "--progressive_dc=1"], 1),
        ("photo_d1_noise", "photo", ["-d", "1", "--photon_noise_iso=3200"], 1),
        ("photo_d1_epf3", "photo", ["-d", "1", "--epf=3"], 1),
        ("wide_d1_resampled", "wide", ["-d", "1", "--resampling=2"], 1),
        ("text_d1_patches", "text", ["-d", "1", "--patches=1"], 1),
        ("text_lossless_patches", "text", ["-d", "0", "--patches=1"], 0),
        ("alpha_d1", "alpha", ["-d", "1"], 1),
        ("photo_modular_lossy", "photo", ["-m", "1", "-d", "1"], 1),
        ("photo_jpeg", "jpeg", [], 1),
        ("anim", "gif", ["-d", "0"], 0),
    ]
    manifest = ["# Regenerate with scripts/regenerate-jxl-reference.py", "# name width height channels tolerance bits"]
    for name, source, args, tolerance in cases:
        src = {"jpeg": jpeg, "gif": gif}.get(source, paths.get(source))
        jxl = os.path.join(OUT, name + ".jxl")
        run("cjxl", src, jxl, *args)
        wide = source == "grey16"
        pam = os.path.join(tmp, name + ".pam")
        run("djxl", jxl, pam, "--bits_per_sample", "16" if wide else "8")
        width, height, depth, maxval, pixels = pam_pixels(pam)
        if maxval > 255:
            # PAM is big-endian; the fixtures are little-endian.
            pixels = b"".join(pixels[i + 1:i + 2] + pixels[i:i + 1] for i in range(0, len(pixels), 2))
        open(os.path.join(OUT, name + ".raw"), "wb").write(pixels)
        manifest.append(f"{name} {width} {height} {depth} {tolerance} {16 if maxval > 255 else 8}")
    open(os.path.join(OUT, "REFERENCE"), "w").write("\n".join(manifest) + "\n")


if __name__ == "__main__":
    main()
