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
reproduces them. Requires `avifenc`/`avifdec` (libavif) and `aomenc`/`aomdec`
(libaom: the super-resolution fixtures and the plane-level references) on PATH,
and Pillow.
"""

import argparse
import os
import struct
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


def textured(width: int, height: int) -> Image.Image:
    """A gradient with deterministic high-frequency noise. Lossy coding of this
    rings, which is exactly what loop restoration is designed to clean up, so it
    is what makes the encoder actually pick Wiener/self-guided units."""
    import random

    rng = random.Random(1234)
    image = gradient(width, height)
    pixels = image.load()
    for y in range(height):
        for x in range(width):
            n = rng.randint(-55, 55)
            pixels[x, y] = tuple(max(0, min(255, c + n)) for c in pixels[x, y])
    return image


def mixed(width: int, height: int, seed: int) -> Image.Image:
    """A gradient split into a 4x3 grid of regions with different noise
    amplitudes, from clean to heavy. Regions that ring differently make the
    encoder choose differently per restoration unit — none, Wiener or
    self-guided, and a switchable frame type — across a multi-unit grid."""
    import random

    amplitudes = [0, 10, 30, 60, 5, 45, 20, 70, 15, 0, 40, 25]
    rng = random.Random(seed)
    image = gradient(width, height)
    pixels = image.load()
    for y in range(height):
        for x in range(width):
            region = (x * 4 // width) + (y * 3 // height) * 4
            amplitude = amplitudes[region % len(amplitudes)]
            n = rng.randint(-amplitude, amplitude) if amplitude else 0
            pixels[x, y] = tuple(max(0, min(255, c + n)) for c in pixels[x, y])
    return image


def with_alpha(image: Image.Image) -> Image.Image:
    """`image` with a horizontal alpha ramp, so the encoder writes an alpha item."""
    rgba = image.convert("RGBA")
    pixels = rgba.load()
    for y in range(rgba.height):
        for x in range(rgba.width):
            r, g, b, _ = pixels[x, y]
            pixels[x, y] = (r, g, b, (x * 255) // max(rgba.width - 1, 1))
    return rgba


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
    "gradient_deblock": (gradient(64, 64), False, 24, "444", 0),
    "blocks_deblock": (blocks(48, 40), False, 20, "444", 0),
    "gradient_odd_deblock": (gradient(50, 34), False, 28, "444", 0),
    "gradient_cdef": (gradient(64, 64), False, 20, "444", 0),
    "blocks_cdef": (blocks(48, 40), False, 20, "444", 0),
    "gradient_odd_cdef": (gradient(50, 34), False, 22, "444", 0),
    # Loop restoration (§7.17). The encoder only picks Wiener/self-guided units
    # at speed <= 4 on content that rings, so these are textured and encoded
    # slower (see `encode`). "restore" isolates loop restoration (deblock + CDEF
    # off); "restore_full" runs the whole in-loop pipeline (deblock + CDEF + LR)
    # so the stripe boundary — where restoration fetches pre-CDEF samples — is
    # exercised for real.
    "textured_nofilter": (textured(128, 128), False, 18, "444", 0),
    "textured_restore": (textured(128, 128), False, 18, "444", 0),
    "textured_odd_restore": (textured(100, 70), False, 22, "444", 0),
    "textured_restore_full": (textured(128, 128), False, 18, "444", 0),
    # Multi-unit restoration: a 2x2 unit grid per plane (256-sample units).
    # "mixed_restore" codes luma as none + Wiener units and chroma as a
    # switchable frame with none, Wiener and self-guided units; the "_full"
    # variant codes luma as switchable Wiener + self-guided under the whole
    # in-loop pipeline. Together they cover the restoration_type symbol, the
    # running per-plane Wiener/self-guided references across units, and unit
    # boundaries inside a stripe.
    "mixed_restore": (mixed(512, 384, 7), False, 18, "444", 0),
    "mixed_restore_full": (mixed(400, 384, 3), False, 25, "444", 0),
    # Super-resolution (§7.16), encoded with aomenc (see `encode_superres`).
    # The odd widths code at a width that is not a multiple of 8, where the
    # upscale reads decoded padding past FrameWidth. "mixed_superres_full" has
    # two restoration-unit columns, so the superres-scaled unit mapping in
    # read_lr decides which superblock reads which unit.
    "gradient_superres": (gradient(64, 48), False, 30, "444", 0),
    "textured_superres": (textured(128, 96), False, 30, "444", 0),
    "textured_odd_superres": (textured(101, 37), False, 30, "444", 0),
    "textured_superres_full": (textured(99, 70), False, 30, "444", 0),
    "mixed_superres_full": (mixed(420, 64, 5), False, 30, "444", 0),
    # Real YUV: subsampled chroma and non-identity matrices, converted to RGB
    # (see PHOTO_ARGS). The AV1 decode under these is pinned exactly by the
    # plane-level suite; what these check is the conversion, against libavif's.
    # libavif converts through libyuv, whose 8-bit matrices carry about six
    # fractional bits, so it sits a step or two from the exact result at full
    # range and up to five at studio range; our conversion matches exact
    # arithmetic (unit-tested in yuv.rs). The tolerances are those measured
    # gaps. "photo_default" is avifenc with no options at all (4:4:4, BT.601,
    # full range, speed 6) — the typical file this decoder meets.
    "photo_default": (mixed(160, 120, 11), False, 0, "", 1),
    "photo_420": (mixed(96, 64, 3), False, 0, "", 2),
    "photo_odd_420": (textured(37, 29), False, 0, "", 3),
    "photo_422": (mixed(96, 64, 3), False, 0, "", 2),
    "photo_444": (mixed(96, 64, 3), False, 0, "", 1),
    "photo_420_bt709_studio": (mixed(96, 64, 3), False, 0, "", 5),
    "photo_420_bt2020": (mixed(96, 64, 3), False, 0, "", 2),
    # 10- and 12-bit: decoded to 16-bit RGB over the full 0..=65535 range.
    # Here libavif converts in floating point rather than through libyuv, and
    # we agree on 99% of samples and within one 16-bit step on the rest — the
    # tolerance is 1 step in 65535.
    "photo_420_10bit": (mixed(96, 64, 3), False, 0, "", 1),
    "photo_420_10bit_studio": (mixed(96, 64, 3), False, 0, "", 1),
    "photo_444_12bit": (mixed(96, 64, 3), False, 0, "", 1),
    "photo_444_10bit_identity": (mixed(96, 64, 3), False, 0, "", 1),
    # Alpha (an auxiliary monochrome AV1 item) and monochrome pictures: RGBA,
    # grey and grey+alpha, 8- and 16-bit.
    "photo_alpha": (with_alpha(mixed(96, 64, 3)), False, 0, "", 1),
    "photo_alpha_420": (with_alpha(mixed(96, 64, 3)), False, 0, "", 2),
    "photo_alpha_10bit": (with_alpha(mixed(96, 64, 3)), False, 0, "", 1),
    "grey": (mixed(96, 64, 3), False, 0, "", 0),
    "grey_studio": (mixed(96, 64, 3), False, 0, "", 0),
    "grey_10bit": (mixed(96, 64, 3), False, 0, "", 1),
    "grey_alpha": (with_alpha(mixed(96, 64, 3)), False, 0, "", 0),
}

# 8-bit fixtures whose reference is taken from avifdec's 16-bit output and
# rounded to 8 bits. avifdec writes an 8-bit monochrome picture as a grey PNG of
# the raw Y samples, skipping the studio-range expansion (Y = 16 stays 16
# rather than becoming black); its 16-bit path converts properly, and rounded
# to 8 bits it matches our decode exactly.
VIA_16 = {"grey_studio"}

# Channels in our raster where it is not RGB: 1 grey, 2 grey+alpha, 4 RGBA.
CHANNELS = {
    "photo_alpha": 4,
    "photo_alpha_420": 4,
    "photo_alpha_10bit": 4,
    "grey": 1,
    "grey_studio": 1,
    "grey_10bit": 1,
    "grey_alpha": 2,
}

# The fixtures whose references are 16 bits per sample.
WIDE = {
    "photo_420_10bit",
    "photo_420_10bit_studio",
    "photo_444_12bit",
    "photo_444_10bit_identity",
    "photo_alpha_10bit",
    "grey_10bit",
}

# avifenc arguments for the "photo" fixtures, used verbatim. Without --cicp
# avifenc writes BT.601 (matrix 6) at full range.
PHOTO_ARGS = {
    "photo_default": [],
    "photo_420": ["-y", "420", "-q", "60"],
    "photo_odd_420": ["-y", "420", "-q", "60"],
    "photo_422": ["-y", "422", "-q", "60"],
    "photo_444": ["-y", "444", "-q", "60"],
    "photo_420_bt709_studio": ["-y", "420", "-q", "60", "--cicp", "1/13/1", "-r", "limited"],
    "photo_420_bt2020": ["-y", "420", "-q", "60", "--cicp", "9/16/9"],
    "photo_420_10bit": ["-d", "10", "-y", "420", "-q", "60"],
    "photo_420_10bit_studio": ["-d", "10", "-y", "420", "-q", "60", "--cicp", "1/13/1", "-r", "limited"],
    "photo_444_12bit": ["-d", "12", "-y", "444", "-q", "60"],
    "photo_444_10bit_identity": ["-d", "10", "-y", "444", "-q", "60", "--cicp", "1/13/0", "-r", "full"],
    "photo_alpha": ["-q", "60"],
    "photo_alpha_420": ["-y", "420", "-q", "60"],
    "photo_alpha_10bit": ["-d", "10", "-q", "60"],
    "grey": ["-y", "400", "-q", "60"],
    "grey_studio": ["-y", "400", "-q", "60", "-r", "limited"],
    "grey_10bit": ["-d", "10", "-y", "400", "-q", "60"],
    "grey_alpha": ["-y", "400", "-q", "60"],
}

# SuperresDenom for each "superres" fixture (SUPERRES_NUM is 8).
SUPERRES_DENOMINATORS = {
    "gradient_superres": 16,
    "textured_superres": 12,
    "textured_odd_superres": 9,
    "textured_superres_full": 15,
    "mixed_superres_full": 10,
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

# "deblock" fixtures keep the deblocking loop filter (§7.14) ON but disable the
# later post-filters we do not implement yet (CDEF, loop restoration) and the
# per-block quantiser variation. Our reconstruct applies deblocking, so these
# must also decode byte-exact.
DEBLOCK_AOM_OPTS = [
    "enable-cdef=0",
    "enable-restoration=0",
    "deltaq-mode=0",
    "enable-tpl-model=0",
]

# "cdef" fixtures turn the CDEF post-filter (§7.15) ON and, to isolate it,
# deblocking OFF (loopfilter-control=0), so a CDEF bug is the only thing that can
# make the raster differ. CDEF must be enabled explicitly here — with these `-a`
# overrides it otherwise defaults off. Loop restoration stays off (unimplemented)
# and the quantiser is uniform. Our reconstruct applies CDEF, so these decode
# byte-exact.
CDEF_AOM_OPTS = [
    "enable-cdef=1",
    "enable-restoration=0",
    "loopfilter-control=0",
    "deltaq-mode=0",
    "enable-tpl-model=0",
]

# "restore" fixtures turn loop restoration (§7.17) ON, isolated by turning
# deblocking and CDEF off, so a restoration bug is the only thing that can make
# the raster differ. Restoration must be enabled explicitly (it defaults off with
# these `-a` overrides).
RESTORE_AOM_OPTS = [
    "enable-restoration=1",
    "enable-cdef=0",
    "loopfilter-control=0",
    "deltaq-mode=0",
    "enable-tpl-model=0",
]

# "restore_full" fixtures run the whole implemented in-loop pipeline: deblocking,
# CDEF and loop restoration all ON. This exercises the restoration stripe
# boundary, where samples are fetched from the pre-CDEF frame rather than the
# CDEF output.
RESTORE_FULL_AOM_OPTS = [
    "enable-restoration=1",
    "enable-cdef=1",
    "deltaq-mode=0",
    "enable-tpl-model=0",
]


# "superres" fixtures code the frame at a reduced width and upscale it (§7.16).
# libavif does not forward aom's super-resolution settings (they are encoder
# configuration, not codec controls), so these are encoded with `aomenc` itself
# and wrapped into an AVIF container by `mux_avif`; their "quality" is aomenc's
# cq-level and their denominator (9..16: the frame is coded at 8/denominator of
# its width) comes from SUPERRES_DENOMINATORS. "superres" isolates the upscale
# with every in-loop filter off; "superres_full" runs deblock + CDEF +
# restoration too, where restoration operates on the upscaled frame.
SUPERRES_AOMENC_OPTS = [
    "--enable-cdef=0",
    "--enable-restoration=0",
    "--loopfilter-control=0",
    "--deltaq-mode=0",
    "--enable-tpl-model=0",
]

SUPERRES_FULL_AOMENC_OPTS = [
    "--enable-cdef=1",
    "--enable-restoration=1",
    "--deltaq-mode=0",
    "--enable-tpl-model=0",
]


def box(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", 8 + len(payload)) + kind + payload


def full_box(kind: bytes, version: int, flags: int, payload: bytes) -> bytes:
    return box(kind, struct.pack(">I", (version << 24) | flags) + payload)


def split_obus(data: bytes) -> list:
    """Split a low-overhead OBU stream (every OBU has obu_has_size_field set,
    as aomenc writes them) into `(obu_type, whole_obu_bytes)` pairs."""
    obus = []
    pos = 0
    while pos < len(data):
        header = data[pos]
        obu_type = (header >> 3) & 0xF
        extension = (header >> 2) & 1
        if not (header >> 1) & 1:
            sys.exit("OBU without a size field")
        cursor = pos + 1 + extension
        size = 0
        shift = 0
        while True:
            byte = data[cursor]
            cursor += 1
            size |= (byte & 0x7F) << shift
            shift += 7
            if not byte & 0x80:
                break
        end = cursor + size
        obus.append((obu_type, data[pos:end]))
        pos = end
    return obus


def sequence_level(sequence_header_obu: bytes) -> tuple:
    """`(seq_profile, seq_level_idx[0], seq_tier[0])` from a sequence header OBU
    with no timing info (as aomenc writes for a single frame)."""
    header = sequence_header_obu[0]
    cursor = 1 + ((header >> 2) & 1)
    while sequence_header_obu[cursor] & 0x80:
        cursor += 1
    cursor += 1
    bits = "".join(f"{b:08b}" for b in sequence_header_obu[cursor:cursor + 8])
    profile = int(bits[0:3], 2)
    reduced = bits[4] == "1"
    if reduced:
        return profile, int(bits[5:10], 2), 0
    if bits[5] == "1":
        sys.exit("sequence header with timing info is not handled")
    # initial_display_delay_present_flag, operating_points_cnt_minus_1,
    # operating_point_idc[0], then seq_level_idx[0] and maybe seq_tier[0].
    at = 6 + 1 + 5 + 12
    level = int(bits[at:at + 5], 2)
    tier = int(bits[at + 5], 2) if level > 7 else 0
    return profile, level, tier


def mux_avif(obu_stream: bytes, width: int, height: int, path: str) -> None:
    """Wrap one 8-bit 4:4:4 identity-matrix AV1 key frame into a minimal AVIF
    still: ftyp, meta (hdlr, pitm, iloc, iinf, iprp with ispe/av1C/colr/pixi)
    and mdat. The temporal delimiter is dropped; the sequence header is both the
    av1C config OBU and the first OBU of the item data, as avifenc writes it."""
    obus = [o for o in split_obus(obu_stream) if o[0] != 2]  # 2: temporal delimiter
    sequence_header = next(o for t, o in obus if t == 1)
    item = b"".join(o for _, o in obus)
    profile, level, tier = sequence_level(sequence_header)

    av1c = box(
        b"av1C",
        bytes([0x81, (profile << 5) | level, tier << 7, 0]) + sequence_header,
    )
    ispe = full_box(b"ispe", 0, 0, struct.pack(">II", width, height))
    # CICP 1/13/0, full range: the identity matrix the other fixtures use.
    colr = box(b"colr", b"nclx" + struct.pack(">HHHB", 1, 13, 0, 0x80))
    pixi = full_box(b"pixi", 0, 0, bytes([3, 8, 8, 8]))
    ipco = box(b"ipco", ispe + av1c + colr + pixi)
    ipma = full_box(b"ipma", 0, 0, struct.pack(">IHB", 1, 1, 4) + bytes([1, 0x82, 3, 4]))
    iprp = box(b"iprp", ipco + ipma)
    hdlr = full_box(b"hdlr", 0, 0, b"\0\0\0\0pict" + b"\0" * 12 + b"\0")
    pitm = full_box(b"pitm", 0, 0, struct.pack(">H", 1))
    infe = full_box(b"infe", 2, 0, struct.pack(">HH", 1, 0) + b"av01" + b"\0")
    iinf = full_box(b"iinf", 0, 0, struct.pack(">H", 1) + infe)
    ftyp = box(b"ftyp", b"avif" + struct.pack(">I", 0) + b"avifmif1miaf")

    def meta_with(offset: int) -> bytes:
        iloc = full_box(
            b"iloc",
            0,
            0,
            bytes([0x44, 0x00])
            + struct.pack(">HHHHII", 1, 1, 0, 1, offset, len(item)),
        )
        return full_box(b"meta", 0, 0, hdlr + pitm + iloc + iinf + iprp)

    # The item's offset depends on the meta size, which does not depend on it.
    offset = len(ftyp) + len(meta_with(0)) + 8
    with open(path, "wb") as out:
        out.write(ftyp + meta_with(offset) + box(b"mdat", item))


def encode_superres(image: Image.Image, path: str, cq_level: int, denominator: int) -> None:
    """Encode `image` with aomenc at a fixed super-resolution denominator and
    mux the key frame into an AVIF (see `mux_avif`)."""
    width, height = image.size
    rgb = image.convert("RGB")
    # Identity matrix: Y = G, U = B, V = R, each a full-resolution plane.
    planes = [rgb.getchannel(c).tobytes() for c in ("G", "B", "R")]
    y4m = path + ".src.y4m"
    obu = path + ".obu"
    with open(y4m, "wb") as out:
        out.write(f"YUV4MPEG2 W{width} H{height} F1:1 Ip A1:1 C444\nFRAME\n".encode())
        out.write(b"".join(planes))
    opts = SUPERRES_FULL_AOMENC_OPTS if "superres_full" in path else SUPERRES_AOMENC_OPTS
    cmd = [
        "aomenc", y4m, "--obu", "-o", obu, "--limit=1", "--usage=2", "--profile=1",
        "--end-usage=q", f"--cq-level={cq_level}", "--superres-mode=1",
        f"--superres-kf-denominator={denominator}", f"--superres-denominator={denominator}",
        "--color-primaries=bt709", "--transfer-characteristics=srgb",
        "--matrix-coefficients=identity", *opts,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with open(obu, "rb") as data:
        mux_avif(data.read(), width, height, path)
    os.remove(y4m)
    os.remove(obu)


def encode(image: Image.Image, path: str, lossless: bool, quality: int, yuv: str) -> None:
    name = os.path.splitext(os.path.basename(path))[0]
    if name in PHOTO_ARGS:
        png = path + ".src.png"
        image.save(png, "PNG")
        subprocess.run(
            ["avifenc", *PHOTO_ARGS[name], png, path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        os.remove(png)
        return
    if name in SUPERRES_DENOMINATORS:
        encode_superres(image, path, quality, SUPERRES_DENOMINATORS[name])
        return
    png = path + ".src.png"
    image.save(png, "PNG")
    basename = os.path.basename(path)
    # The encoder only searches loop-restoration units at speed <= 4, so the
    # "restore" fixtures are encoded slower. The "nofilter" fixtures are too:
    # speed 4 also reaches block shapes (4x16/16x4 with angle deltas and
    # palettes) that speed 6 never picks. Everything else stays at speed 6.
    speed = "4" if ("restore" in basename or "nofilter" in basename) else "6"
    cmd = ["avifenc", "-s", speed, "-y", yuv]
    if lossless:
        cmd.append("--lossless")
    else:
        # Lossy with an identity colour matrix (matrix_coefficients == 0, full
        # range) so the decode compares in the same RGB == (V, Y, U) space the
        # lossless fixtures use, and with the relevant post-filters selected.
        cmd += ["-q", str(quality), "-r", "full", "--cicp", "1/13/0"]
        if "restore_full" in basename:
            opts = RESTORE_FULL_AOM_OPTS
        elif "restore" in basename:
            opts = RESTORE_AOM_OPTS
        elif "cdef" in basename:
            opts = CDEF_AOM_OPTS
        elif "deblock" in basename:
            opts = DEBLOCK_AOM_OPTS
        else:
            opts = NOFILTER_AOM_OPTS
        for opt in opts:
            cmd += ["-a", opt]
    cmd += [png, path]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.remove(png)


def read_png16(path: str) -> tuple:
    """`(width, height, channels, samples)` of a non-interlaced 16-bit PNG
    (grey, grey+alpha, RGB or RGBA), the samples as a flat list of ints.
    Pillow cannot hold 16-bit colour, so this is a minimal reader for exactly
    what `avifdec -d 16` writes."""
    import zlib

    with open(path, "rb") as f:
        data = f.read()
    pos, idat, header = 8, b"", None
    while pos < len(data):
        length, kind = struct.unpack(">I4s", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + length]
        if kind == b"IHDR":
            header = struct.unpack(">IIBBBBB", body)
        elif kind == b"IDAT":
            idat += body
        pos += 12 + length
    width, height, depth, colour, _, _, interlace = header
    channels = {0: 1, 4: 2, 2: 3, 6: 4}.get(colour)
    if depth != 16 or interlace != 0 or channels is None:
        sys.exit(f"{path}: expected a 16-bit non-interlaced PNG, got {header}")
    raw = zlib.decompress(idat)
    bpp = channels * 2
    stride = width * bpp
    samples = []
    prev = bytearray(stride)
    for y in range(height):
        kind = raw[y * (stride + 1)]
        line = bytearray(raw[y * (stride + 1) + 1:(y + 1) * (stride + 1)])
        for i in range(stride):
            a = line[i - bpp] if i >= bpp else 0
            b = prev[i]
            c = prev[i - bpp] if i >= bpp else 0
            if kind == 1:
                line[i] = (line[i] + a) & 0xFF
            elif kind == 2:
                line[i] = (line[i] + b) & 0xFF
            elif kind == 3:
                line[i] = (line[i] + (a + b) // 2) & 0xFF
            elif kind == 4:
                pa, pb, pc = abs(b - c), abs(a - c), abs(a + b - 2 * c)
                pred = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
                line[i] = (line[i] + pred) & 0xFF
        prev = line
        samples += [(line[i] << 8) | line[i + 1] for i in range(0, stride, 2)]
    return width, height, channels, samples


def to_layout(name: str, samples: list, have: int, want: int) -> list:
    """Rearrange pixels of `have` channels into our layout of `want`: grey
    (1), grey+alpha (2), RGB (3) or RGBA (4). libavif writes grey as equal
    R, G and B; that is checked, then one of them is kept."""
    out = []
    for i in range(0, len(samples), have):
        px = samples[i:i + have]
        colour, alpha = (px[:-1], px[-1:]) if have in (2, 4) else (px, [])
        if want in (1, 2):
            if len(set(colour)) != 1:
                sys.exit(f"{name}: a grey fixture decoded to non-grey {colour}")
            colour = colour[:1]
        elif len(colour) == 1:
            colour = colour * 3
        out += colour + (alpha if want in (2, 4) else [])
    return out


def reference_samples(path: str, name: str, channels: int, bits: int) -> tuple:
    """libavif's decode of these bytes in our layout: `(width, height, raster
    bytes)`, one byte per sample at 8 bits and little-endian pairs at 16."""
    png = path + ".ref.png"
    if bits == 8 and name in VIA_16:
        width, height, raster = reference_samples(path, name, channels, 16)
        wide = struct.unpack(f"<{len(raster) // 2}H", raster)
        return width, height, bytes((v * 255 * 2 + 65535) // (2 * 65535) for v in wide)
    subprocess.run(
        ["avifdec", "-d", str(bits), path, png],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if bits == 16:
        width, height, have, samples = read_png16(png)
        values = to_layout(name, samples, have, channels)
        raster = b"".join(struct.pack("<H", v) for v in values)
    else:
        with Image.open(png) as decoded:
            decoded.load()
            have = {"L": 1, "LA": 2, "RGB": 3, "RGBA": 4}.get(decoded.mode)
            if have is None:
                sys.exit(f"{name}: unexpected PNG mode {decoded.mode}")
            width, height = decoded.size
            values = to_layout(name, list(decoded.tobytes()), have, channels)
            raster = bytes(values)
    os.remove(png)
    return width, height, raster


def soft(width: int, height: int) -> Image.Image:
    """A very gentle three-channel ramp. Nearly flat content coded slowly at low
    quality is what makes the encoder pick 128x128 blocks (with a 128x128
    superblock), whose residual the decoder walks in 64x64 chunks."""
    image = Image.new("RGB", (width, height))
    pixels = image.load()
    for y in range(height):
        for x in range(width):
            pixels[x, y] = (
                100 + x * 40 // width,
                90 + y * 30 // height,
                120 + (x + y) * 20 // (width + height),
            )
    return image


# Plane-level fixtures: the decoded Y/U/V planes, before any colour conversion,
# checked against libaom's `aomdec --rawvideo` output of the same coded frame
# (`tests/planes.rs`). This is exact regardless of how YUV later becomes RGB,
# so it is where subsampled chroma is verified. name -> (image, avifenc args).
FILTERS_OFF = [
    "-a", "enable-cdef=0", "-a", "enable-restoration=0", "-a", "loopfilter-control=0",
    "-a", "deltaq-mode=0", "-a", "enable-tpl-model=0",
]
# Every in-loop filter left at the encoder's default (on); only the per-block
# quantizer deltas the decoder does not implement are turned off.
FILTERS_ON = ["-a", "deltaq-mode=0", "-a", "enable-tpl-model=0"]
PLANES = {
    # 4:2:0 with every filter off, then with them all on (speed 4 also reaches
    # restoration and the 4x16/16x4 shapes); the odd sizes put blocks — and a
    # chroma-from-luma block's luma — over the frame edge.
    "gradient_420_nofilter": (gradient(64, 48), ["-y", "420", "-q", "40", "-s", "6", *FILTERS_OFF]),
    "gradient_odd_420_nofilter": (gradient(37, 29), ["-y", "420", "-q", "40", "-s", "6", *FILTERS_OFF]),
    "textured_420": (textured(128, 96), ["-y", "420", "-q", "40", "-s", "4", *FILTERS_ON]),
    "textured_odd_420": (textured(101, 37), ["-y", "420", "-q", "40", "-s", "4", *FILTERS_ON]),
    "mixed_420": (mixed(256, 192, 3), ["-y", "420", "-q", "50", "-s", "4", *FILTERS_ON]),
    "textured_422": (textured(99, 70), ["-y", "422", "-q", "40", "-s", "4", *FILTERS_ON]),
    # Screen content: palettes on subsampled chroma, including blocks whose
    # colour map overhangs the frame edge (only the on-screen part is coded).
    "blocks_420_palette": (blocks(90, 70), ["-y", "420", "-q", "40", "-s", "6", *FILTERS_OFF]),
    "blocks_422_palette": (blocks(51, 37), ["-y", "422", "-q", "60", "-s", "6", *FILTERS_ON]),
    "blocks_444_palette": (
        blocks(50, 36),
        ["-y", "444", "-r", "full", "--cicp", "1/13/0", "-q", "60", "-s", "6", *FILTERS_ON],
    ),
    # Monochrome (4:0:0): aomdec writes the Y plane alone.
    "textured_odd_400": (textured(101, 37), ["-y", "400", "-q", "40", "-s", "4", *FILTERS_ON]),
    "blocks_400_palette": (blocks(90, 70), ["-y", "400", "-q", "40", "-s", "6", *FILTERS_ON]),
    # 10- and 12-bit: aomdec writes these planes as 16-bit little-endian.
    "textured_odd_420_10bit": (textured(101, 37), ["-d", "10", "-y", "420", "-q", "40", "-s", "4", *FILTERS_ON]),
    "textured_444_12bit": (textured(99, 70), ["-d", "12", "-y", "444", "-q", "40", "-s", "4", *FILTERS_ON]),
    "blocks_420_10bit_palette": (blocks(90, 70), ["-d", "10", "-y", "420", "-q", "40", "-s", "6", *FILTERS_ON]),
    # 128x128 superblocks holding 128-wide blocks: the 64x64 residual chunks.
    "soft_420_sb128": (soft(136, 72), ["-y", "420", "-q", "5", "-s", "4", "-a", "sb-size=128", *FILTERS_ON]),
}


def carve_primary_item(path: str) -> bytes:
    """The `mdat` payload of an avifenc still: its one coded item's OBUs."""
    with open(path, "rb") as f:
        data = f.read()
    pos = 0
    while pos < len(data):
        size, kind = struct.unpack(">I4s", data[pos:pos + 8])
        header = 8
        if size == 1:
            size = struct.unpack(">Q", data[pos + 8:pos + 16])[0]
            header = 16
        elif size == 0:
            size = len(data) - pos
        if kind == b"mdat":
            return data[pos + header:pos + size]
        pos += size
    sys.exit(f"{path}: no mdat box")


def write_planes(fixtures: str) -> None:
    directory = os.path.join(fixtures, "planes")
    os.makedirs(directory, exist_ok=True)
    for name, (image, args) in sorted(PLANES.items()):
        base = os.path.join(directory, name)
        image.save(base + ".src.png", "PNG")
        subprocess.run(
            ["avifenc", *args, base + ".src.png", base + ".avif"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        os.remove(base + ".src.png")
        with open(base + ".obu", "wb") as out:
            out.write(carve_primary_item(base + ".avif"))
        # aomdec writes the display-cropped planes, Y then U then V, 8-bit.
        subprocess.run(
            ["aomdec", "--rawvideo", "-o", base + ".yuv", base + ".obu"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        os.remove(base + ".obu")
        print(f"planes/{name}: {os.path.getsize(base + '.avif')} bytes")


# Files this decoder must *refuse*: each uses one coding tool it does not
# implement yet, and decoding past such a tool without it yields a wrong image
# with no error. `tests/unsupported.rs` asserts every file here decodes to
# `Unsupported`; when a tool lands, its file moves into FIXTURES with a
# reference raster. name -> (image, avifenc arguments). All lossy, 4:4:4 with
# the identity matrix unless the tool under test is the colour format itself.
IDENTITY_444 = ["-y", "444", "-r", "full", "--cicp", "1/13/0", "-q", "60"]
UNSUPPORTED = {
    "delta_q": (mixed(128, 64, 3), [*IDENTITY_444, "-a", "deltaq-mode=3"]),
    "qmatrix": (textured(64, 64), [*IDENTITY_444, "-a", "enable-qm=1"]),
    "premultiplied": (with_alpha(textured(32, 32)), ["-q", "60", "--premultiply"]),
    "ycgco": (textured(32, 32), ["-y", "444", "-q", "60", "--cicp", "1/13/8"]),
    "tiles": (textured(128, 64), [*IDENTITY_444, "--tilecolslog2", "1"]),
}


def write_unsupported(fixtures: str) -> None:
    directory = os.path.join(fixtures, "unsupported")
    os.makedirs(directory, exist_ok=True)
    for name, (image, args) in sorted(UNSUPPORTED.items()):
        png = os.path.join(directory, name + ".src.png")
        image.save(png, "PNG")
        path = os.path.join(directory, name + ".avif")
        subprocess.run(
            ["avifenc", "-s", "6", *args, png, path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        os.remove(png)
        print(f"unsupported/{name}: {os.path.getsize(path)} bytes")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixtures", help="directory to write fixtures into")
    args = parser.parse_args()
    os.makedirs(args.fixtures, exist_ok=True)

    manifest = [
        "# Regenerate with scripts/regenerate-avif-reference.py",
        "# name width height channels tolerance bits",
    ]
    for name, (image, lossless, quality, yuv, tolerance) in sorted(FIXTURES.items()):
        path = os.path.join(args.fixtures, f"{name}.avif")
        encode(image, path, lossless, quality, yuv)

        # 10- and 12-bit files decode to 16-bit samples (little-endian here).
        bits = 16 if name in WIDE else 8
        channels = CHANNELS.get(name, 3)
        width, height, raster = reference_samples(path, name, channels, bits)
        if lossless and raster != image.convert("RGB").tobytes():
            sys.exit(f"{name}: lossless fixture did not round-trip through libavif")

        expected = width * height * channels * bits // 8
        if len(raster) != expected:
            sys.exit(f"{name}: raster is {len(raster)} bytes, expected {expected}")

        with open(os.path.join(args.fixtures, f"{name}.raw"), "wb") as out:
            out.write(raster)
        manifest.append(f"{name} {width} {height} {channels} {tolerance} {bits}")
        print(f"{name}: {width}x{height}x{channels}, {os.path.getsize(path)} bytes")

    write_unsupported(args.fixtures)
    write_planes(args.fixtures)

    with open(os.path.join(args.fixtures, "REFERENCE"), "w") as out:
        out.write("\n".join(manifest) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
