#!/usr/bin/env python3
"""Regenerate the ICC fixtures for the otf-pixels facade.

Profiles are written here from first principles, not copied from anywhere:
each is a matrix/TRC display profile built from its primaries, white point
and transfer curve, so the fixtures carry no third-party data. Every one is
then handed to lcms2 (through Pillow's ImageCms), which is the reference the
conversion is checked against — never our own colour code, which is what is on
trial.

Profiles
--------
``display_p3.icc``  ICC v4, DCI-P3 primaries, D65, the sRGB curve as a
                    parametric ``para`` (function 3).
``adobe_rgb.icc``   ICC v2, Adobe RGB (1998) primaries, D65, gamma 563/256
                    as a single-entry ``curv``.
``prophoto.icc``    ICC v2, ROMM primaries, D50 white (no adaptation), gamma
                    1.8.
``rec2020.icc``     ICC v2, BT.2020 primaries, D65, the BT.709 curve sampled
                    into a 1024-entry ``curv`` table.
``grey_gamma22.icc`` ICC v2 grey profile, ``kTRC`` gamma 2.2.

Fixtures
--------
``{profile}.png``          a saturated test card tagged with the profile.
``{profile}.srgb.raw``     lcms2's relative-colorimetric conversion of it to
                           sRGB: RGB8 (grey: Gray8), row-major, no header.
``display_p3_alpha.png``   RGBA; its ``.srgb.raw`` keeps alpha untouched.
``display_p3.{jpg,webp,tif,avif}``, ``display_p3_progressive.jpg``
                           the same profile in every container that has a
                           place for one, for extraction.

Requires Pillow (with lcms2 and WebP) and avifenc (libavif).
"""

import argparse
import math
import os
import struct
import subprocess
import tempfile

from PIL import Image, ImageCms

D50 = (0.9642, 1.0, 0.8249)
D65_XY = (0.3127, 0.3290)
D50_XY = (0.3457, 0.3585)

BRADFORD = [
    [0.8951, 0.2664, -0.1614],
    [-0.7502, 1.7135, 0.0367],
    [0.0389, -0.0685, 1.0296],
]


def mat_mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def mat_vec(a, v):
    return [sum(a[i][k] * v[k] for k in range(3)) for i in range(3)]


def mat_inv(m):
    (a, b, c), (d, e, f), (g, h, i) = m
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    return [
        [(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det],
        [(f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det],
        [(d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det],
    ]


def xy_to_xyz(x, y):
    return [x / y, 1.0, (1 - x - y) / y]


def colorants(primaries, white):
    """The D50-adapted rXYZ/gXYZ/bXYZ columns, as ICC.1 Annex D derives them."""
    cols = [xy_to_xyz(*p) for p in primaries]
    m = [[cols[j][i] for j in range(3)] for i in range(3)]
    w = xy_to_xyz(*white)
    s = mat_vec(mat_inv(m), w)
    rgb_to_xyz = [[m[i][j] * s[j] for j in range(3)] for i in range(3)]
    src = mat_vec(BRADFORD, w)
    dst = mat_vec(BRADFORD, list(D50))
    scale = [[dst[i] / src[i] if i == j else 0.0 for j in range(3)] for i in range(3)]
    adapt = mat_mul(mat_inv(BRADFORD), mat_mul(scale, BRADFORD))
    adapted = mat_mul(adapt, rgb_to_xyz)
    return [[adapted[i][j] for i in range(3)] for j in range(3)]


def s15(v):
    return struct.pack(">i", round(v * 65536))


def xyz_tag(xyz):
    return b"XYZ \0\0\0\0" + b"".join(s15(v) for v in xyz)


def curv_gamma(gamma):
    return b"curv\0\0\0\0" + struct.pack(">IH", 1, round(gamma * 256)) + b"\0\0"


def curv_table(f, n):
    body = b"curv\0\0\0\0" + struct.pack(">I", n)
    body += b"".join(struct.pack(">H", round(f(i / (n - 1)) * 65535)) for i in range(n))
    return body + b"\0" * (len(body) % 4)


def para_srgb():
    g, a, b, c, d = 2.4, 1 / 1.055, 0.055 / 1.055, 1 / 12.92, 0.04045
    return b"para\0\0\0\0" + struct.pack(">HH", 3, 0) + b"".join(s15(v) for v in (g, a, b, c, d))


def desc_v2(text):
    raw = text.encode() + b"\0"
    return b"desc\0\0\0\0" + struct.pack(">I", len(raw)) + raw + b"\0" * 12 + b"\0" * 67


def mluc(text):
    raw = text.encode("utf-16-be")
    return b"mluc\0\0\0\0" + struct.pack(">II", 1, 12) + b"enUS" + struct.pack(">II", len(raw), 28) + raw


def text_v2(text):
    return b"text\0\0\0\0" + text.encode() + b"\0"


def profile(version, space, tags):
    """An ICC profile: header, tag table, 4-byte aligned tag data."""
    table = struct.pack(">I", len(tags))
    data = b""
    offset = 128 + 4 + 12 * len(tags)
    entries = []
    for sig, body in tags:
        while (offset + len(data)) % 4:
            data += b"\0"
        entries.append((sig, offset + len(data), len(body)))
        data += body
    for sig, at, size in entries:
        table += sig + struct.pack(">II", at, size)
    size = 128 + len(table) + len(data)
    header = struct.pack(">I", size) + b"\0\0\0\0" + struct.pack(">I", version)
    header += b"mntr" + space + b"XYZ " + struct.pack(">6H", 2026, 1, 1, 0, 0, 0)
    header += b"acsp" + b"\0" * 4 + struct.pack(">I", 0) + b"\0" * 8 + b"\0" * 8
    header += struct.pack(">I", 0) + b"".join(s15(v) for v in D50) + b"\0" * 4
    header += b"\0" * 16 + b"\0" * 28
    assert len(header) == 128
    return header + table + data


def rgb_profile(name, version, primaries, white, trc):
    cols = colorants(primaries, white)
    describe = mluc if version >= 0x04000000 else desc_v2
    copyright = mluc if version >= 0x04000000 else text_v2
    tags = [
        (b"desc", describe(name)),
        (b"cprt", copyright("No copyright, use freely")),
        (b"wtpt", xyz_tag(D50)),
        (b"rXYZ", xyz_tag(cols[0])),
        (b"gXYZ", xyz_tag(cols[1])),
        (b"bXYZ", xyz_tag(cols[2])),
        (b"rTRC", trc),
        (b"gTRC", trc),
        (b"bTRC", trc),
    ]
    return profile(version, b"RGB ", tags)


def bt709_eotf(v):
    return v / 4.5 if v < 0.081 else ((v + 0.099) / 1.099) ** (1 / 0.45)


PROFILES = {
    "display_p3": rgb_profile(
        "Display P3 (test)", 0x04300000,
        [(0.680, 0.320), (0.265, 0.690), (0.150, 0.060)], D65_XY, para_srgb(),
    ),
    "adobe_rgb": rgb_profile(
        "Adobe RGB (test)", 0x02100000,
        [(0.64, 0.33), (0.21, 0.71), (0.15, 0.06)], D65_XY, curv_gamma(563 / 256),
    ),
    "prophoto": rgb_profile(
        "ProPhoto (test)", 0x02100000,
        [(0.7347, 0.2653), (0.1596, 0.8404), (0.0366, 0.0001)], D50_XY, curv_gamma(1.8),
    ),
    "rec2020": rgb_profile(
        "BT.2020 (test)", 0x02100000,
        [(0.708, 0.292), (0.170, 0.797), (0.131, 0.046)], D65_XY, curv_table(bt709_eotf, 1024),
    ),
    "grey_gamma22": profile(0x02100000, b"GRAY", [
        (b"desc", desc_v2("Grey gamma 2.2 (test)")),
        (b"cprt", text_v2("No copyright, use freely")),
        (b"wtpt", xyz_tag(D50)),
        (b"kTRC", curv_gamma(2.2)),
    ]),
}


def card(width, height):
    """Saturated hue sweeps over a lightness ramp, plus neutral and pure rows."""
    image = Image.new("RGB", (width, height))
    px = image.load()
    for y in range(height):
        for x in range(width):
            h = x / width * 6
            i, f = int(h) % 6, h - int(h)
            r, g, b = [(1, f, 0), (1 - f, 1, 0), (0, 1, f), (0, 1 - f, 1), (f, 0, 1), (1, 0, 1 - f)][i]
            light = y / (height - 1)
            if y % 16 < 2:
                r = g = b = x / (width - 1)
            else:
                r, g, b = (c * light + (1 - light) * 0.5 * c for c in (r, g, b))
            px[x, y] = tuple(round(c * 255) for c in (r, g, b))
    return image


def to_srgb(image, icc, mode):
    """lcms2's conversion, relative colorimetric, unoptimised (full precision)."""
    src = ImageCms.ImageCmsProfile(__import__("io").BytesIO(icc))
    dst = ImageCms.createProfile("sRGB")
    return ImageCms.profileToProfile(
        image, src, dst,
        renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
        outputMode=mode,
        flags=ImageCms.Flags.NOOPTIMIZE,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fixtures")
    out = parser.parse_args().fixtures
    os.makedirs(out, exist_ok=True)
    base = card(96, 64)
    for name, icc in PROFILES.items():
        with open(os.path.join(out, name + ".icc"), "wb") as f:
            f.write(icc)
        grey = name.startswith("grey")
        image = base.convert("L") if grey else base
        image.save(os.path.join(out, name + ".png"), icc_profile=icc)
        converted = to_srgb(image, icc, "RGB")
        if grey:
            # lcms maps grey onto sRGB's neutral axis; it stays grey.
            r, g, b = converted.split()
            assert r.tobytes() == g.tobytes() == b.tobytes(), name
            converted = r
        with open(os.path.join(out, name + ".srgb.raw"), "wb") as f:
            f.write(converted.tobytes())
        print(name, len(icc), "bytes")

    # Alpha passes through the conversion untouched.
    p3 = PROFILES["display_p3"]
    alpha = Image.linear_gradient("L").resize(base.size)
    rgba = base.copy()
    rgba.putalpha(alpha)
    rgba.save(os.path.join(out, "display_p3_alpha.png"), icc_profile=p3)
    converted = to_srgb(base, p3, "RGB")
    converted.putalpha(alpha)
    with open(os.path.join(out, "display_p3_alpha.srgb.raw"), "wb") as f:
        f.write(converted.tobytes())

    # The profile in every other container, for extraction.
    base.save(os.path.join(out, "display_p3.jpg"), quality=90, icc_profile=p3)
    base.save(os.path.join(out, "display_p3_progressive.jpg"), quality=90, progressive=True, icc_profile=p3)
    base.save(os.path.join(out, "display_p3.webp"), quality=90, icc_profile=p3)
    base.save(os.path.join(out, "display_p3.tif"), icc_profile=p3)
    with tempfile.TemporaryDirectory() as tmp:
        png = os.path.join(tmp, "card.png")
        icc_path = os.path.join(tmp, "p3.icc")
        base.save(png)
        with open(icc_path, "wb") as f:
            f.write(p3)
        subprocess.run(
            ["avifenc", "-s", "8", "-q", "80", "--icc", icc_path, png, os.path.join(out, "display_p3.avif")],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    # Sanity: every container hands the profile back to Pillow intact.
    for ext in ("png", "jpg", "webp", "tif"):
        assert Image.open(os.path.join(out, "display_p3." + ext)).info.get("icc_profile") == p3, ext
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
