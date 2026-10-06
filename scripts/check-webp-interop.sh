#!/usr/bin/env bash
# Verify that our WebP output is accepted by libwebp, and decodes as it must.
#
# Our own decoder cannot validate our own encoder: a misreading of the format
# shared by both would round-trip perfectly and still produce files nothing
# else reads. `tests/{lossless,lossy,animated}.rs` check the decode direction
# against libwebp; this checks the encode direction, three ways:
#
# - Lossless files (`WxH_kind.webp`) must decode to exactly the pixels we put
#   in. There is no tolerance for a bug to hide in.
# - Lossy files (`WxH_kind_lossy_qN.webp`) must decode, in libwebp, to exactly
#   what our decoder makes of them (VP8 decoding is exact by specification),
#   with their alpha, which is coded losslessly, exactly as given.
# - VP8 option variants (`vp8_*.webp`), written with token partitions, the
#   simple filter, sharpness, segments and filter deltas that no Pillow-made
#   file has, must likewise decode in libwebp exactly as in ours.
set -euo pipefail

dir="$(mktemp -d)"
trap 'rm -rf "$dir"' EXIT

OTF_EMIT_DIR="$dir" cargo test -p otf-pixels-codec-webp --test interop_emit -- --nocapture >/dev/null
OTF_EMIT_DIR="$dir" cargo test -p otf-pixels-codec-webp --lib emit_option_variants -- --nocapture >/dev/null

python3 - "$dir" <<'PYEOF'
import glob, os, re, sys

try:
    from PIL import Image
except ImportError:
    sys.exit("Pillow is required: pip install pillow")

# What our encoder was given -> what libwebp should hand back. WebP has no
# greyscale mode, so a single channel legitimately returns as three.
EXPECTED = {"rgb": "RGB", "rgba": "RGBA", "gray": "RGB", "graya": "RGBA"}
CHANNELS = {"rgb": 3, "rgba": 4, "gray": 1, "graya": 2}

directory = sys.argv[1]
ok = failed = 0


def load(path, mode=None):
    image = Image.open(path)
    image.load()
    return image, (image.convert(mode) if mode else image).tobytes()


def as_rgb(source, kind, width, height):
    """The source in the mode libwebp returns: greyscale as the RGB it became."""
    channels = CHANNELS[kind]
    out = bytearray()
    for i in range(width * height):
        pixel = source[i * channels:(i + 1) * channels]
        if kind == "gray":
            out += bytes([pixel[0]] * 3)
        elif kind == "graya":
            out += bytes([pixel[0]] * 3) + bytes([pixel[1]])
        else:
            out += pixel
    return bytes(out)


for path in sorted(glob.glob(os.path.join(directory, "*.webp"))):
    name = os.path.basename(path)
    base = path[:-5]
    try:
        if name.startswith("vp8_"):
            _, actual = load(path)
            ours = open(base + ".ours", "rb").read()
            if actual != ours:
                raise ValueError("libwebp decodes it differently from us")
        else:
            match = re.match(r"(\d+)x(\d+)_(rgb|rgba|gray|graya)(_lossy_q\d+)?\.webp", name)
            if not match:
                sys.exit(f"unexpected fixture name: {name}")
            width, height, kind, lossy = int(match.group(1)), int(match.group(2)), match.group(3), match.group(4)
            mode = EXPECTED[kind]
            image, actual = load(path, mode)
            if image.size != (width, height):
                raise ValueError(f"decoded as {image.size}, expected {(width, height)}")
            expected = as_rgb(open(base + ".raw", "rb").read(), kind, width, height)
            if not lossy:
                if actual != expected:
                    differing = sum(1 for a, b in zip(actual, expected) if a != b)
                    raise ValueError(f"{differing} of {len(expected)} bytes differ")
            else:
                ours = open(base + ".ours", "rb").read()
                if actual != as_rgb(ours, "rgba" if mode == "RGBA" else "rgb", width, height):
                    raise ValueError("libwebp decodes it differently from us")
                if mode == "RGBA" and actual[3::4] != expected[3::4]:
                    raise ValueError("the alpha channel did not come back exact")
    except Exception as error:
        failed += 1
        print(f"FAILED {name}: {error}")
        continue
    ok += 1

if ok == 0:
    print("no WebP files were emitted; the test did not run")
    sys.exit(1)
print(f"libwebp accepted {ok}/{ok + failed} of our WebP files with the pixels they must have")
sys.exit(1 if failed else 0)
PYEOF
