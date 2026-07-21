#!/usr/bin/env bash
# Verify that our WebP output is accepted by libwebp, and decodes to the exact
# pixels we put in.
#
# Our own decoder cannot validate our own encoder here: both sit on the same
# wrapped crate, so a fault in it would round-trip perfectly and still produce
# a file nothing else reads. `tests/reference.rs` checks the decode direction
# against libwebp; this checks the encode direction.
#
# The comparison is exact, because our encoder writes lossless WebP. There is
# no tolerance for a bug to hide in.
set -euo pipefail

dir="$(mktemp -d)"
trap 'rm -rf "$dir"' EXIT

OTF_EMIT_DIR="$dir" cargo test -p otf-pixels-codec-webp --test interop_emit -- --nocapture >/dev/null

python3 - "$dir" <<'PYEOF'
import glob, os, re, sys

try:
    from PIL import Image
except ImportError:
    sys.exit("Pillow is required: pip install pillow")

# What our encoder was given -> what libwebp should hand back. WebP has no
# greyscale mode, so a single channel legitimately returns as three.
EXPECTED = {"rgb": "RGB", "rgba": "RGBA", "gray": "RGB", "graya": "RGBA"}

directory = sys.argv[1]
ok = failed = 0

for path in sorted(glob.glob(os.path.join(directory, "*.webp"))):
    name = os.path.basename(path)
    match = re.match(r"(\d+)x(\d+)_(rgb|rgba|gray|graya)\.webp", name)
    if not match:
        sys.exit(f"unexpected fixture name: {name}")
    width, height, kind = int(match.group(1)), int(match.group(2)), match.group(3)
    mode = EXPECTED[kind]

    source = open(path[:-5] + ".raw", "rb").read()
    try:
        image = Image.open(path)
        image.load()
        if image.size != (width, height):
            raise ValueError(f"decoded as {image.size}, expected {(width, height)}")
        actual = image.convert(mode).tobytes()
    except Exception as error:
        failed += 1
        print(f"REJECTED {name}: {error}")
        continue

    # Rebuild what the source means in the mode libwebp returns, so greyscale
    # is compared as the RGB it necessarily became.
    channels = {"rgb": 3, "rgba": 4, "gray": 1, "graya": 2}[kind]
    expected = bytearray()
    for i in range(width * height):
        pixel = source[i * channels:(i + 1) * channels]
        if kind == "gray":
            expected += bytes([pixel[0]] * 3)
        elif kind == "graya":
            expected += bytes([pixel[0]] * 3) + bytes([pixel[1]])
        else:
            expected += pixel

    if actual == bytes(expected):
        ok += 1
    else:
        differing = sum(1 for a, b in zip(actual, expected) if a != b)
        failed += 1
        print(f"MISMATCH {name}: {differing} of {len(expected)} bytes differ")

if ok == 0:
    print("no WebP files were emitted; the test did not run")
    sys.exit(1)
print(f"libwebp accepted {ok}/{ok + failed} of our WebP files with exact pixels")
sys.exit(1 if failed else 0)
PYEOF
