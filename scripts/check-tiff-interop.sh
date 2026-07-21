#!/usr/bin/env bash
# Verify that our TIFF output is accepted by libtiff, and decodes to the
# pixels we put in.
#
# Our own decoder cannot validate our own encoder: a shared misreading of the
# specification round-trips perfectly and is still wrong.
set -euo pipefail

dir="$(mktemp -d)"
trap 'rm -rf "$dir"' EXIT

OTF_EMIT_DIR="$dir" cargo test -p otf-pixels-codec-tiff --test interop_emit -- --nocapture >/dev/null

python3 - "$dir" <<'PY'
import glob, os, sys

try:
    from PIL import Image
except ImportError:
    sys.exit("Pillow is required: pip install pillow")

directory = sys.argv[1]
ok = failed = 0

for path in sorted(glob.glob(os.path.join(directory, "*.tif"))):
    name = os.path.basename(path)
    expected = open(path[:-4] + ".raw", "rb").read()
    meta = open(path[:-4] + ".meta").read().split()
    width, height, fmt = int(meta[0]), int(meta[1]), meta[2]
    try:
        image = Image.open(path)
        image.load()
    except Exception as error:
        failed += 1
        print(f"REJECTED {name}: {error}")
        continue

    if (image.width, image.height) != (width, height):
        failed += 1
        print(f"MISMATCH {name}: libtiff read {image.size}, we wrote {(width, height)}")
        continue

    # Pillow narrows 16-bit *colour* on load (mode RGB), while reporting
    # 16-bit greyscale at full precision (mode I;16). So for 16-bit colour the
    # expected data is narrowed the same way before comparing — the same
    # asymmetry the PNG suite documents. Narrowing our side rather than
    # skipping the case keeps the check meaningful.
    actual = image.tobytes()
    if fmt in ("rgb16", "rgba16") and image.mode in ("RGB", "RGBA"):
        expected = bytes(expected[i] for i in range(1, len(expected), 2))

    if actual == expected:
        ok += 1
    else:
        failed += 1
        differing = sum(1 for a, b in zip(actual, expected) if a != b)
        print(
            f"MISMATCH {name} ({fmt}): {differing} of {len(expected)} bytes differ, "
            f"libtiff read mode {image.mode}"
        )

if ok == 0:
    print("no TIFFs were emitted; the test did not run")
    sys.exit(1)
print(f"libtiff accepted {ok}/{ok + failed} of our TIFFs with matching pixels")
sys.exit(1 if failed else 0)
PY
