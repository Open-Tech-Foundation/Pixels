#!/usr/bin/env bash
# Verify that our PNG output is accepted by libpng, and decodes to the pixels
# we put in.
#
# Our own decoder cannot validate our own encoder: a shared misreading of the
# PNG specification round-trips perfectly and is still wrong. `tests/pngsuite.rs`
# checks the decode direction against libpng's output; this checks the encode
# direction, by handing libpng our files.
set -euo pipefail

dir="$(mktemp -d)"
trap 'rm -rf "$dir"' EXIT

OTF_EMIT_DIR="$dir" cargo test -p otf-pixels-codec-png --test interop_emit -- --nocapture >/dev/null

python3 - "$dir" <<'PY'
import glob, os, sys

try:
    from PIL import Image
except ImportError:
    sys.exit("Pillow is required: pip install pillow")

directory = sys.argv[1]
ok = failed = 0

for path in sorted(glob.glob(os.path.join(directory, "*.png"))):
    name = os.path.basename(path)
    expected = open(path[:-4] + ".rgba", "rb").read()
    try:
        image = Image.open(path)
        image.load()
    except Exception as error:
        failed += 1
        print(f"REJECTED {name}: {error}")
        continue

    # Mode I;16 is Pillow's 16-bit greyscale, whose convert("RGBA") clips
    # instead of scaling. Narrow it here, as the reference script does.
    if image.mode.startswith("I"):
        actual = bytearray()
        for sample in image.getdata():
            grey = sample >> 8
            actual += bytes((grey, grey, grey, 255))
        actual = bytes(actual)
    else:
        actual = image.convert("RGBA").tobytes()

    if actual == expected:
        ok += 1
    else:
        failed += 1
        differing = sum(1 for a, b in zip(actual, expected) if a != b)
        print(
            f"MISMATCH {name}: {differing} of {len(expected)} bytes differ "
            f"(decoded as {image.mode} {image.size})"
        )

if ok == 0:
    print("no PNGs were emitted; the test did not run")
    sys.exit(1)
print(f"libpng accepted {ok}/{ok + failed} of our PNGs with matching pixels")
sys.exit(1 if failed else 0)
PY
