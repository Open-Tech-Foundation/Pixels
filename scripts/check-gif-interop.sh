#!/usr/bin/env bash
# Verify that our GIF output is accepted by libgif, and decodes to the pixels
# we put in.
#
# Our own decoder cannot validate our own encoder: a shared misreading of the
# specification round-trips perfectly and is still wrong. `tests/reference.rs`
# checks the decode direction against libgif's output; this checks the encode
# direction, by handing libgif our files.
set -euo pipefail

dir="$(mktemp -d)"
trap 'rm -rf "$dir"' EXIT

OTF_EMIT_DIR="$dir" cargo test -p otf-pixels-codec-gif --test interop_emit -- --nocapture >/dev/null

python3 - "$dir" <<'PY'
import glob, os, sys

try:
    from PIL import Image
except ImportError:
    sys.exit("Pillow is required: pip install pillow")

directory = sys.argv[1]
ok = failed = 0

for path in sorted(glob.glob(os.path.join(directory, "*.gif"))):
    name = os.path.basename(path)
    expected = open(path[:-4] + ".rgb", "rb").read()
    try:
        image = Image.open(path)
        image.load()
        actual = image.convert("RGB").tobytes()
    except Exception as error:
        failed += 1
        print(f"REJECTED {name}: {error}")
        continue

    if actual == expected:
        ok += 1
    else:
        # A palette image we said was exactly representable must come back
        # exactly; anything else is a real disagreement, not rounding.
        differing = sum(1 for a, b in zip(actual, expected) if a != b)
        failed += 1
        print(
            f"MISMATCH {name}: {differing} of {len(expected)} bytes differ "
            f"(decoded as {image.mode} {image.size})"
        )

if ok == 0:
    print("no GIFs were emitted; the test did not run")
    sys.exit(1)
print(f"libgif accepted {ok}/{ok + failed} of our GIFs with matching pixels")
sys.exit(1 if failed else 0)
PY
