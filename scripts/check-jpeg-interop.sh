#!/usr/bin/env bash
# Verify that our JPEG output is accepted by libjpeg, and decodes to
# approximately the pixels we put in.
#
# Our own decoder cannot validate our own encoder: a shared misreading of the
# specification round-trips perfectly and is still wrong. `tests/reference.rs`
# checks the decode direction against libjpeg's output; this checks the encode
# direction, by handing libjpeg our files.
#
# "Approximately" is not a weakening. JPEG is lossy by construction, so the
# question is not whether the pixels match but whether they differ by only as
# much as the quantization at that quality can account for. A container bug, a
# missing byte-stuff or a mis-ordered MCU does not produce a small error — it
# produces a rejected file or a scrambled one.
#
# How much *is* the quantization at a given quality worth? Rather than guess a
# fixed tolerance — which is wrong at both ends, because a steep gradient in a
# 7x3 image loses far more to 4:2:0 than the same gradient across 64x48 does —
# the same raster is encoded by libjpeg at the same settings and the two errors
# are compared. The question becomes "is our loss in the same league as the
# reference encoder's", which is scale-free and is the thing actually worth
# asserting.
set -euo pipefail

dir="$(mktemp -d)"
trap 'rm -rf "$dir"' EXIT

OTF_EMIT_DIR="$dir" cargo test -p otf-pixels-codec-jpeg --test interop_emit -- --nocapture >/dev/null

python3 - "$dir" <<'PYEOF'
import glob, io, os, re, sys

try:
    from PIL import Image
except ImportError:
    sys.exit("Pillow is required: pip install pillow")

# How much worse than libjpeg's own loss we tolerate. Some margin is expected
# and legitimate: libjpeg upsamples chroma with a triangle filter where we
# replicate, and it rounds the quantization tables slightly differently. A
# margin this size cannot hide a structural bug, which shows up as a rejected
# file or an error several times the reference, not as a fraction more.
SCALE = 1.6
FLOOR = 1.5

SUBSAMPLING = {"444": 0, "422": 1, "420": 2}

directory = sys.argv[1]
ok = failed = 0
worst = (0.0, "")


def mean_error(a: bytes, b: bytes) -> float:
    return sum(abs(x - y) for x, y in zip(a, b)) / max(len(a), 1)


for path in sorted(glob.glob(os.path.join(directory, "*.jpg"))):
    name = os.path.basename(path)
    match = re.match(r"(\d+)x(\d+)_(rgb|gray)_(\d+)_q(\d+)\.jpg", name)
    if not match:
        sys.exit(f"unexpected fixture name: {name}")
    width, height, kind, sub, quality = match.groups()
    width, height, quality = int(width), int(height), int(quality)
    mode = "L" if kind == "gray" else "RGB"

    expected = open(path[:-4] + ".raw", "rb").read()
    try:
        image = Image.open(path)
        image.load()
        if image.size != (width, height):
            raise ValueError(f"decoded as {image.size}, expected {(width, height)}")
        if image.mode != mode:
            raise ValueError(f"decoded as {image.mode}, expected {mode}")
        actual = image.tobytes()
    except Exception as error:
        failed += 1
        print(f"REJECTED {name}: {error}")
        continue

    if len(actual) != len(expected):
        failed += 1
        print(f"MISMATCH {name}: {len(actual)} bytes, expected {len(expected)}")
        continue

    # The same pixels through libjpeg's encoder at the same settings, so the
    # comparison is against what this image costs rather than a guess.
    source = Image.frombytes(mode, (width, height), expected)
    buffer = io.BytesIO()
    options = {"quality": quality}
    if mode == "RGB":
        options["subsampling"] = SUBSAMPLING[sub]
    source.save(buffer, "JPEG", **options)
    buffer.seek(0)
    with Image.open(buffer) as theirs:
        theirs.load()
        reference = theirs.convert(mode).tobytes()

    ours = mean_error(actual, expected)
    libjpeg = mean_error(reference, expected)
    limit = libjpeg * SCALE + FLOOR

    if ours <= limit:
        ok += 1
    else:
        failed += 1
        print(
            f"MISMATCH {name}: mean error {ours:.2f} against libjpeg's "
            f"{libjpeg:.2f} (limit {limit:.2f})"
        )
    ratio = ours / max(libjpeg, 0.01)
    if ratio > worst[0]:
        worst = (ratio, f"{name}: {ours:.2f} vs {libjpeg:.2f}")

if ok == 0:
    print("no JPEGs were emitted; the test did not run")
    sys.exit(1)
print(f"libjpeg accepted {ok}/{ok + failed} of our JPEGs within tolerance")
print(f"  furthest from libjpeg: {worst[1]}")
sys.exit(1 if failed else 0)
PYEOF
