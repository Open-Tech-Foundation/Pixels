#!/usr/bin/env bash
# Verify that our AVIF output is read by libavif, and decodes as it must.
#
# Our own decoder cannot validate our own encoder: a misreading of the format
# shared by both would round-trip perfectly and still produce files nothing
# else reads. Two checks:
#
# - Bitstream: the AV1 streams the encoder's unit tests emit must decode, in
#   libaom's `aomdec`, to exactly the planes our decoder reconstructs. AV1
#   decoding is exact by specification, so any difference is a bug.
# - Container: every AVIF file must decode in `avifdec` with each AV1 decoder
#   it was built with (libaom, dav1d, libgav1) to our pixels: the colour within
#   2 (libyuv's YUV-to-RGB rounding differs from ours), the alpha exactly.
set -euo pipefail

dir="$(mktemp -d)"
trap 'rm -rf "$dir"' EXIT

OTF_EMIT_DIR="$dir" cargo test -q -p otf-pixels-codec-avif --lib emit_streams_for_interop >/dev/null
OTF_EMIT_DIR="$dir" cargo test -q -p otf-pixels-codec-avif --test interop_emit >/dev/null

ok=0
failed=0
for obu in "$dir"/*.obu; do
    base="${obu%.obu}"
    if aomdec --rawvideo -o "$base.aom" "$obu" 2>/dev/null && cmp -s "$base.aom" "$base.yuv"; then
        ok=$((ok + 1))
    else
        echo "FAILED $(basename "$obu"): aomdec decodes it differently from us"
        failed=$((failed + 1))
    fi
done
echo "aomdec agreed with us on $ok/$((ok + failed)) AV1 streams"

python3 - "$dir" <<'PYEOF' || failed=$((failed + 1))
import glob, os, re, subprocess, sys
from PIL import Image

directory = sys.argv[1]
codecs = [c for c in ("aom", "dav1d", "libgav1")
          if c in subprocess.run(["avifdec", "--version"], capture_output=True, text=True).stdout]
mode_of = {"rgb": "RGB", "rgba": "RGBA", "gray": "L"}
ok = failed = 0
for path in sorted(glob.glob(os.path.join(directory, "*.avif"))):
    name = os.path.basename(path)
    m = re.match(r"(\d+)x(\d+)_(rgb|rgba|gray)_q\d+\.avif", name)
    width, height, kind = int(m.group(1)), int(m.group(2)), m.group(3)
    ours = open(path[:-5] + ".ours", "rb").read()
    channels = {"rgb": 3, "rgba": 4, "gray": 1}[kind]
    for codec in codecs:
        png = f"{path}.{codec}.png"
        try:
            run = subprocess.run(["avifdec", "-c", codec, "-d", "8", path, png], capture_output=True, text=True)
            if run.returncode != 0:
                raise ValueError(run.stderr.strip().splitlines()[-1] if run.stderr.strip() else "avifdec failed")
            image = Image.open(png)
            if image.size != (width, height):
                raise ValueError(f"decoded as {image.size}")
            theirs = image.convert(mode_of[kind]).tobytes()
            worst = 0
            for i, (a, b) in enumerate(zip(ours, theirs)):
                d = abs(a - b)
                if kind == "rgba" and i % 4 == 3 and d:
                    raise ValueError(f"alpha differs at byte {i}: {a} vs {b}")
                worst = max(worst, d)
            if worst > 2:
                raise ValueError(f"colour differs by up to {worst}")
            ok += 1
        except Exception as error:
            failed += 1
            print(f"FAILED {name} [{codec}]: {error}")
print(f"avifdec ({', '.join(codecs)}) agreed with us on {ok}/{ok + failed} decodes")
sys.exit(1 if failed or not ok else 0)
PYEOF
[ "$failed" -eq 0 ]
