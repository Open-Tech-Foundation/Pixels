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
    if name.startswith("anim_"):
        continue
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

# Animations: every frame as libgif composites it, with its delay and the
# loop count. Transparent pixels compare by alpha alone.
for path in sorted(glob.glob(os.path.join(directory, "anim_*.gif"))):
    name = os.path.basename(path)
    width, height, count, loop, *delays = map(int, open(path[:-4] + ".meta").read().split())
    expected = open(path[:-4] + ".rgba", "rb").read()
    size = width * height * 4
    problems = []
    try:
        image = Image.open(path)
        if getattr(image, "n_frames", 1) != count:
            problems.append(f"{image.n_frames} frames, expected {count}")
        if image.info.get("loop", None) != loop:
            problems.append(f"loop {image.info.get('loop')}, expected {loop}")
        for k in range(min(count, getattr(image, "n_frames", 1))):
            image.seek(k)
            if image.info.get("duration") != delays[k]:
                problems.append(f"frame {k} lasts {image.info.get('duration')} ms, expected {delays[k]}")
            actual = image.convert("RGBA").tobytes()
            want = expected[k * size:(k + 1) * size]
            for i in range(0, size, 4):
                a, w = actual[i:i + 4], want[i:i + 4]
                if (a[3] == 0) != (w[3] == 0) or (w[3] and a[:3] != w[:3]):
                    problems.append(f"frame {k} pixel {i // 4}: {tuple(a)} vs {tuple(w)}")
                    break
    except Exception as error:
        problems.append(f"rejected: {error}")
    if problems:
        failed += 1
        print(f"MISMATCH {name}: " + "; ".join(problems))
    else:
        ok += 1

if ok == 0:
    print("no GIFs were emitted; the test did not run")
    sys.exit(1)
print(f"libgif accepted {ok}/{ok + failed} of our GIFs with matching pixels")
sys.exit(1 if failed else 0)
PY
