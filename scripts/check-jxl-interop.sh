#!/usr/bin/env bash
# Verify our JPEG XL output in libjxl: every lossless sample of the emit test
# must decode through `djxl` to exactly the pixels we put in, and every lossy
# one must decode and stay above a PSNR floor at distance 1.
set -euo pipefail

command -v djxl >/dev/null || { echo "djxl is required (libjxl-tools)"; exit 1; }
dir="$(mktemp -d)"
trap 'rm -rf "$dir"' EXIT

OTF_EMIT_DIR="$dir" cargo test --release -p otf-pixels-codec-jxl --test emit -- --nocapture >/dev/null

python3 -I - "$dir" <<'PYEOF'
import glob, os, struct, subprocess, sys

CHANNELS = {"gray8": 1, "graya8": 2, "rgb8": 3, "rgba8": 4, "gray16": 1, "rgb16": 3, "rgba16": 4}
ok = failed = 0
import math

# Distance 1 on these smooth scenes; a tiny image of steep ramps is the
# hardest case.
FLOOR = 32.0
for path in sorted(p for p in glob.glob(os.path.join(sys.argv[1], "*_lossy.jxl"))
                   if not os.path.basename(p).startswith("anim_")):
    name = os.path.basename(path)
    ppm = path[:-4] + ".ppm"
    run = subprocess.run(["djxl", path, ppm], capture_output=True, text=True)
    try:
        if run.returncode != 0:
            raise ValueError("djxl failed: " + (run.stderr or run.stdout).strip())
        data = open(ppm, "rb").read()
        # P6 header: magic, width, height, maxval, then one whitespace byte.
        fields, at = [], 0
        while len(fields) < 4:
            while data[at:at + 1].isspace():
                at += 1
            start = at
            while not data[at:at + 1].isspace():
                at += 1
            fields.append(data[start:at])
        pixels = data[at + 1:]
        source = open(path[:-4] + ".rgb", "rb").read()
        if len(pixels) != len(source):
            raise ValueError(f"{len(pixels)} bytes decoded, expected {len(source)}")
        mse = sum((a - b) ** 2 for a, b in zip(pixels, source)) / len(source)
        psnr = 99.0 if mse == 0 else 10 * math.log10(255 * 255 / mse)
        print(f"  {name}: {psnr:.1f} dB")
        if psnr < FLOOR:
            raise ValueError(f"PSNR {psnr:.1f} dB is under {FLOOR}")
        ok += 1
    except ValueError as error:
        failed += 1
        print(f"FAIL {name}: {error}")

for path in sorted(p for p in glob.glob(os.path.join(sys.argv[1], "*.jxl"))
                   if not p.endswith("_lossy.jxl") and not os.path.basename(p).startswith("anim_")):
    name = os.path.basename(path)
    kind = name.rsplit("_", 1)[1][:-4]
    wide = kind.endswith("16")
    pam = path[:-4] + ".pam"
    run = subprocess.run(["djxl", path, pam, "--bits_per_sample", "16" if wide else "8"],
                         capture_output=True, text=True)
    try:
        if run.returncode != 0:
            raise ValueError("djxl failed: " + (run.stderr or run.stdout).strip())
        data = open(pam, "rb").read()
        end = data.index(b"ENDHDR\n") + 7
        header = dict(line.split(" ", 1) for line in data[:end].decode().splitlines()[1:-1])
        if int(header["DEPTH"]) != CHANNELS[kind]:
            raise ValueError(f"{header['DEPTH']} channels, expected {CHANNELS[kind]}")
        pixels = data[end:]
        raw = open(path[:-4] + ".raw", "rb").read()
        if wide:
            pixels = struct.unpack(">%dH" % (len(pixels) // 2), pixels)
            raw = struct.unpack("=%dH" % (len(raw) // 2), raw)
        if list(pixels) != list(raw):
            raise ValueError("libjxl decodes different pixels")
        ok += 1
    except ValueError as error:
        failed += 1
        print(f"FAIL {name}: {error}")
# Animations: every displayed frame through `djxl --output_frames`, and the
# durations and loop count `jxlinfo` reports.
import re
for path in sorted(glob.glob(os.path.join(sys.argv[1], "anim_*.jxl"))):
    name = os.path.basename(path)
    width, height, channels, bits, count, loops, *durations = map(int, open(path[:-4] + ".meta").read().split())
    expected = open(path[:-4] + ".frames", "rb").read()
    lossy = name.endswith("_lossy.jxl")
    try:
        out = path[:-4] + ".pam"
        run = subprocess.run(["djxl", path, out, "--output_frames", "--bits_per_sample", str(bits)],
                             capture_output=True, text=True)
        if run.returncode != 0:
            raise ValueError("djxl failed: " + (run.stderr or run.stdout).strip())
        size = width * height * channels * (bits // 8)
        k = 0
        while os.path.exists(f"{out}-{k}.pam"):
            data = open(f"{out}-{k}.pam", "rb").read()
            pixels = data[data.index(b"ENDHDR\n") + 7:]
            want = expected[k * size:(k + 1) * size]
            if bits == 16:
                pixels = struct.unpack(">%dH" % (len(pixels) // 2), pixels)
                want = struct.unpack("=%dH" % (len(want) // 2), want)
            if len(pixels) != len(want):
                raise ValueError(f"frame {k}: {len(pixels)} samples, expected {len(want)}")
            worst = max((abs(a - b) for a, b in zip(pixels, want)), default=0)
            if worst > (1 if lossy else 0):
                raise ValueError(f"frame {k}: off by {worst}")
            k += 1
        if k != count:
            raise ValueError(f"{k} frames, expected {count}")
        info = subprocess.run(["jxlinfo", path], capture_output=True, text=True).stdout
        seen = [round(float(d)) for d in re.findall(r"duration: ([0-9.]+) ms", info)]
        if seen != durations:
            raise ValueError(f"durations {seen}, expected {durations}")
        # Forever reads "(looping)"; a finite count, as the length of every
        # play together.
        length = re.search(r"Animation length: ([0-9.]+) seconds", info)
        if loops == 0:
            looping = "(looping)" in info
        else:
            looping = length is not None and abs(float(length.group(1)) - sum(durations) * loops / 1000) < 0.001
        if not looping:
            raise ValueError(f"jxlinfo does not report {loops} loops: {info.strip().splitlines()[-1]}")
        ok += 1
    except ValueError as error:
        failed += 1
        print(f"FAIL {name}: {error}")
print(f"{ok} files decode as they must in libjxl, {failed} failed")
sys.exit(1 if failed else 0)
PYEOF
