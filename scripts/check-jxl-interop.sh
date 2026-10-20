#!/usr/bin/env bash
# Verify that our JPEG XL output decodes in libjxl to exactly the pixels we
# put in: every sample file of the emit test, through `djxl` as PAM.
set -euo pipefail

command -v djxl >/dev/null || { echo "djxl is required (libjxl-tools)"; exit 1; }
dir="$(mktemp -d)"
trap 'rm -rf "$dir"' EXIT

OTF_EMIT_DIR="$dir" cargo test -p otf-pixels-codec-jxl --test emit -- --nocapture >/dev/null

python3 -I - "$dir" <<'PYEOF'
import glob, os, struct, subprocess, sys

CHANNELS = {"gray8": 1, "graya8": 2, "rgb8": 3, "rgba8": 4, "gray16": 1, "rgb16": 3, "rgba16": 4}
ok = failed = 0
for path in sorted(glob.glob(os.path.join(sys.argv[1], "*.jxl"))):
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
print(f"{ok} files decode exactly in libjxl, {failed} failed")
sys.exit(1 if failed else 0)
PYEOF
