#!/usr/bin/env bash
# Verify that our DEFLATE output is accepted by the reference zlib.
#
# Our own inflate cannot validate our own deflate: a shared misreading of
# RFC 1951 round-trips perfectly and is still wrong (ADR-0010). This drives
# the env-gated emit test and decompresses every stream with Python's zlib,
# which is the reference C implementation.
set -euo pipefail

dir="$(mktemp -d)"
trap 'rm -rf "$dir"' EXIT

OTF_EMIT_DIR="$dir" cargo test -p otf-pixels-codec-png --test interop_emit -- --nocapture >/dev/null

python3 - "$dir" <<'PY'
import glob, os, sys, zlib

directory = sys.argv[1]
ok = failed = 0
for path in sorted(glob.glob(os.path.join(directory, "*.zlib"))):
    expected = open(path[:-5] + ".raw", "rb").read()
    stream = open(path, "rb").read()
    name = os.path.basename(path)
    try:
        actual = zlib.decompress(stream)
    except Exception as error:
        failed += 1
        print(f"REJECTED {name}: {error}")
        continue
    if actual == expected:
        ok += 1
    else:
        failed += 1
        print(f"MISMATCH {name}: got {len(actual)} bytes, expected {len(expected)}")

if ok == 0:
    print("no streams were emitted; the test did not run")
    sys.exit(1)
print(f"reference zlib accepted {ok}/{ok + failed} of our streams")
sys.exit(1 if failed else 0)
PY
