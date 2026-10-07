#!/usr/bin/env bash
# Run every fuzz target briefly: a regression gate, not a discovery campaign.
# `-max_total_time` bounds each run; any crash fails it. Needs cargo-fuzz and a
# nightly toolchain.
#
#   scripts/fuzz.sh [seconds-per-target]   (default 120)
#
# `--target` is the nightly toolchain's host, passed explicitly: cargo-fuzz
# otherwise defaults to the triple it was itself built for, and a prebuilt
# cargo-fuzz (as CI installs) is static musl — no std installed for it, and
# the address sanitizer refuses a statically linked libc.
set -euo pipefail

seconds="${1:-120}"
target="$(rustc +nightly -vV | sed -n 's/^host: //p')"
cd "$(dirname "$0")/../fuzz"

for fuzz_target in png_decode inflate png_roundtrip jpeg_decode jpeg_roundtrip avif_decode; do
    cargo +nightly fuzz run --target "$target" "$fuzz_target" -- -max_total_time="$seconds"
done
