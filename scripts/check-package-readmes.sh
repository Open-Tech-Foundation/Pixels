#!/usr/bin/env bash
# Verify that every published crate declares a README and ships it, so its
# crates.io page is not blank, and that the README has no relative links.
#
# `cargo package --list` is what `cargo publish` would upload, without building
# anything. A crate whose manifest declares `readme` lists it as `README.md`
# wherever the file lives; one that declares none lists no README at all.
#
# A relative link works on GitHub but breaks on crates.io, which resolves it
# against the crate's own directory: `otf-pixels` ships the root README from
# `../../README.md`, so `docs/EMBEDDING.md` there pointed nowhere. Links must
# be absolute URLs or in-page anchors.
set -euo pipefail

cd "$(dirname "$0")/.."
status=0

for manifest in crates/*/Cargo.toml; do
    crate="$(sed -n 's/^name = "\(.*\)"$/\1/p' "$manifest" | head -n 1)"
    if ! grep -q '^readme = ' "$manifest"; then
        echo "✗ $crate: no \`readme\` in $manifest" >&2
        status=1
        continue
    fi
    if ! cargo package --list --allow-dirty --quiet -p "$crate" | grep -qx 'README.md'; then
        echo "✗ $crate: README.md is not in the package" >&2
        status=1
        continue
    fi
    readme="$(dirname "$manifest")/$(sed -n 's/^readme = "\(.*\)"$/\1/p' "$manifest")"
    relative="$(grep -noE '\]\([^)[:space:]]+\)' "$readme" | grep -vE '\]\((https?://|mailto:|#)' || true)"
    if [ -n "$relative" ]; then
        echo "✗ $crate: relative links in $readme break on crates.io:" >&2
        echo "$relative" | sed 's/^/    /' >&2
        status=1
        continue
    fi
    echo "✓ $crate"
done

exit "$status"
