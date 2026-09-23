#!/usr/bin/env bash
# Vendor a source snapshot of 0xSero/sglang-exl3 (private) into ./src at one commit, so the release-image workflow
# builds without repository access. Usage: ./sync-src.sh [path-to-sglang-exl3-checkout] [commit]
set -euo pipefail
SRC=${1:-$HOME/rtx-3090/sglang-exl3}; COMMIT=${2:-$(git -C "$SRC" rev-parse HEAD)}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rm -rf "$HERE/src"; mkdir -p "$HERE/src"
git -C "$SRC" archive "$COMMIT" | tar -x -C "$HERE/src"
rm -rf "$HERE/src/tests" "$HERE/src/docker" "$HERE/src/scripts" "$HERE/src/notes"
echo "$COMMIT" > "$HERE/VERSION"
echo "vendored sglang-exl3 @ $COMMIT into $HERE/src"
