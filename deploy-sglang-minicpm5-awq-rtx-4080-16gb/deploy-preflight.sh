#!/bin/sh
set -e
N="MiniCPM5-2B-AWQ-INT4"; D="/opt/models/$N"
if [ ! -f "$D/.omp-ready" ]; then
  echo "[deploy-preflight] snapshot_download cyankiwi/MiniCPM5-2B-AWQ-INT4@bc59eebcb2e12f5e1e714c57a4b6b7527f6cd845 -> $D"
  timeout 3600 /opt/sglang/bin/python3 -c 'import sys,huggingface_hub as h;h.snapshot_download(sys.argv[1],revision=sys.argv[2],local_dir=sys.argv[3])' "cyankiwi/MiniCPM5-2B-AWQ-INT4" "bc59eebcb2e12f5e1e714c57a4b6b7527f6cd845" "$D"
  touch "$D/.omp-ready"
fi
exec /opt/entrypoint.sh "$@"
