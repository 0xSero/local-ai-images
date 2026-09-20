#!/bin/sh
set -e
N="Qwen3.8-Flash-Next-EXL3-205bpw-H4"; D="/opt/models/$N"
if [ ! -f "$D/.omp-ready" ]; then
  echo "[deploy-preflight] snapshot_download turboderp/Qwen3.8-Flash-Next-exl3@65c895314393431c09050b2e04e250836b3a6eb4 -> $D"
  timeout 3600 python3 -c 'import sys,huggingface_hub as h;h.snapshot_download(sys.argv[1],revision=sys.argv[2],local_dir=sys.argv[3])' "turboderp/Qwen3.8-Flash-Next-exl3" "65c895314393431c09050b2e04e250836b3a6eb4" "$D"
  touch "$D/.omp-ready"
fi
cd /app
exec python3 main.py
