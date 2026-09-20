#!/bin/sh
set -e
N="Qwen3.8-27B-EXL3-SC3bpw-H4-V4"; D="/opt/models/$N"
if [ ! -f "$D/.omp-ready" ]; then
  echo "[deploy-preflight] snapshot_download turboderp/Qwen3.8-27B-exl3@004a887127d8304ca2d5475d3a3c41f1761fdd27 -> $D"
  timeout 3600 python3 -c 'import sys,huggingface_hub as h;h.snapshot_download(sys.argv[1],revision=sys.argv[2],local_dir=sys.argv[3])' "turboderp/Qwen3.8-27B-exl3" "004a887127d8304ca2d5475d3a3c41f1761fdd27" "$D"
  touch "$D/.omp-ready"
fi
cd /app
exec python3 main.py
