#!/bin/sh
set -e
N="Qwen3.8-27B-EXL3-SC5bpw-H6-V6"; D="/opt/models/$N"
if [ ! -f "$D/.omp-ready" ]; then
  echo "[deploy-preflight] snapshot_download turboderp/Qwen3.8-27B-exl3@f33f26d929e2b20ef21361145d582f5239e3831f -> $D"
  timeout 3600 python3 -c 'import sys,huggingface_hub as h;h.snapshot_download(sys.argv[1],revision=sys.argv[2],local_dir=sys.argv[3])' "turboderp/Qwen3.8-27B-exl3" "f33f26d929e2b20ef21361145d582f5239e3831f" "$D"
  touch "$D/.omp-ready"
fi
cd /app
exec python3 main.py
