#!/bin/sh
set -e
N="Qwen3.8-27B-EXL3-SC4bpw-H5"; D="/opt/models/$N"
if [ ! -f "$D/.omp-ready" ]; then
  echo "[deploy-preflight] snapshot_download turboderp/Qwen3.8-27B-exl3@4acd9ad5af224ec9e8815a54d71a033f378a7e9d -> $D"
  timeout 3600 python3 -c 'import sys,huggingface_hub as h;h.snapshot_download(sys.argv[1],revision=sys.argv[2],local_dir=sys.argv[3])' "turboderp/Qwen3.8-27B-exl3" "4acd9ad5af224ec9e8815a54d71a033f378a7e9d" "$D"
  touch "$D/.omp-ready"
fi
cd /app
exec python3 main.py
