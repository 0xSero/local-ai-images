#!/bin/sh
set -e
N="Qwen3.8-27B-EXL3-SC2bpw-H3-V3"; D="/opt/models/$N"
if [ ! -f "$D/.omp-ready" ]; then
  echo "[deploy-preflight] snapshot_download turboderp/Qwen3.8-27B-exl3@e5e1f4b37c77641e465e24adf5f44d01bdd48de2 -> $D"
  timeout 3600 python3 -c 'import sys,huggingface_hub as h;h.snapshot_download(sys.argv[1],revision=sys.argv[2],local_dir=sys.argv[3])' "turboderp/Qwen3.8-27B-exl3" "e5e1f4b37c77641e465e24adf5f44d01bdd48de2" "$D"
  touch "$D/.omp-ready"
fi
cd /app
exec python3 main.py
