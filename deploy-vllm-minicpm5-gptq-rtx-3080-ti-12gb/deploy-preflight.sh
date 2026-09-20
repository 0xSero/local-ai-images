#!/bin/sh
set -e
N="MiniCPM5-2B-GPTQ"; D="/opt/models/$N"
if [ ! -f "$D/.omp-ready" ]; then
  echo "[deploy-preflight] snapshot_download openbmb/MiniCPM5-2B-GPTQ@48d57c14fba6b1f86109e1a54fafb0848c163f04 -> $D"
  timeout 3600 python3 -c 'import sys,huggingface_hub as h;h.snapshot_download(sys.argv[1],revision=sys.argv[2],local_dir=sys.argv[3])' "openbmb/MiniCPM5-2B-GPTQ" "48d57c14fba6b1f86109e1a54fafb0848c163f04" "$D"
  touch "$D/.omp-ready"
fi
exec /opt/entrypoint.sh "$@"
