#!/bin/sh
set -e
N="Qwen3.8-27B-AWQ-INT4"; D="/opt/models/$N"
if [ ! -f "$D/.omp-ready" ]; then
  echo "[deploy-preflight] snapshot_download cyankiwi/Qwen3.8-27B-AWQ-INT4@63768c10df38c0395e12ef49edac1bd539eaeeea -> $D"
  timeout 3600 python3 -c 'import sys,huggingface_hub as h;h.snapshot_download(sys.argv[1],revision=sys.argv[2],local_dir=sys.argv[3])' "cyankiwi/Qwen3.8-27B-AWQ-INT4" "63768c10df38c0395e12ef49edac1bd539eaeeea" "$D"
  touch "$D/.omp-ready"
fi
exec /opt/entrypoint.sh "$@"
