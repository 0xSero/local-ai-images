#!/bin/sh
set -e
N="Ornith-1.5-9B"; D="/opt/models/$N"
if [ ! -f "$D/.omp-ready" ]; then
  echo "[deploy-preflight] snapshot_download ornith-ai/Ornith-1.5-9B@489cb97981b8654bcfcf30ce1f94ed1b62e07b53 -> $D"
  timeout 3600 python3 -c 'import sys,huggingface_hub as h;h.snapshot_download(sys.argv[1],revision=sys.argv[2],local_dir=sys.argv[3])' "ornith-ai/Ornith-1.5-9B" "489cb97981b8654bcfcf30ce1f94ed1b62e07b53" "$D"
  touch "$D/.omp-ready"
fi
exec /opt/entrypoint.sh "$@"
