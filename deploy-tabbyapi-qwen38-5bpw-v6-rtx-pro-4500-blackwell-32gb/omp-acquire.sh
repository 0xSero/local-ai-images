#!/bin/sh
# /opt/omp-acquire — pinned model acquisition. Idempotent, fail-closed, revision-validated.
# Progress -> stdout+stderr (SSH capture IS the log) and /opt/logs/deploy-preflight.log.
# ANY failure -> nonzero exit + NO .omp-ready marker. Marker stores the pinned revision.
set -e
N="Qwen3.8-27B-EXL3-SC5bpw-H6-V6"; D="/opt/models/$N"; REPO="turboderp/Qwen3.8-27B-exl3"; REV="f33f26d929e2b20ef21361145d582f5239e3831f"; LOG="/opt/logs/deploy-preflight.log"
mkdir -p /opt/logs "$D"
log() { echo "[omp-acquire] $*"; echo "[omp-acquire] $*" >>"$LOG" 2>/dev/null || true; }
if [ -f "$D/.omp-ready" ] && [ "$(cat "$D/.omp-ready" 2>/dev/null)" = "$REV" ]; then
  log "model already at pinned revision $REV"; exit 0
fi
log "acquiring $REPO@$REV -> $D"; rm -f "$D/.omp-ready"
# huggingface_hub lives in the image's own virtualenv, not in the system
# interpreter, so resolve an interpreter that can import it and fail
# closed if none can (the venv path is not on PATH by default).
PY=""
for cand in /opt/venv/bin/python3 /opt/sglang/bin/python3 /opt/vllm/bin/python3 python3; do
  if command -v "$cand" >/dev/null 2>&1 && "$cand" -c 'import huggingface_hub' 2>/dev/null; then PY="$cand"; break; fi
done
[ -n "$PY" ] || { log "no interpreter in this image can import huggingface_hub (fail-closed)"; exit 3; }
if timeout "${OMP_ACQUIRE_TIMEOUT:-3600}" "$PY" -c \
    'import sys,huggingface_hub as h;h.snapshot_download(sys.argv[1],revision=sys.argv[2],local_dir=sys.argv[3])' \
    "$REPO" "$REV" "$D" >/tmp/omp-acquire.out 2>&1; then
  cat /tmp/omp-acquire.out; cat /tmp/omp-acquire.out >>"$LOG" 2>/dev/null || true
  printf '%s' "$REV" > "$D/.omp-ready"; log "acquired pinned revision $REV; marker written"; exit 0
else
  rc=$?; cat /tmp/omp-acquire.out; cat /tmp/omp-acquire.out >>"$LOG" 2>/dev/null || true
  log "ACQUIRE FAILED rc=$rc (fail-closed: no marker written)"; exit 3
fi
