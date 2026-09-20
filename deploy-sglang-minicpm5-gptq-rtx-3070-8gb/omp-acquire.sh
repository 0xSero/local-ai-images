#!/bin/sh
# /opt/omp-acquire — pinned model acquisition. Idempotent, fail-closed, revision-validated.
# Progress -> stdout+stderr (SSH capture IS the log) and /opt/logs/deploy-preflight.log.
# ANY failure -> nonzero exit + NO .omp-ready marker. Marker stores the pinned revision.
set -e
N="MiniCPM5-2B-GPTQ"; D="/opt/models/$N"; REPO="openbmb/MiniCPM5-2B-GPTQ"; REV="48d57c14fba6b1f86109e1a54fafb0848c163f04"; LOG="/opt/logs/deploy-preflight.log"
mkdir -p /opt/logs "$D"
log() { echo "[omp-acquire] $*"; echo "[omp-acquire] $*" >>"$LOG" 2>/dev/null || true; }
if [ -f "$D/.omp-ready" ] && [ "$(cat "$D/.omp-ready" 2>/dev/null)" = "$REV" ]; then
  log "model already at pinned revision $REV"; exit 0
fi
log "acquiring $REPO@$REV -> $D"; rm -f "$D/.omp-ready"
if timeout "${OMP_ACQUIRE_TIMEOUT:-3600}" /opt/sglang/bin/python3 -c \
    'import sys,huggingface_hub as h;h.snapshot_download(sys.argv[1],revision=sys.argv[2],local_dir=sys.argv[3])' \
    "$REPO" "$REV" "$D" >/tmp/omp-acquire.out 2>&1; then
  cat /tmp/omp-acquire.out; cat /tmp/omp-acquire.out >>"$LOG" 2>/dev/null || true
  printf '%s' "$REV" > "$D/.omp-ready"; log "acquired pinned revision $REV; marker written"; exit 0
else
  rc=$?; cat /tmp/omp-acquire.out; cat /tmp/omp-acquire.out >>"$LOG" 2>/dev/null || true
  log "ACQUIRE FAILED rc=$rc (fail-closed: no marker written)"; exit 3
fi
