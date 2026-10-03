#!/bin/bash
# Build sovereign-trellis/ds41-exl3-spark:dev on a DGX Spark (arm64). Run in tmux on the host, e.g.
#   tmux new -d -s dsv41-build 'bash ~/dsv41-st/build/ctx/build.sh 2>&1 | tee -a ~/dsv41-st/build/build.log'
# Copy first from the Mac (current tree):  deploy/serve-spark/* -> ~/dsv41-st/build/ctx/ and deploy/serve/patch/ ->
#   ~/dsv41-st/build/ctx/patch/ (sync.sh does both; patch/ includes st_exl3_prefill.py and st_moe_ext/ sources)
# Steps: 1 fetch pinned sources (shallow) | 2 build wheels in the base container (CPU, memory-capped) |
#        3 docker build | 4 GPU validation (validate.py). STEPS=34 reruns only 3 and 4.
set -euo pipefail
W=${W:-$HOME/dsv41-st/build}
BASE=${BASE:-vllm/vllm-openai:v0.30.0}
IMG=${IMG:-sovereign-trellis/ds41-exl3-spark:dev}
STEPS=${STEPS:-1234}
VLLM_COMMIT=66c293578412417476f842c1da5805d3a3d959a8   # LIL vLLM r38 (pop-os production)
B12X_COMMIT=ce419b52681b7922bb0972d4b58b590a3fd005b2   # b12x pinned by r38
EXL3_COMMIT=d3739fd393337b1ff4d6c2a342b12f0c87a9592f   # turboderp exllamav3 v1.5.3 (MIT)
MAX_JOBS=${MAX_JOBS:-10}; MEM=${MEM:-88g}; CPUS=${CPUS:-16}
mkdir -p "$W/src" "$W/wheels" "$W/logs"
log() { echo "[$(date '+%F %T')] $*"; }

fetch() {  # dir url commit
  local d=$W/src/$1
  [ -d "$d/.git" ] || { git init -q "$d"; git -C "$d" remote add origin "$2"; }
  if [ "$(git -C "$d" rev-parse HEAD 2>/dev/null)" != "$3" ]; then
    git -C "$d" fetch -q --depth 1 origin "$3"; git -C "$d" checkout -q FETCH_HEAD
  fi
  log "src $1 $(git -C "$d" rev-parse HEAD)"
}

if [[ $STEPS == *1* ]]; then
  fetch vllm https://github.com/local-inference-lab/vllm.git $VLLM_COMMIT
  fetch b12x https://github.com/local-inference-lab/b12x.git $B12X_COMMIT
  fetch exllamav3 https://github.com/turboderp-org/exllamav3.git $EXL3_COMMIT
fi

if [[ $STEPS == *2* ]]; then
  log "wheels: MAX_JOBS=$MAX_JOBS mem=$MEM cpus=$CPUS"
  t0=$(date +%s)
  docker rm -f dsv41-wheels >/dev/null 2>&1 || true
  docker run --rm --name dsv41-wheels --memory "$MEM" --memory-swap "$MEM" --cpus "$CPUS" \
    -e MAX_JOBS="$MAX_JOBS" -e SKIP_VLLM="${SKIP_VLLM:-0}" -e SKIP_EXL3="${SKIP_EXL3:-0}" -e SKIP_STX="${SKIP_STX:-0}" \
    -v "$W/src:/src" -v "$W/wheels:/out" -v "$W/logs:/logs" -v "$W/ctx/build_wheels.sh:/build_wheels.sh:ro" -v "$W/ctx/patch:/patch:ro" \
    --entrypoint bash "$BASE" /build_wheels.sh
  log "wheels done in $(( $(date +%s) - t0 ))s: $(ls "$W/wheels")"
fi

if [[ $STEPS == *3* ]]; then
  C=$W/ctx
  rm -rf "$C/wheels" "$C/b12x"; mkdir -p "$C/wheels"
  cp "$W"/wheels/vllm-*.whl "$W"/wheels/exllamav3-*.whl "$C/wheels/"
  rm -rf "$C/st_moe_ext_so"; cp -r "$W/wheels/st_moe_ext" "$C/st_moe_ext_so"
  git -C "$W/src/b12x" archive --prefix=b12x/ HEAD | tar -x -C "$C"
  t0=$(date +%s)
  docker build --pull=false --build-arg BASE="$BASE" -t "$IMG" "$C" 2>&1 | tee "$W/logs/docker-build.log"
  log "image built in $(( $(date +%s) - t0 ))s: $(docker image inspect "$IMG" --format '{{.Id}} {{.Size}}')"
fi

if [[ $STEPS == *4* ]]; then
  docker run --rm --gpus all --ipc host --ulimit memlock=-1 --entrypoint python3 "$IMG" /opt/st/validate.py \
    2>&1 | tee "$W/logs/validate.log"
fi
