#!/bin/bash
# DeepSeek-V4.1-Flash, sovereign-trellis EXL3 routed experts (plan J268-ho-v31-K5n), TP2 over 2x DGX Spark.
# Run on the head (rank 0, API). Starts rank 1 on WORKER over ssh (fabric 10.10.10.x), then rank 0 here.
# Default roles: head spark-557f (10.10.10.13, has WiFi/Tailscale), worker spark-2822 (10.10.10.11, user sero).
# A host memguard on both nodes kills the model container if MemAvailable < MEMGUARD_GIB (GB10 unified memory:
# a GPU OOM otherwise wedges the host until the watchdog reboots it, as on 2026-10-03 04:00).
# Flags follow LIL r38 serve-ds41-flash.sh (Engram on disk, DSpark, B12X) + LIL's Spark TP2 launcher (sm_121a, RoCE).
#   ./run_2spark.sh            start both ranks (detached containers st-ds41-r0 / st-ds41-r1)
#   ./run_2spark.sh stop       stop both
# Knobs (env): KV_BYTES MAX_LEN MAX_SEQS NSPEC BATCH PREFILL ALLREDUCE PROJ_TP RES_SCALES GPU_UTIL CAPS EXTRA
# Never caps outputs; never touches GPU power/clocks.
set -euo pipefail
IMG=${IMG:-ghcr.io/0xsero/deepseek-v4.1-flash-spark:s016}   # same image as local tag sovereign-trellis/ds41-exl3-spark:v41fix3
WORKER=${WORKER:-sero@10.10.10.11}
WSSH="ssh -i $HOME/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=10 $WORKER"
HEAD_IP=${HEAD_IP:-10.10.10.13}; MPORT=${MPORT:-29655}; PORT=${PORT:-8000}
PLAN=${PLAN:-J268-ho-v31-K5n.json}
KV_BYTES=${KV_BYTES:-2700000000}          # S004: 2.7e9 B = 1,541,640 tok with --swa-block-size 128 (1,751 B/tok) -> 3.6e9 ~ 2.05M
MAX_LEN=${MAX_LEN:-262144}; MAX_SEQS=${MAX_SEQS:-2}; NSPEC=${NSPEC:-7}; BATCH=${BATCH:-2048}
PREFILL=${PREFILL:-st}; ALLREDUCE=${ALLREDUCE:-rocenante}; PROJ_TP=${PROJ_TP:-false}; RES_SCALES=${RES_SCALES:-false}
GPU_UTIL=${GPU_UTIL:-0.88}; ADAPT=${ADAPT:-true}
GRAPH_MODE=${GRAPH_MODE:-FULL_DECODE_ONLY}   # S006: FULL_AND_PIECEWISE ran rank 1 (2822) to MemAvailable 2 GiB in capture
CAP=$((MAX_SEQS * (NSPEC + 1)))
CAPS=${CAPS:-$(python3 -c "d=$NSPEC+1;c=$CAP;print(','.join(map(str,sorted(set(list(range(1,min(d,c)+1))+list(range(d,c+1,4))+[c])))))")}
API_KEY_FILE=${API_KEY_FILE:-$HOME/dsv41-st/api_key}
MEMGUARD_GIB=${MEMGUARD_GIB:-1}

if [ "${1:-}" = stop ]; then
  timeout 60 docker rm -f st-ds41-r0 >/dev/null 2>&1 || true
  timeout 60 $WSSH "docker rm -f st-ds41-r1 >/dev/null 2>&1 || true; bash ~/dsv41-st/memguard.sh stop"
  bash ~/dsv41-st/memguard.sh stop
  echo stopped; exit 0
fi
[ -f "$API_KEY_FILE" ] || { umask 077; python3 -c "import secrets;print(secrets.token_urlsafe(32))" > "$API_KEY_FILE"; }

# per-node paths (same layout on both nodes; 557f home is /home/valentine)
plan_rewrite() {  # plan copy with qdir pointing at the container mount points
  python3 - "$1" "$2" <<'PY'
import json, sys
p = json.load(open(sys.argv[1])); p["qdir"] = "/banks/qn:/banks/q31:/banks/q31s"; json.dump(p, open(sys.argv[2], "w"))
PY
}
mkdir -p ~/dsv41-st/plans; plan_rewrite ~/dsv41-st/plans-src/$PLAN ~/dsv41-st/plans/$PLAN
timeout 60 $WSSH "mkdir -p ~/dsv41-st/plans ~/dsv41-st/cache"
scp -q -i ~/.ssh/id_ed25519 ~/dsv41-st/plans/$PLAN $WORKER:dsv41-st/plans/$PLAN

SPEC=$(printf '{"method":"dspark","num_speculative_tokens":%s,"draft_tensor_parallel_size":2,"attention_backend":"B12X","draft_sample_method":"greedy","rejection_sample_method":"standard","enable_adaptive_verification":%s}' "$NSPEC" "$ADAPT")
ENGRAM=$(printf '{"cpu_offload":false,"table_memory":"disk","disk_resident_scales":%s,"disk_prefetch_max_tokens":0,"projection_tp":%s}' "$RES_SCALES" "$PROJ_TP")
COMP=$(printf '{"cudagraph_mode":"%s","custom_ops":["all"],"cudagraph_capture_sizes":[%s]}' "$GRAPH_MODE" "$CAPS")

ENVS=(-e CUDA_VISIBLE_DEVICES=0 -e CUTE_DSL_ARCH=sm_121a -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e VLLM_USE_V2_MODEL_RUNNER=1 -e VLLM_USE_BREAKABLE_CUDAGRAPH=0 -e VLLM_USE_FLASHINFER_SAMPLER=1
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e OMP_NUM_THREADS=16 -e MALLOC_ARENA_MAX=2
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e SAFETENSORS_FAST_GPU=1
  -e VLLM_ENABLE_PCIE_ALLREDUCE=0 -e NCCL_IB_DISABLE=0 -e NCCL_NET_PLUGIN=none -e NCCL_IB_GID_INDEX=3
  -e NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1 -e NCCL_IB_MERGE_NICS=1 -e NCCL_SOCKET_IFNAME=enp1s0f1np1
  -e GLOO_SOCKET_IFNAME=enp1s0f1np1 -e NCCL_DEBUG=WARN
  -e B12X_COMPILE_CACHE_DIR=/cache/b12x -e XDG_CACHE_HOME=/cache
  -e ST_EXL3_PLAN=/plans/$PLAN -e ST_EXL3_PREFILL=$PREFILL -e ST_EXL3_PREFILL_MIN_ROWS=${MIN_ROWS:-1})
[ -n "${NCCL_PROTO:-}" ] && ENVS+=(-e NCCL_PROTO=$NCCL_PROTO)   # S015: forcing Simple cut prefill 4-10%
if [ "$ALLREDUCE" = rocenante ]; then
  ENVS+=(-e VLLM_ENABLE_ROCE_ALLREDUCE=1 -e VLLM_ROCE_ALLREDUCE_MAX_SIZE=2MB -e VLLM_ROCE_ALLGATHER_MAX_SIZE=16MB
         -e B12X_ROCE_CACHE_DIR=/cache/b12x-roce -e B12X_ROCE_TRAFFIC_CLASS=106 -e NCCL_IB_TC=106)
fi
dock() {  # $1 = node home dir
  echo -d --gpus all --network host --ipc host --privileged --ulimit memlock=-1 --shm-size 32g \
    -v $1/models/DeepSeek-V4.1-Flash:/model:ro -v $1/dsv41-st/banks:/banks:ro \
    -v $1/dsv41-st/plans:/plans:ro -v $1/dsv41-st/cache:/cache
}
WHOME=${WHOME:-/home/sero}
ARGS=(/model --served-model-name deepseek-v4.1-flash --dtype bfloat16 --tensor-parallel-size 2
  --nnodes 2 --master-addr $HEAD_IP --master-port $MPORT
  --kv-cache-dtype fp8 --block-size 256 --swa-block-size 128 --kv-cache-memory-bytes $KV_BYTES --gpu-memory-utilization $GPU_UTIL
  --max-model-len $MAX_LEN --max-num-seqs $MAX_SEQS --max-num-batched-tokens $BATCH
  --max-cudagraph-capture-size $CAP --compilation-config "$COMP"
  --enable-chunked-prefill --async-scheduling --enable-prefix-caching --safetensors-load-strategy lazy
  --engram-config "$ENGRAM" --attention-backend B12X --linear-backend b12x --moe-backend b12x
  --speculative-config "$SPEC" --generation-config vllm --tokenizer-mode deepseek_v41
  ${EXTRA:-})
q() { printf '%q ' "$@"; }

# host memguard (both nodes): memguard.sh next to this script, copied to the worker
G="nohup setsid bash ~/dsv41-st/memguard.sh $MEMGUARD_GIB >/dev/null 2>&1 < /dev/null &"
scp -q -i ~/.ssh/id_ed25519 ~/dsv41-st/memguard.sh $WORKER:dsv41-st/memguard.sh
bash -c "$G"; timeout 30 $WSSH "$G"

# rank 1 (headless) on 557f
timeout 60 $WSSH "docker rm -f st-ds41-r1 >/dev/null 2>&1; docker run --name st-ds41-r1 $(dock $WHOME) $(q "${ENVS[@]}") -e VLLM_HOST_IP=${WORKER#*@} $IMG $(q "${ARGS[@]}") --node-rank 1 --headless"
# rank 0 (API) here
timeout 60 docker rm -f st-ds41-r0 >/dev/null 2>&1 || true
timeout 60 docker run --name st-ds41-r0 $(dock $HOME) "${ENVS[@]}" -e VLLM_HOST_IP=$HEAD_IP $IMG "${ARGS[@]}" --node-rank 0 \
  --host 0.0.0.0 --port $PORT --api-key "$(cat $API_KEY_FILE)" \
  --reasoning-parser deepseek_v41 --tool-call-parser deepseek_v41 --enable-auto-tool-choice \
  --enable-prompt-tokens-details --enable-force-include-usage
echo "started: head st-ds41-r0 (port $PORT), worker st-ds41-r1 on $WORKER; KV $KV_BYTES B, len $MAX_LEN, seqs $MAX_SEQS, dspark $NSPEC, prefill=$PREFILL, allreduce=$ALLREDUCE"
