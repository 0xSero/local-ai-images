#!/bin/bash
# Runs INSIDE vllm/vllm-openai:v0.30.0 (arm64, torch 2.13.0+cu130, nvcc 13.0, py3.12), CPU only, from build.sh.
# Builds two wheels for GB10 (sm_121) into /out:
#   1. local-inference-lab/vllm r38 (66c29357), all compiled ops for TORCH_CUDA_ARCH_LIST=12.1a
#      (no 0.30.0 .so files are reused; the build dir /src/vllm/build persists, so a rerun is incremental)
#   2. turboderp exllamav3 1.5.3 (d3739fd, MIT), exllamav3_ext for sm_121
#   3. st_moe_ext (sovereign-trellis EXL3 prefill grouped GEMM, deploy/serve/patch/st_moe_ext), built against the
#      exllamav3 1.5.3 source headers, sm_121 -> /out/st_moe_ext/st_moe_ext*.so (patch dir mounted at /patch, ro)
# The Rust frontend (vllm-rs, _rust_tool_parser) is optional in r38 and only used by the minimax_m3 tool parser;
# no cargo is installed, so setuptools-rust skips it (logged).
set -euo pipefail
: "${MAX_JOBS:=10}" "${NVCC_THREADS:=1}" "${VLLM_VERSION:=0.26.1rc0+glm53.r38.sm121}"
export MAX_JOBS NVCC_THREADS
# The base image ships the CUDA math headers (cusparse.h, ...) only in the cu13 pip wheel, not in /usr/local/cuda;
# torch's ATen headers need them (deepgemm, exllamav3_ext).
export CPATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include${CPATH:+:$CPATH}
mkdir -p /out
# CMake FetchContent clones cutlass, flash-attention, FlashMLA, ... (the base image has no git)
command -v git >/dev/null || { apt-get update -qq && apt-get install -y -qq --no-install-recommends git >/dev/null; }
git config --global --add safe.directory '*'
python3 -m pip install -q "cmake>=3.26.1,<4" "setuptools-rust>=1.9.0" "setuptools>=77.0.3,<81" "setuptools-scm>=8" build wheel
cmake --version | head -1

if [ "${SKIP_VLLM:-0}" != 1 ]; then
  cd /src/vllm
  t0=$(date +%s)
  # Source patch (only one in vLLM): r38's CUDA>=13.0 branch lists 12.0 but not 12.1 in CUDA_SUPPORTED_ARCHS, so
  # TORCH_CUDA_ARCH_LIST=12.1a is narrowed to 12.0 and the per-kernel "12.0a;12.1a" lists then resolve to sm_120a,
  # whose cubins do not load on GB10. The CUDA 12.8 branch already lists 12.1; add it here too (idempotent).
  if ! grep -q '"7.5;8.0;8.6;8.7;8.9;9.0;10.0;11.0;12.0;12.1")' CMakeLists.txt; then
    test "$(grep -c '"7.5;8.0;8.6;8.7;8.9;9.0;10.0;11.0;12.0")' CMakeLists.txt)" = 1
    sed -i 's/"7.5;8.0;8.6;8.7;8.9;9.0;10.0;11.0;12.0")/"7.5;8.0;8.6;8.7;8.9;9.0;10.0;11.0;12.0;12.1")/' CMakeLists.txt
    echo "patched CMakeLists.txt: CUDA>=13.0 supported archs += 12.1"
  fi
  # shallow clone: no tags for setuptools-scm; pin the version string (pop-os r38 reports 0.26.1rc0+glm53.r38)
  SETUPTOOLS_SCM_PRETEND_VERSION="$VLLM_VERSION" VLLM_TARGET_DEVICE=cuda TORCH_CUDA_ARCH_LIST=12.1a \
  VLLM_DISABLE_SCCACHE=1 CMAKE_BUILD_TYPE=Release \
    python3 -m pip wheel -v --no-build-isolation --no-deps -w /out . 2>&1 | tee /logs/vllm-wheel.log | \
    grep -E --line-buffered "^\[[0-9]+/[0-9]+\]|error|Error|Skipping|rust|Built wheel|Created wheel" || true
  ls /out/vllm-*.whl
  echo "vllm wheel built in $(( $(date +%s) - t0 ))s"
fi

if [ "${SKIP_EXL3:-0}" != 1 ]; then
  cd /src/exllamav3
  t0=$(date +%s)
  TORCH_CUDA_ARCH_LIST=12.1 MAX_JOBS=8 \
    python3 -m pip wheel -v --no-build-isolation --no-deps -w /out . > /logs/exl3-wheel.log 2>&1
  ls /out/exllamav3-*.whl
  echo "exllamav3 wheel built in $(( $(date +%s) - t0 ))s"
fi

if [ "${SKIP_STX:-0}" != 1 ]; then
  t0=$(date +%s)
  rm -rf /tmp/stx && cp -r /patch/st_moe_ext /tmp/stx && cd /tmp/stx
  EXL3_EXT_DIR=/src/exllamav3/exllamav3/exllamav3_ext TORCH_CUDA_ARCH_LIST=12.1 MAX_JOBS=4 \
    python3 setup.py build_ext --inplace > /logs/st_moe_ext.log 2>&1
  rm -rf /out/st_moe_ext && mkdir -p /out/st_moe_ext && cp st_moe_ext*.so /out/st_moe_ext/
  ls /out/st_moe_ext
  echo "st_moe_ext built in $(( $(date +%s) - t0 ))s"
fi
