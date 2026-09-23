#!/usr/bin/env bash
# Build the EXL3 kernels for sm_86 inside the sglang-exl3:dev container:  ./build.sh [small|moe|probe]
# Result: build/lib/aikido_exl3_kernels*.so (+ aikido_exl3_moe_kernels*.so); log build.log ends with BUILD_OK/BUILD_FAILED
cd "$(dirname "$0")"
[ "$1" = small ] && export AIKIDO_MBS=0,1 AIKIDO_CODEBOOKS=2 AIKIDO_MOE=0
[ "$1" = moe ] && export AIKIDO_MOE=only
[ "$1" = probe ] && export AIKIDO_PROBES=1 AIKIDO_MBS=0,1 AIKIDO_CODEBOOKS=${3:-2,3,4,5} AIKIDO_KBITS=4 AIKIDO_MOE=0
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.6} MAX_JOBS=${MAX_JOBS:-40}
if python setup.py build_ext --build-lib build/lib > build.log 2>&1; then echo BUILD_OK >> build.log; else echo BUILD_FAILED >> build.log; fi
tail -3 build.log
