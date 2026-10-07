# ghcr.io/0xsero/sglang-exl3-flashnext: Qwen3.8-Flash-Next EXL3 (3.05bpw_h5_ng5) on ONE RTX 3090 (sm_86), with
# two-tier offload: routed experts in pinned host RAM behind a global GPU expert cache, and the 32.6 GB EXL3 n-gram
# table on NVMe (buffered reads, page cache as elastic L2) behind an 8 GB RAM row cache. The serving code is
# trellis-serve (github.com/0xSero/trellis-serve, branch flashnext-phase-d) at b1e74af: the vision build (1e72b25)
# plus the Phase D defaults (5-bit QSA KV, expert-cache reserve 8.26 GB, 12k prefill chunks), the NVMe row-cache
# hint/eviction fix (37f9017), buffered x64 n-gram IO and CLOCK cold insertion, validated in INT058b (2026-09-30).
#
# Base is ghcr.io/0xsero/sglang-exl3 v0.7.0-ampere by digest: lmsysorg/sglang v0.5.20 (CUDA 13.0, torch 2.13, nvcc
# present) + exllamav3 1.5.1 (6b84a21, = v1.5.1 plus an MSVC-only fix) built for sm_86, shim entrypoint, openssh.
# Built by this repo's release-image workflow (run 36210770340, attested). The campaign's dev image had the same
# SGLang build (commit 94602c9) and exllamav3 v1.5.1 on the same lmsysorg base.
#
# This image adds trellis-serve at the pinned commit and compiles its kernels AOT for sm_86:
#   - trellis_exl3_kernels (dense Marlin-template EXL3 GEMMs, MUL1 codebook, K = 3/4/5/6)
#   - trellis_exl3_moe_kernels (grouped MoE + GPU expert cache, K = 3/4, fp16-accumulate prefill variants)
#   - trellis_offload_kernels (the offload runtime's JIT extension, pre-built into the JIT cache)
# The n-gram NVMe row store (csrc/offload/ngram_nvme.cu) is compiled with -march=native, so it is deliberately NOT
# pre-built here: it JIT-compiles on the serving host at first start (~1 min, nvcc + ninja are in the image) and is
# cached in SGLANG_EXL3_JIT_DIR for the life of the container.
#
# Build: pushed by .github/workflows/release-image.yml with BuildKit provenance, an SBOM, and a GitHub build
# attestation on the digest.
FROM ghcr.io/0xsero/sglang-exl3@sha256:c8922bd7256caf1b273af7ac49221e652be9e6072c1cdec4b959f506fd415d79

LABEL org.opencontainers.image.source="https://github.com/sybil-solutions/local-ai-images"
LABEL ai.omarchy.engine="sglang"
LABEL ai.omarchy.engine.version="0.5.20"
LABEL ai.omarchy.quant="exl3"
LABEL ai.omarchy.hardware="rtx-3090-24gb"
LABEL ai.omarchy.cuda.archs="8.6"
LABEL ai.omarchy.model.repo="turboderp/Qwen3.8-Flash-Next-exl3"
LABEL ai.omarchy.model.revision="69e33439ae950f17bcbe95c98f117d80f759ab6d"

ARG TRELLIS_SERVE_REPO=https://github.com/sybil-solutions/trellis-serve
ARG TRELLIS_SERVE_COMMIT=b1e74afcf98203dc5326349623b556ba2df4b16a
LABEL ai.omarchy.trellis-serve.commit="b1e74afcf98203dc5326349623b556ba2df4b16a"
ENV TORCH_CUDA_ARCH_LIST=8.6
# Phase D server defaults (INT058b): 5-bit exllamav3-quantized QSA KV (9,216 B/token, 209,984 tokens at 204,800
# context) and an 8.26 GB reserve for the expert-cache auto-fit (11.08 GB of experts on a 24 GB card). Recipes also
# pass these explicitly, with --chunked-prefill-size 12288.
ENV SGLANG_EXL3_KV_BITS=5 \
    SGLANG_EXL3_EXPERT_CACHE_RESERVE_GB=8.26

# Source at the pinned commit; fail closed if the checkout is anything else.
RUN git init -q /opt/trellis-serve \
 && cd /opt/trellis-serve \
 && git fetch -q --depth 1 "${TRELLIS_SERVE_REPO}" "${TRELLIS_SERVE_COMMIT}" \
 && git checkout -q --detach FETCH_HEAD \
 && test "$(git rev-parse HEAD)" = "${TRELLIS_SERVE_COMMIT}" \
 && echo "${TRELLIS_SERVE_COMMIT}" > /opt/trellis-serve/COMMIT \
 && rm -rf .git

# The trellis-serve `sglang-exl3` distribution replaces the base image's plugin of the same name (same entry point,
# `sglang.srt.plugins: exl3`); exllamav3 stays (the n-gram and CPU-worker paths import it).
RUN pip install -q --no-deps --no-build-isolation -e /opt/trellis-serve/core -e /opt/trellis-serve/cuda \
 && python3 -c "from importlib.metadata import entry_points as e, distribution as d; \
assert any(x.value == 'sglang_exl3.plugin:activate' for x in e(group='sglang.srt.plugins')); \
print('sglang-exl3 dist at', d('sglang-exl3').read_text('direct_url.json'))"

# Kernels, exactly the campaign's two build.sh invocations (dense, then MoE only), MUL1 codebook.
# build.sh never exits non-zero; its log ends in BUILD_OK or BUILD_FAILED, so check the log.
RUN cd /opt/trellis-serve/cuda/csrc \
 && export MAX_JOBS=4 \
 && TRELLIS_CODEBOOKS=2 TRELLIS_MOE=0 ./build.sh \
 && { grep -q BUILD_OK build.log || { tail -60 build.log; exit 1; }; } \
 && TRELLIS_MOE=only TRELLIS_CODEBOOKS=2 ./build.sh \
 && { grep -q BUILD_OK build.log || { tail -60 build.log; exit 1; }; } \
 && rm -rf build/temp.* build/gen build.log \
 && ls -la build/lib

ENV PYTHONPATH=/opt/trellis-serve/cuda/csrc/build/lib \
    SGLANG_EXL3_JIT_DIR=/opt/trellis-serve/cuda/csrc/build/jit

# Pre-build the offload runtime's JIT extension (no host-specific flags) and prove every AOT module imports
# (CPU only: no GPU is needed to load the extensions).
RUN python3 -c "import torch, trellis_exl3_kernels, trellis_exl3_moe_kernels, importlib; \
importlib.import_module('sglang_exl3.offload._ext').load(); print('trellis-serve kernels ok')"

# Entrypoint (/opt/entrypoint.sh shim), ports 30000/22 and openssh are inherited from the base.
