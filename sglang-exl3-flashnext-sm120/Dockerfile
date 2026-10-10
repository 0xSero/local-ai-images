# ghcr.io/0xsero/sglang-exl3-flashnext-sm120: Qwen3.8-Flash-Next EXL3 (3.05bpw_h5_ng5) on ONE 12 GB consumer Blackwell
# card (sm_120, RTX 5070 class) with 32 GB of host RAM: routed experts in a file-backed store on NVMe read by the GPU
# through HMM (open kernel modules, "Addressing Mode: HMM"), a popularity-ordered pinned RAM band per layer, the
# global GPU expert cache in the VRAM left over, and the EXL3 n-gram table on NVMe behind a small RAM row cache.
# Serving code: trellis-serve flashnext-phase-d b1e74af + the HMM store (SGLANG_EXL3_OFFLOAD_STORE=hmm, _SORT, _PIN_GB,
# opt-in _MASK_TAU), see the pinned commit below.
#
# Same build as ./sglang-exl3-flashnext (the sm_86 image) with two differences: the base is the v0.7.0-multiarch
# sglang-exl3 image (exllamav3 1.5.1 built for sm_86/sm_89/sm_120) and the trellis-serve kernels are compiled for 12.0.
# The n-gram NVMe row store still JIT-compiles on the serving host at first start (-march=native, ~1 min).
#
# Run needs (measured on RTX 5070 12 GB, Core Ultra 9 285, 32 GB DDR5, Samsung PM9F1): --ulimit memlock=-1 (the
# pinned band), the 85.14 GB checkpoint read-only, and a writable NVMe directory at /nvx for the 45.8 GB expert store
# plus its sorted copy (both written on the first starts).
#
# Status: DRAFT, not desktop-safe on a 32 GB box yet. A run with a 3.0 GB expert-cache reserve (~50 MB VRAM left) and a
# ~14 GB pinned band (~3 GB host MemAvailable) hung the host (NVRM NV_ERR_NO_MEMORY, then kcompactd0 and UVM GPU1 BH
# blocked > 120 s). The reserve default here is 4.2 GB (>= 1 GB VRAM free next to a desktop); see trellis-serve's draft
# PR for the measurements and the memcg-aware budget it still needs.
FROM ghcr.io/0xsero/sglang-exl3@sha256:9d95829ab41a9307e9b25c86e4af1ec52c6a02f7888f0e13a1ce93c0a7ff3814

LABEL org.opencontainers.image.source="https://github.com/sybil-solutions/local-ai-images"
LABEL ai.omarchy.engine="sglang"
LABEL ai.omarchy.engine.version="0.5.20"
LABEL ai.omarchy.quant="exl3"
LABEL ai.omarchy.hardware="rtx-5070-12gb"
LABEL ai.omarchy.cuda.archs="12.0"
LABEL ai.omarchy.model.repo="turboderp/Qwen3.8-Flash-Next-exl3"
LABEL ai.omarchy.model.revision="69e33439ae950f17bcbe95c98f117d80f759ab6d"

ARG TRELLIS_SERVE_REPO=https://github.com/IlyaasK/trellis-serve
ARG TRELLIS_SERVE_COMMIT=f3a05eb95c4887c7797d71a3624e33b2d7096011
LABEL ai.omarchy.trellis-serve.commit="f3a05eb95c4887c7797d71a3624e33b2d7096011"
ENV TORCH_CUDA_ARCH_LIST=12.0
# Phase D server defaults (INT058b): 5-bit exllamav3-quantized QSA KV (9,216 B/token, 209,984 tokens at 204,800
# context) and an 8.26 GB reserve for the expert-cache auto-fit (11.08 GB of experts on a 24 GB card). Recipes also
# pass these explicitly, with --chunked-prefill-size 12288.
ENV SGLANG_EXL3_KV_BITS=5 \
    SGLANG_EXL3_EXPERT_CACHE_RESERVE_GB=4.2

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
