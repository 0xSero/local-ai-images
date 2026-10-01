# ghcr.io/0xsero/glm53-flash-offload: GLM-5.3-Flash EXL3 (3.05bpw, turboderp/GLM-5.3-Flash-exl3) on ONE 24 GB
# RTX 3090 (sm_86) + host DDR4, OpenAI-compatible API on :30000. Serving code: github.com/0xSero/glm53-flash-offload
# at the pinned commit below (monkeypatches over stock exllamav3 1.5.1: pinned zero-copy home copy of all routed
# experts, elastic CLOCK GPU expert cache + prefill staging, AVX2 CPU tier for cold decode misses).
#   GLM53_MODE=fast  (default) campaign run G067: prefill 8k 710 / 32k 951 tok/s, decode C1 28.15 / C2 31.57 /
#                    C4 33.98 agg, KV 131,072; prefill exact (panel 1.0000 / KL 0), decode NOT exact (CPU tier: paired
#                    KL 0.0053, top-1 0.977; fix in progress); host RAM ~218 GiB
#   GLM53_MODE=exact run G066a: prefill 697 / 945, decode ~12.6, bit-exact with exllamav3; host RAM ~111 GiB
#
# Base = the stack the numbers were measured on: lmsysorg/sglang v0.5.20 (CUDA 13.0, torch, Triton, nvcc; same digest
# as ./sglang-exl3) + exllamav3 v1.5.1 (tag commit 958ec93) built from source for sm_86 (binaries also run on sm_89).
# Baked in: the repo's routing stats, exllamav3 MoE tune cache, the exact Triton autotune pins (empty Triton cache, no
# benchmarking), and pre-built torch extensions (expert cache, CPU tier host -mavx2 -mfma -mf16c + GPU split/combine,
# fast split, fused hyper-connection kernels).
# The model is NOT in the image: mount the checkpoint at /models, or mount an empty writable dir there and the
# entrypoint downloads turboderp/GLM-5.3-Flash-exl3 @ 332ab457 (branch 3.05bpw, 125.3 GB).
#
# A CPU-tier fix ships as a new digest by bumping GLM53_COMMIT (the exllamav3 layer stays cached).
# Build: pushed by .github/workflows/release-image.yml with BuildKit provenance, an SBOM, and a build attestation.
FROM lmsysorg/sglang@sha256:06e4f2ed21afde4ff513cda65070124e727ba23ccaeff7712b8c40e1097d611f

LABEL org.opencontainers.image.source="https://github.com/0xSero/local-ai-images"
LABEL ai.omarchy.engine="exllamav3"
LABEL ai.omarchy.engine.version="1.5.1"
LABEL ai.omarchy.quant="exl3"
LABEL ai.omarchy.hardware="rtx-3090-24gb"
LABEL ai.omarchy.cuda.archs="8.6"
LABEL ai.omarchy.model.repo="turboderp/GLM-5.3-Flash-exl3"
LABEL ai.omarchy.model.revision="332ab457b709b7ba30dd9a448be5de03b80a7ac9"
LABEL ai.omarchy.model.mount="/models"

ARG TORCH_CUDA_ARCH_LIST="8.6"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ninja-build \
 && rm -rf /var/lib/apt/lists/*

# ExLlamaV3 v1.5.1 (MIT, turboderp), exactly the tag the measurements used. pip's exit status is lost in the pipe,
# so `pip show` is the check.
ARG EXLLAMAV3_COMMIT=958ec933361b24eb8426ec7222e5b0062a679dcd
RUN git clone https://github.com/turboderp-org/exllamav3 /opt/exllamav3-src \
 && cd /opt/exllamav3-src && git checkout -q ${EXLLAMAV3_COMMIT} \
 && test "$(git rev-parse HEAD)" = "${EXLLAMAV3_COMMIT}" \
 && MAX_JOBS=4 pip install --no-build-isolation --no-deps -v . 2>&1 | tail -3 \
 && pip install --no-deps marisa-trie \
 && pip show exllamav3 >/dev/null \
 && rm -rf /opt/exllamav3-src

# Serving code at the pinned commit; fail closed if the checkout is anything else.
ARG GLM53_REPO=https://github.com/0xSero/glm53-flash-offload
ARG GLM53_COMMIT=e1b6febf248306a798751897f130006b02f51cea
LABEL ai.omarchy.glm53-flash-offload.commit="e1b6febf248306a798751897f130006b02f51cea"
RUN git init -q /opt/glm53 \
 && cd /opt/glm53 \
 && git fetch -q --depth 1 "${GLM53_REPO}" "${GLM53_COMMIT}" \
 && git checkout -q --detach FETCH_HEAD \
 && test "$(git rev-parse HEAD)" = "${GLM53_COMMIT}" \
 && echo "${GLM53_COMMIT}" > /opt/glm53/COMMIT \
 && rm -rf .git \
 && chmod 0755 docker/entrypoint.sh scripts/install.sh \
 && MAX_JOBS=4 scripts/install.sh /opt/glm53

ENV GLM53_MODE=fast \
    GLM53_ZC_STATS=/opt/glm53/data/stats_own_dec.json GLM53_EC_WARM=/opt/glm53/data/stats_own_dec.json \
    GLM53_TRITON_PIN=/opt/glm53/data/triton_pin_exact.json \
    EXLLAMAV3_TUNE_CACHE=/opt/glm53/data/coop_autotune_1gpu.bin TRITON_CACHE_DIR=/opt/glm53/triton_cache

EXPOSE 30000
ENTRYPOINT ["/opt/glm53/docker/entrypoint.sh"]
