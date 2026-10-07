# ghcr.io/0xsero/dsv41-flash-offload: DeepSeek-V4.1-Flash EXL3 (3.0bpw, Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-3.0bpw) on ONE
# 24 GB RTX 3090 (sm_86) + host DDR4 + NVMe, OpenAI-compatible API on :8000 (chat with deepseek_v41 tool calls and
# reasoning, completions). Serving code: github.com/0xSero/dsv41-flash-offload at the pinned commit below = campaign run
# D118: staged-DMA prefill (16k chunks, copy-engine staging, FP16 GEMM for busy experts), AVX2 CPU expert tier for
# decode misses, elastic VRAM expert mirror cache, Engram n-gram tables memory-mapped from NVMe.
#   measured (max-model-len 65536, KV 1 GiB): D061 prefill 8k 565 / 32k 745 tok/s; D107 decode C1 21.34 / C2 26.27 /
#   C4 28.30 agg; D117 (D061 + non-elastic cache) prefill 575 / 766, decode C1 15.55.
#   prefill fidelity (D061 path) vs the D026 reference: top-1 0.9856, mean KL 0.0068 nats (inherited guard: fail).
#   D119 (this default: --max-model-len 262144, KV 1.5 GiB) C1 prefill/decode 32k 774.6/19.42, 65k 813.2/20.51, 261k 764.1/20.59;
#   container RAM peak 212.1 GiB (256 GiB host), NVMe 488 GB free, VRAM 21,852 MiB.
#   D141 (this commit's defaults, 2026-10-03; D135 + CPU-assisted prefill for 513-2048-token turns + near-full-window max_tokens accepted): prefix caching (exact: identical greedy tokens cold vs cached), <=512-token
#   steps on the CPU-tier split path, safe elastic expert cache, 8 staging slots + cross-layer DMA prefetch, decode share;
#   chunk 8192. Decode C1/C2/C4 ~19/22/24 agg tok/s; prefill 8k/64k/261k 787/769/694 tok/s; cached 131k-token turn TTFT
#   0.76 s; 30-min 4-worker soak 58 requests, 0 errors, 0 OOM; VRAM peak 23.1 GB. results/D130.md in the serving repo.
#
# Base + upstream pins = the campaign runtime image dsv41-exl3:ampere-d005 (run D005) as one Dockerfile:
#   lazymio/vllm-backport v0.13.0 amd64 (vLLM 0.13.0 with DeepSeek-V4.1, torch 2.13.0+cu130, CUDA 13.0)
#   Tokha233/deepseek-v4.1-flash-a100-turbo @ fae324ae (Ampere V4.1 file patches, hash-checked)
#   turboderp-org/exllamav3 @ 5be88657 built for sm_86
#   vcruz305/vllm-exl3 @ d3cfd394 + the repo's patches/vllm-exl3-host-dma.patch (D004), VLLM_EXL3_NO_CUDA=1
# The repo's scripts/install.sh applies the runtime patches at build time and fails the build unless every patched
# file's before/after sha256 equals the receipts of the measured runs (runtime/expected-receipts.json).
# The model is NOT in the image: `docker run ... IMAGE prepare` downloads the two pinned inputs (422 GB) into /models
# and builds the overlay pack /models/DSV41-EXL3-3090-D010; then serve with /models mounted read-only.
# Build: pushed by .github/workflows/release-image.yml with BuildKit provenance, an SBOM, and a build attestation.
FROM lazymio/vllm-backport@sha256:349690323ab9aba712111529ed1ca60730199205d8202f67895ffde85b451be3
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

LABEL org.opencontainers.image.source="https://github.com/sybil-solutions/local-ai-images"
LABEL ai.omarchy.engine="vllm"
LABEL ai.omarchy.engine.version="0.13.0"
LABEL ai.omarchy.quant="exl3"
LABEL ai.omarchy.hardware="rtx-3090-24gb"
LABEL ai.omarchy.cuda.archs="8.6"
LABEL ai.omarchy.model.repo="Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-3.0bpw"
LABEL ai.omarchy.model.revision="c5534b90602b4090980a1c3ded1eb3b4d99a38d5"
LABEL ai.omarchy.model.mount="/models"

ARG TORCH_CUDA_ARCH_LIST="8.6"
ARG MAX_JOBS=4
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
RUN apt-get update \
 && apt-get install -y --no-install-recommends git patch libcusparse-dev-13-0 libcusolver-dev-13-0 \
 && rm -rf /var/lib/apt/lists/*

ARG A100_TURBO_COMMIT=fae324ae62ac5cef31b7d38f5d369618e1cae1fa
RUN git init -q /tmp/a100 && cd /tmp/a100 \
 && git fetch -q --depth 1 https://github.com/Tokha233/deepseek-v4.1-flash-a100-turbo "${A100_TURBO_COMMIT}" \
 && git checkout -q --detach FETCH_HEAD && test "$(git rev-parse HEAD)" = "${A100_TURBO_COMMIT}" \
 && cp -a patches /opt/dsv41-ampere-patches && rm -rf /tmp/a100 \
 && python3 /opt/dsv41-ampere-patches/install.py

# ExLlamaV3 (MIT, turboderp) at the commit the campaign image used; pipefail keeps pip's exit status through the tail.
ARG EXLLAMAV3_COMMIT=5be886578ec80324c2c715269387be2058724b6e
RUN git clone -q https://github.com/turboderp-org/exllamav3.git /opt/exllamav3 \
 && git -C /opt/exllamav3 checkout -q "${EXLLAMAV3_COMMIT}" \
 && test "$(git -C /opt/exllamav3 rev-parse HEAD)" = "${EXLLAMAV3_COMMIT}" \
 && MAX_JOBS=${MAX_JOBS} python3 -m pip install --no-build-isolation --no-deps /opt/exllamav3 2>&1 | tail -3 \
 && python3 -m pip show exllamav3 >/dev/null

# Serving code at the pinned commit; fail closed if the checkout is anything else.
ARG DSV41_REPO=https://github.com/sybil-solutions/dsv41-flash-offload
# D141 serving code plus generated-pack hash verification. Prior serving/quality
# observations above remain historical; this rebuilt digest needs fresh acceptance.
ARG DSV41_COMMIT=d890a1c897f1c9d6d981473ec106224aeb9cbb4d
LABEL ai.omarchy.dsv41-flash-offload.commit="${DSV41_COMMIT}"
RUN git init -q /opt/dsv41 && cd /opt/dsv41 \
 && git fetch -q --depth 1 "${DSV41_REPO}" "${DSV41_COMMIT}" \
 && git checkout -q --detach FETCH_HEAD \
 && test "$(git rev-parse HEAD)" = "${DSV41_COMMIT}" \
 && echo "${DSV41_COMMIT}" > /opt/dsv41/COMMIT \
 && rm -rf .git results

ARG VLLM_EXL3_COMMIT=d3cfd394920360d69f820d2dc96f8292a9e10283
RUN git clone -q https://github.com/vcruz305/vllm-exl3.git /opt/vllm-exl3 \
 && git -C /opt/vllm-exl3 checkout -q "${VLLM_EXL3_COMMIT}" \
 && test "$(git -C /opt/vllm-exl3 rev-parse HEAD)" = "${VLLM_EXL3_COMMIT}" \
 && cd /opt/vllm-exl3 && patch -p1 < /opt/dsv41/patches/vllm-exl3-host-dma.patch \
 && VLLM_EXL3_NO_CUDA=1 python3 -m pip install --no-build-isolation --no-deps . \
 && python3 -m pip install --no-cache-dir marisa-trie==1.4.1

RUN chmod 0755 /opt/dsv41/docker/entrypoint.sh /opt/dsv41/scripts/install.sh \
 && /opt/dsv41/scripts/install.sh /opt/dsv41

ENV VLLM_PLUGINS=vllm_exl3 EXL3_MOE_KERNEL=exllamav3 \
    PYTHONPATH=/opt/dsv41/runtime:/opt/dsv41/pypath \
    DSV41_ENGRAM_LIB=/opt/dsv41/build/engram_disk.so
EXPOSE 8000
ENTRYPOINT ["/opt/dsv41/docker/entrypoint.sh"]
