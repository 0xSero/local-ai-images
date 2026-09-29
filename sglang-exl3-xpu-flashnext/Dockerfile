# ghcr.io/0xsero/sglang-exl3-xpu-flashnext: Qwen3.8-Flash-Next EXL3 (3.05bpw_h5_ng5) on ONE Intel Arc Pro B70
# (Xe2 / Battlemage) through SGLang v0.5.20-xpu, with two-tier offload: routed experts in USM host RAM behind a
# device-managed expert cache (9,000 slots), and the 32.6 GB EXL3 n-gram table on NVMe behind an 8 GB RAM row
# cache. The serving code is trellis-serve (github.com/0xSero/trellis-serve, branch ft-xpu-vision) at 72fd2d4: the
# X008 NVMe-tier build (13f7ac1) plus EXL3_SYNC_LOGITS and image input on XPU kernels (XVIS20, 2026-09-29).
#
# Base is lmsysorg/sglang:v0.5.20-xpu by digest (linux/amd64 index, inspected 2026-09-29; the same digest the
# campaign's B70 containers ran). It carries torch 2.13 XPU, oneAPI (icpx 2026.0) and Level Zero.
#
# This image adds, exactly as the campaign built them inside that base:
#   - exl3xpu/_C_sgl.so          (scripts/build_ext.sh, EXL3_OUT; dense EXL3 ops, oneDNN int8 prefill GEMM)
#   - exl3xpu/_moe_sgl.so        (scripts/build_moe.sh, EXL3_MOE_OUT; grouped MoE + expert cache)
#   - exl3xpu/exl3xpu_ngram_nvme.so (scripts/build_ngram_nvme.sh; NVMe row store + XPU publish/wait handshake)
#   - oneDNN 2026.0.0 from pip (the campaign installed `onednn` at every container start; baked in here)
# Device code is spir64 (JIT): the Level Zero driver finalises the kernels on first launch and caches them.
# The one change to the build: build_ngram_nvme.sh compiles host code with -march=native; a CI runner is not the
# serving host, so this build uses -march=x86-64-v3 -mclflushopt (AVX2 + CLFLUSHOPT; the host code needs only
# _mm_clflushopt/_mm_sfence/_mm_pause). Any Zen 2+ / Skylake+ host runs it.
#
# Build: pushed by .github/workflows/release-image.yml with BuildKit provenance, an SBOM, and a GitHub build
# attestation on the digest.
FROM lmsysorg/sglang@sha256:e4e10d66de374e6e8cc88136e8dbfc3182787070df8a48e581fb007cc7a51c36

LABEL org.opencontainers.image.source="https://github.com/0xSero/local-ai-images"
LABEL ai.omarchy.engine="sglang"
LABEL ai.omarchy.engine.version="0.5.20-xpu"
LABEL ai.omarchy.quant="exl3"
LABEL ai.omarchy.hardware="intel-arc-pro-b70-32gb"
LABEL ai.omarchy.model.repo="turboderp/Qwen3.8-Flash-Next-exl3"
LABEL ai.omarchy.model.revision="69e33439ae950f17bcbe95c98f117d80f759ab6d"

ARG TRELLIS_SERVE_REPO=https://github.com/0xSero/trellis-serve
ARG TRELLIS_SERVE_COMMIT=72fd2d4ca8e8de3fce03ab09e3ed1692957c2f8c
LABEL ai.omarchy.trellis-serve.commit="72fd2d4ca8e8de3fce03ab09e3ed1692957c2f8c"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Source at the pinned commit; fail closed if the checkout is anything else.
RUN git init -q /opt/trellis-serve \
 && cd /opt/trellis-serve \
 && git fetch -q --depth 1 "${TRELLIS_SERVE_REPO}" "${TRELLIS_SERVE_COMMIT}" \
 && git checkout -q --detach FETCH_HEAD \
 && test "$(git rev-parse HEAD)" = "${TRELLIS_SERVE_COMMIT}" \
 && echo "${TRELLIS_SERVE_COMMIT}" > /opt/trellis-serve/COMMIT \
 && rm -rf .git

# oneDNN: runtime (libdnnl.so.3 into the venv) and headers for build_ext.sh's EXL3_DNNL path.
RUN pip install -q --no-deps onednn==2026.0.0 onednn-devel==2026.0.0

RUN cd /opt/trellis-serve/xpu \
 && X=/opt/trellis-serve/xpu/exl3xpu \
 && EXL3_DNNL_DIR="$(python3 -c 'import sys; print(sys.prefix)')" EXL3_OUT=$X/_C_sgl.so bash scripts/build_ext.sh \
 && EXL3_MOE_OUT=$X/_moe_sgl.so bash scripts/build_moe.sh \
 && sed -i 's/-march=native/-march=x86-64-v3 -mclflushopt/' scripts/build_ngram_nvme.sh \
 && bash scripts/build_ngram_nvme.sh \
 && test -s $X/_C_sgl.so && test -s $X/_moe_sgl.so && test -s $X/exl3xpu_ngram_nvme.so \
 && (readelf -d $X/_C_sgl.so | grep -q 'libdnnl.so' || { echo "_C_sgl.so built without oneDNN"; exit 1; })

RUN pip install -q --no-deps --no-build-isolation -e /opt/trellis-serve/core -e /opt/trellis-serve/xpu \
 && python3 -c "from importlib.metadata import entry_points as e; \
assert any(x.name == 'exl3xpu' for x in e(group='sglang.srt.plugins')); print('exl3xpu sglang plugin registered')"

ENV EXL3_LIB=/opt/trellis-serve/xpu/exl3xpu/_C_sgl.so \
    EXL3_MOE_LIB=/opt/trellis-serve/xpu/exl3xpu/_moe_sgl.so \
    TORCHINDUCTOR_COMPILE_THREADS=1

# Every library loads next to torch the way the server loads it (CPU only, no device needed).
RUN python3 -c "import os, torch, importlib.util as u; \
torch.ops.load_library(os.environ['EXL3_LIB']); torch.ops.load_library(os.environ['EXL3_MOE_LIB']); \
p = '/opt/trellis-serve/xpu/exl3xpu/exl3xpu_ngram_nvme.so'; s = u.spec_from_file_location('exl3xpu_ngram_nvme', p); \
s.loader.exec_module(u.module_from_spec(s)); print('exl3xpu libraries load')"

COPY entrypoint.sh /opt/entrypoint.sh
RUN chmod 0755 /opt/entrypoint.sh
EXPOSE 30250
ENTRYPOINT ["/opt/entrypoint.sh"]
