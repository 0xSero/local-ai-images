# ghcr.io/0xsero/deepseek-v4.1-flash-spark: DeepSeek-V4.1-Flash-Spark (DeepSeek-V4.1-Flash with EXL3 trellis routed
# experts) served by vLLM with tensor parallel 2 across TWO NVIDIA DGX Spark (GB10, sm_121, arm64) over the RoCE
# fabric. OpenAI-compatible API on :8000 (deepseek_v41 reasoning + tool calls, image input).
#
# linux/arm64 ONLY. Architecture-specific ops target GB10 (12.1a); FA2 retains upstream 8.0+PTX.
# Hopper-only FA3 is excluded. This image is intended for DGX Spark.
#
# Stack (all pinned; sources fetched as GitHub archives with sha256 checksums):
#   base         vllm/vllm-openai:v0.30.0 (multi-arch index, arm64 used): torch 2.13.0+cu130, py3.12, nvcc 13.0,
#                flashinfer 0.6.18.post1, NCCL 2.30.7. Its vLLM 0.30.0 package is removed completely.
#   vLLM         local-inference-lab/vllm r38 @ 66c29357 (Apache-2.0), compiled ops rebuilt for GB10
#                (build_wheels.sh adds CUDA>=13 arch 12.1 and excludes Hopper-only FA3)
#   b12x         local-inference-lab/b12x @ ce419b52 (Apache-2.0), CuTe-DSL JIT for sm_121a
#   exllamav3    turboderp-org/exllamav3 v1.5.3 @ d3739fd3 (MIT), exllamav3_ext built for sm_121
#                (upstream turboderp, NOT the brandonmmusic fork that other Spark images bundle)
#   patch/       this repo's own code (MIT): st_exl3_moe (EXL3 routed-expert method for vLLM's deepseek_v4 /
#                deepseek_v4_1 models), st_exl3_prefill + st_moe_ext (prefill grouped GEMM over EXL3 experts, built
#                against the exllamav3 1.5.3 headers), wired in by patch_vllm.py (each edit asserts its anchor once).
# Weights are NOT in the image (mount the native base checkpoint at /model, the EXL3 expert banks at /banks, the plan
# at /plans); see README.md and spark/run_2spark.sh.
#
# Build: .github/workflows/release-image.yml with platform=linux/arm64 (ubuntu-24.04-arm runner; no GPU is needed to
# compile). The digest currently pinned by the registry was built on a DGX Spark with spark/host-build.sh +
# spark/overlay.Dockerfile and pushed from the host (unattested); see README.md.
FROM vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90 AS wheels
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]
ARG MAX_JOBS=4
ADD --checksum=sha256:a3c458d57d695f3d12a403a64ea693db296f9a8afa96905ec2e91bfed6d0a7ec https://github.com/local-inference-lab/vllm/archive/66c293578412417476f842c1da5805d3a3d959a8.tar.gz /tmp/vllm.tgz
ADD --checksum=sha256:df9bde275b90adc4c5830e895e9f1889c295d71826c3c9c17f009865225ca460 https://github.com/local-inference-lab/b12x/archive/ce419b52681b7922bb0972d4b58b590a3fd005b2.tar.gz /tmp/b12x.tgz
ADD --checksum=sha256:cb8066e9077ba539e4340e3576be8806bdbcae4f99f1a8f99a2c4ebdef9ab447 https://github.com/turboderp-org/exllamav3/archive/d3739fd393337b1ff4d6c2a342b12f0c87a9592f.tar.gz /tmp/exllamav3.tgz
RUN for p in vllm b12x exllamav3; do mkdir -p /src/$p && tar -xzf /tmp/$p.tgz -C /src/$p --strip-components=1; done \
 && rm -f /tmp/*.tgz && mkdir -p /out /logs
COPY build_wheels.sh /build_wheels.sh
COPY patch/ /patch/
# CPU-only compile; four nvcc jobs with 24 GB swap in the hosted release workflow.
RUN MAX_JOBS=${MAX_JOBS} bash /build_wheels.sh

FROM vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

LABEL org.opencontainers.image.source="https://github.com/0xSero/local-ai-images"
LABEL org.opencontainers.image.description="DeepSeek-V4.1-Flash-Spark: DeepSeek-V4.1-Flash, EXL3 routed experts, vLLM TP2 over 2x DGX Spark (sm_121, arm64)"
LABEL ai.omarchy.engine="vllm"
LABEL ai.omarchy.engine.version="0.26.1rc0+glm53.r38.sm121"
LABEL ai.omarchy.quant="exl3"
LABEL ai.omarchy.hardware="dgx-spark"
LABEL ai.omarchy.cuda.archs="12.1a"
LABEL ai.omarchy.model.repo="0xSero/DeepSeek-V4.1-Flash-Spark"
LABEL ai.omarchy.model.mount="/model"
LABEL st.vllm.repo="local-inference-lab/vllm" st.vllm.commit="66c293578412417476f842c1da5805d3a3d959a8" \
      st.b12x.repo="local-inference-lab/b12x" st.b12x.commit="ce419b52681b7922bb0972d4b58b590a3fd005b2" \
      st.exllamav3.repo="turboderp-org/exllamav3" st.exllamav3.commit="d3739fd393337b1ff4d6c2a342b12f0c87a9592f" \
      st.cuda.arch="12.1a" ds41.engram-table-memory="disk"

# liburing: Engram tables on disk (io_uring); b12x's C loader / RoCE helpers compile at first use with cc + liburing
RUN apt-get update \
 && apt-get install -y --no-install-recommends liburing-dev pkg-config \
 && rm -rf /var/lib/apt/lists/* \
 && echo "liburing $(pkg-config --modversion liburing)"

COPY --from=wheels /out/ /opt/st/wheels/
COPY --from=wheels /src/b12x/ /opt/st/b12x-src/
# r38 and b12x pin CuTe-DSL 4.6.2 and quack 0.6.4 (base ships 4.7.1 / 0.6.5). --no-deps everywhere: quack-kernels'
# resolver otherwise downgrades the base NCCL (nvidia-nccl-cu13 2.30.7 -> 2.29.7).
RUN python3 -m pip uninstall -y vllm \
 && rm -rf /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 -m pip install --no-deps /opt/st/wheels/vllm-*.whl /opt/st/wheels/exllamav3-*.whl \
 && python3 -m pip install --no-deps --no-build-isolation /opt/st/b12x-src \
 && python3 -m pip install --no-deps \
      "nvidia-cutlass-dsl==4.6.2" "nvidia-cutlass-dsl-libs-base==4.6.2" "nvidia-cutlass-dsl-libs-core==4.6.2" \
      "nvidia-cutlass-dsl-libs-cu12==4.6.2" "nvidia-cutlass-dsl-libs-cu13==4.6.2" "quack-kernels==0.6.4" \
      marisa_trie \
 && mkdir -p /opt/st/patch/st_moe_ext && cp /opt/st/wheels/st_moe_ext/st_moe_ext*.so /opt/st/patch/st_moe_ext/ \
 && rm -rf /root/.cache/pip /opt/st/wheels \
 && python3 -c "import importlib.metadata as m; print('nccl', m.version('nvidia-nccl-cu13'))" \
 && (python3 -m pip check || true)

COPY patch/ /opt/st/patch/
# patch_vllm.py fails the build if any anchor is missing; then check both DeepSeek-V4.1 hooks landed and the
# exllamav3 extension imports (CPU import only: the build host has no GPU)
RUN VD=$(python3 -c "import importlib.util,os;print(os.path.dirname(importlib.util.find_spec('vllm').origin))") \
 && python3 /opt/st/patch/patch_vllm.py "$VD" \
 && grep -q st_exl3 "$VD/models/deepseek_v4_1/quant_config.py" \
 && grep -q skip_checkpoint_tensor "$VD/models/deepseek_v4_1/nvidia/model.py" \
 && python3 -c "import exllamav3; from exllamav3.modules.quant.exl3 import LinearEXL3; print('exllamav3', exllamav3.__file__)"

COPY validate.py /opt/st/validate.py
ENV TORCH_CUDA_ARCH_LIST=12.1a \
    CUTE_DSL_ARCH=sm_121a \
    PYTHONPATH=/opt/st/patch \
    ST_EXL3_FUSED_ROWS=512
EXPOSE 8000
ENTRYPOINT ["vllm", "serve"]
