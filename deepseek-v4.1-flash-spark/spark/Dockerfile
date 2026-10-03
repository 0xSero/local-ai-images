# sovereign-trellis serving image for DGX Spark (GB10, sm_121, arm64): DeepSeek-V4.1-Flash with EXL3 (trellis)
# routed experts on local-inference-lab vLLM r38 (66c29357, pop-os production parity).
#
# Base vllm/vllm-openai:v0.30.0 supplies the arm64 runtime (torch 2.13.0+cu130, py3.12, nvcc 13.0, flashinfer
# 0.6.18.post1, NCCL 2.30.7). Its vLLM 0.30.0 package is removed completely; the r38 wheel (all compiled ops rebuilt
# for TORCH_CUDA_ARCH_LIST=12.1a by build_wheels.sh) replaces it, so 0.30.0 .so files never mix with r38 Python.
# Added: b12x ce419b52 (CuTe-DSL JIT, CUTE_DSL_ARCH=sm_121a; its C loader/RoCE helpers compile at first use with cc
# + liburing), turboderp exllamav3 1.5.3 d3739fd (MIT; exllamav3_ext for sm_121), liburing for disk Engram io_uring,
# the st_exl3_moe routed-expert method + st_exl3_prefill wired in by patch_vllm.py, and the st_moe_ext prefill
# grouped-GEMM extension (.so in /opt/st/patch/st_moe_ext, found by st_exl3_prefill).
# Weights are NOT in the image. Entrypoint is `vllm serve`; launch with explicit args.
# Build with build.sh (it prepares wheels/ and b12x/ in the context).
ARG BASE=vllm/vllm-openai:v0.30.0
FROM ${BASE}
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

RUN apt-get update \
 && apt-get install -y --no-install-recommends liburing-dev pkg-config \
 && rm -rf /var/lib/apt/lists/* \
 && echo "liburing $(pkg-config --modversion liburing)"

COPY wheels/ /opt/st/wheels/
COPY b12x/ /opt/st/b12x-src/
# r38 and b12x pin CuTe-DSL 4.6.2 and quack 0.6.4 (base ships 4.7.1 / 0.6.5); the rest of r38's requirements are
# already in the base at equal or compatible versions (flashinfer 0.6.18.post1 vs pinned 0.6.17: pop-os r38 runs 0.6.18).
# --no-deps everywhere: quack-kernels' resolver otherwise downgrades the base NCCL (nvidia-nccl-cu13 2.30.7 -> 2.29.7).
RUN python3 -m pip uninstall -y vllm \
 && rm -rf /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 -m pip install --no-deps /opt/st/wheels/vllm-*.whl /opt/st/wheels/exllamav3-*.whl \
 && python3 -m pip install --no-deps --no-build-isolation /opt/st/b12x-src \
 && python3 -m pip install --no-deps \
      "nvidia-cutlass-dsl==4.6.2" "nvidia-cutlass-dsl-libs-base==4.6.2" "nvidia-cutlass-dsl-libs-core==4.6.2" \
      "nvidia-cutlass-dsl-libs-cu12==4.6.2" "nvidia-cutlass-dsl-libs-cu13==4.6.2" "quack-kernels==0.6.4" \
      marisa_trie \
 && rm -rf /root/.cache/pip /opt/st/wheels \
 && python3 -c "import importlib.metadata as m; print('nccl', m.version('nvidia-nccl-cu13'))" \
 && (python3 -m pip check || true)

COPY patch/ /opt/st/patch/
# st_moe_ext: EXL3 prefill grouped GEMM (ST_EXL3_PREFILL=st), prebuilt for sm_121 by build_wheels.sh
COPY st_moe_ext_so/ /opt/st/patch/st_moe_ext/
RUN VD=$(python3 -c "import importlib.util,os;print(os.path.dirname(importlib.util.find_spec('vllm').origin))") \
 && python3 /opt/st/patch/patch_vllm.py "$VD" \
 && python3 -c "import torch, exllamav3_ext, exllamav3; from exllamav3.modules.quant.exl3 import LinearEXL3; print('exllamav3 ok', exllamav3_ext.__file__)"

COPY validate.py /opt/st/validate.py
# TORCH_CUDA_ARCH_LIST also steers runtime JIT (flashinfer / torch extensions); CUTE_DSL_ARCH steers b12x.
ENV TORCH_CUDA_ARCH_LIST=12.1a \
    CUTE_DSL_ARCH=sm_121a \
    PYTHONPATH=/opt/st/patch \
    ST_EXL3_FUSED_ROWS=512
ENTRYPOINT ["vllm", "serve"]
LABEL org.opencontainers.image.source="https://github.com/0xSero/sovereign-trellis" \
      org.opencontainers.image.description="DeepSeek-V4.1-Flash, EXL3 routed experts (sovereign-trellis), LIL vLLM r38 for DGX Spark sm_121" \
      st.vllm.repo="local-inference-lab/vllm" st.vllm.commit="66c293578412417476f842c1da5805d3a3d959a8" \
      st.b12x.repo="local-inference-lab/b12x" st.b12x.commit="ce419b52681b7922bb0972d4b58b590a3fd005b2" \
      st.exllamav3.repo="turboderp-org/exllamav3" st.exllamav3.commit="d3739fd393337b1ff4d6c2a342b12f0c87a9592f" \
      st.cuda.arch="12.1a" ds41.engram-table-memory="disk"
