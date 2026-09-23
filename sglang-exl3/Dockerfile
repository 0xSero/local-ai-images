# ghcr.io/0xsero/sglang-exl3 — stock SGLang v0.5.20 + EXL3 (ExLlamaV3 trellis) quantization for Ampere (sm_86) and up.
# Adds: exllamav3 1.5.1 extension built for the arch list below, the sglang-exl3 plugin (registers `--quantization exl3`
# through the sglang.srt.plugins entry point, no SGLang source edits) and its Marlin-template EXL3 kernels (K=3/4/6).
# Same shim entrypoint contract as ./sglang (argv exec'd directly; one shell string -> /bin/sh -c).
# Pinned digest is lmsysorg/sglang v0.5.20, linux/amd64, inspected 2026-09-23.
FROM lmsysorg/sglang@sha256:06e4f2ed21afde4ff513cda65070124e727ba23ccaeff7712b8c40e1097d611f
LABEL org.opencontainers.image.source="https://github.com/0xSero/local-ai-images"
LABEL ai.omarchy.engine="sglang"
LABEL ai.omarchy.engine.version="0.5.20"
LABEL ai.omarchy.quant="exl3"
ARG TORCH_CUDA_ARCH_LIST="8.6"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ninja-build openssh-server \
 && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /run/sshd
# ExLlamaV3 v1.5.1 (MIT, turboderp) extension, AOT for the arch list, no extra Python deps.
ARG EXLLAMAV3_COMMIT=6b84a21b6f1e5da3f291b9e1019061f0de788279
RUN git clone https://github.com/turboderp-org/exllamav3 /opt/exllamav3-src \
 && cd /opt/exllamav3-src && git checkout ${EXLLAMAV3_COMMIT} \
 && MAX_JOBS=4 pip install --no-build-isolation --no-deps -v . 2>&1 | tail -3 \
 && pip install --no-deps marisa-trie \
 && rm -rf /opt/exllamav3-src
# The plugin and its kernels (source snapshot vendored in this directory; see VERSION for the upstream commit).
COPY src /opt/sglang-exl3
RUN cd /opt/sglang-exl3/csrc && MAX_JOBS=4 AIKIDO_MOE=1 python3 setup.py build_ext --build-lib build/lib 2>&1 | tail -3 \
 && rm -rf build/temp.* build/gen \
 && pip install --no-deps /opt/sglang-exl3
ENV PYTHONPATH=/opt/sglang-exl3/csrc/build/lib
COPY entrypoint.sh /opt/entrypoint.sh
RUN chmod 0755 /opt/entrypoint.sh
EXPOSE 30000
EXPOSE 22
ENTRYPOINT ["/opt/entrypoint.sh"]
