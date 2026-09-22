# Stock upstream llama.cpp, for the local-ai registry (Omarchy Local AI).
#
# This is plain ggml-org/llama.cpp, unmodified: the image the campaign uses for
# any standard-quant GGUF row (IQ3_M, Q4_K_M, Q5_K_M, ...) so the brief's "run
# the latest llama.cpp" requirement is met. It is deliberately NOT the
# llamacpp-prism fork: Prism exists only because Ternary-Bonsai-2-27B ships
# custom PTQ1_0/PQ2_0 ternary packings that upstream rejects; every other GGUF
# must run on real upstream, and this image is it.
#
# Pins ggml-org/llama.cpp release b11112 (latest at time of build, checked
# 2026-09-22). The asset is the official ubuntu CUDA 12.8 x64 tarball, which
# links against the base image's CUDA runtime rather than bundling its own.
#
# Base is the CUDA 12.8 runtime, not 13.x: the pinned tarball links CUDA 12.8
# and runs on every driver from 525.60.13 up, which is what the campaign's
# rented cards have. Kept identical to the Prism image so both llama.cpp images
# share one CUDA floor.
#
# Build: pushed by .github/workflows/release-image.yml with BuildKit provenance,
# an SBOM, and a GitHub build attestation on the digest.
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04@sha256:828c4d878adcaa4265d80c95d8ec877149b49bb2419a4cf3bb6aa889bbb7ca2e

ENV DEBIAN_FRONTEND=noninteractive
# libgomp1 for the OpenMP runtime, libcurl4 for the built-in model fetcher, and
# openssh-server so the validation bootstrap does not have to install it before
# the first SSH check.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl libgomp1 libcurl4 openssh-server \
    && rm -rf /var/lib/apt/lists/*

# checksum= is the digest GitHub publishes for this exact release asset, verified
# against a local sha256sum of the download on 2026-09-22.
ARG LLAMA_URL=https://github.com/ggml-org/llama.cpp/releases/download/b11112/llama-b11112-bin-ubuntu-cuda-12.8-x64.tar.gz
ADD --checksum=sha256:6bf0a2220122c42baff971210fbdddf33df14b3b4dd0e51ac036a66118694951 \
    ${LLAMA_URL} /tmp/llama.tar.gz
RUN mkdir -p /opt/llama \
    && tar -xzf /tmp/llama.tar.gz -C /opt/llama --strip-components=1 \
    && rm /tmp/llama.tar.gz

COPY entrypoint.sh /opt/llama/entrypoint.sh
RUN chmod 0755 /opt/llama/entrypoint.sh

ENV LD_LIBRARY_PATH=/opt/llama
ENV PATH=/opt/llama:$PATH
EXPOSE 8080

# The dispatcher exists so one image serves two callers with one argv contract:
# the registry recipe passes /opt/llama/llama-server and its flags as separate
# arguments, and the campaign's SSH bootstrap passes a single shell string.
ENTRYPOINT ["/opt/llama/entrypoint.sh"]
