# TabbyAPI + ExLlamaV3 on a CUDA 12.8 base, for the 13.0-only consumer GPU pool.
#
# tabbyapi-exl3 is built on the cu13 tag (CUDA 13.2, NVIDIA_REQUIRE_CUDA cuda>=13.2).
# The campaign's 12 GB / 16 GB community cards (RTX 3080 Ti, RTX 5080, and today's
# RTX 2000 Ada snapshot) only expose a CUDA 13.0 (or 12.8) host, so nvidia-container
# refuses any cuda>=13.2 image on them: the EXL3 Qwen rows are blocked by the image's
# CUDA floor, not by capacity. Upstream ships a 12.8 line whose floor is cuda>=12.8,
# which every one of those hosts satisfies.
#
# Same rationale as tabbyapi-exl3 for the extra layer: ExLlamaV3 JIT-compiles its
# gated-delta-net CUDA utility (Qwen3.5, Qwen3.8) against Python.h through Triton, so
# python3-dev + build-essential are added. The fix is idempotent, so it is kept even if
# a given upstream revision already carries the headers.
#
# Base is latest-extras (the extras variant bundles the additional ExLlamaV3 kernels),
# CUDA 12.8.1, NVIDIA_REQUIRE_CUDA cuda>=12.8, linux/amd64, inspected 2026-09-20.
#
# Build: pushed by .github/workflows/release-image.yml with BuildKit provenance and a
# GitHub build attestation on the digest.
FROM ghcr.io/theroyallab/tabbyapi:latest-extras@sha256:b7e500f9b1ee91346226b27850997880e22186254c0ff80359059bfbf7c99fbc

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3-dev build-essential \
    && rm -rf /var/lib/apt/lists/*

# Entrypoint, command, working directory, and venv PATH are inherited:
# ENTRYPOINT ["python3"]  CMD ["main.py", "--host", "0.0.0.0"]  WORKDIR /app
