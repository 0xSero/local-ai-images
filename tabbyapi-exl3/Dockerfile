# TabbyAPI + ExLlamaV3 for the local-ai registry (Omarchy Local AI).
#
# Upstream image pinned by digest, unchanged except for one thing it lacks:
# the Python development headers. ExLlamaV3 runs its gated-delta-net kernels
# (Qwen3.5, Qwen3.8) through Triton, and Triton JIT-compiles a small CUDA
# utility module against Python.h at first use. Without the headers the model
# load dies with "fatal error: Python.h: No such file or directory". Gemma-4
# never touches that path, which is why only Gemma recipes validated upstream.
#
# Build: pushed by .github/workflows/release-image.yml with BuildKit provenance,
# an SBOM, and a GitHub build attestation on the digest.
FROM ghcr.io/theroyallab/tabbyapi:cu13@sha256:ffa8388f310d3c8a2727c66d14a76bb663fdb99a2a49e68a43c3bebb5a3e53f1

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3-dev build-essential \
    && rm -rf /var/lib/apt/lists/*

# Entrypoint, command, working directory, and venv PATH are inherited:
# ENTRYPOINT ["python3"]  CMD ["main.py", "--host", "0.0.0.0"]  WORKDIR /app
