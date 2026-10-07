# Intel's llm-scaler vLLM for Arc / Arc Pro (XPU), unchanged, for the local-ai registry (Omarchy Local AI).
#
# Base: intel/llm-scaler-vllm:0.26.0-b2 (vLLM 0.26.1.dev0+g568afb3a1.xpu, oneAPI, oneCCL 2021.15), pinned by digest.
# The only addition is the entrypoint: it links the render nodes the container was given under /dev/dri/by-path before
# the base entrypoint's oneAPI/oneCCL setup and `vllm serve "$@"`, so the image runs with just `--device
# /dev/dri/renderDNNN` (no by-path bind mount, no --ipc host, no capability or seccomp flags). Measured on one Arc Pro
# B70 with Qwen3.6-35B-A3B GPTQ Int4 + MTP (local-ai-registry lab gates: decode 98.9 tok/s, prefill 1,609 tok/s).
#
# Build: pushed by .github/workflows/release-image.yml with BuildKit provenance and a GitHub build attestation.
FROM intel/llm-scaler-vllm@sha256:52218ad85513ab6686d4c090c83c2bd8c5b02423c63aa4dabd41837fe641fe3b

LABEL org.opencontainers.image.source="https://github.com/sybil-solutions/local-ai-images"
LABEL ai.omarchy.engine="vllm"
LABEL ai.omarchy.engine.version="0.26.1.dev0+g568afb3a1.xpu"
LABEL ai.omarchy.hardware="intel-arc-pro-b70-32gb"

COPY --chmod=0755 xpu-entrypoint.sh /opt/omarchy/xpu-entrypoint.sh
ENTRYPOINT ["/opt/omarchy/xpu-entrypoint.sh"]
