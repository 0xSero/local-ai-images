# ghcr.io/sybil-solutions/glm53-b70-expert-server: the B70 expert server for GLM-5.3-Flash's optional B70 expert tier
# (glm53-flash-offload GLM53_B70=1). One Intel Arc Pro B70 32 GB holds 3,000 routed experts (28.4 GB) next to the RTX
# 3090 engine (ghcr.io/sybil-solutions/glm53-flash-offload, GLM53_MODE=nvme) in the same machine. Serving code:
# github.com/sybil-solutions/glm-flash-lite b70tier/ at the pinned commit below.
#   Started FIRST. It creates the ring /run/local-ai/shared/b70.ring (GLM53_B70_RING) in a host tmpfs dir that both
#   containers mount. The engine attaches at warm-up and sends the expert keys. The server reads them from the packed
#   NVMe store at /nvx (O_DIRECT, the engine's store, read-only), then computes the B70 picks of every decode MoE layer
#   call with exl3xpu's grouped MoE kernel (_moe_n128.so: pointer-table experts, swiglu clamp 10).
#   Measured (campaign N137, omarchy, 2026-10-09, same session, 55 GiB cap): decode C1 17.48 -> 26.40-26.61, C2 17.84 ->
#   30.34-30.63, C4 19.11 -> 33.00-33.54, 32k C1 16.28 -> 24.60-24.92, 32k C2 17.35 -> 29.09 tok/s; prefill unchanged
#   (exact); paired decode-KL 0.0034 / top-1 0.988 (shipped nvme mode 0.0070 / 0.980).
# Device: one render node (--device /dev/dri/renderDNNN, ZE_AFFINITY_MASK=0), --memory 16g; no network, no --ipc host.
#
# Base = the exact image the numbers were measured on (ghcr.io/0xsero/sglang-exl3-xpu-flashnext, trellis-serve 72fd2d4,
# torch 2.13 XPU, oneAPI 2026.0). This layer builds _moe_n128.so (spir64, the campaign's flags) from the repo sources.
# Build: pushed by .github/workflows/release-image.yml with BuildKit provenance, an SBOM, and a build attestation.
FROM ghcr.io/0xsero/sglang-exl3-xpu-flashnext@sha256:24c872759256d4b0d3b9d0419ebdf6f3547e0fdbd3891fbb9f90e32ad255c94f

LABEL org.opencontainers.image.source="https://github.com/sybil-solutions/local-ai-images"
LABEL ai.omarchy.engine="glm53-b70-expert-server"
LABEL ai.omarchy.quant="exl3"
LABEL ai.omarchy.hardware="intel-arc-pro-b70-32gb"
LABEL ai.omarchy.role="companion"
LABEL ai.omarchy.model.repo="turboderp/GLM-5.3-Flash-exl3"
LABEL ai.omarchy.model.revision="332ab457b709b7ba30dd9a448be5de03b80a7ac9"

ARG GLM53_REPO=https://github.com/sybil-solutions/glm-flash-lite
ARG GLM53_COMMIT=afdf39062a3f8aa3497c5013b2993ac017f54822
LABEL ai.omarchy.glm53-flash-offload.commit="${GLM53_COMMIT}"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN git init -q /opt/glm53 \
 && cd /opt/glm53 \
 && git fetch -q --depth 1 "${GLM53_REPO}" "${GLM53_COMMIT}" \
 && git checkout -q --detach FETCH_HEAD \
 && test "$(git rev-parse HEAD)" = "${GLM53_COMMIT}" \
 && echo "${GLM53_COMMIT}" > /opt/glm53/COMMIT \
 && rm -rf .git

RUN bash /opt/glm53/b70tier/kernels/moe-n128/build.sh \
 && test -s /opt/glm53/b70tier/kernels/moe-n128/_moe_n128.so

ENV GLM53_B70_MOE_LIB=/opt/glm53/b70tier/kernels/moe-n128/_moe_n128.so \
    GLM53_B70_RING=/run/local-ai/shared/b70.ring \
    GLM53_NV_STORE=/nvx/glm53_flash_exl3_3.05bpw_experts.bin \
    NEOReadDebugKeys=1 EnableSharedSystemUsmSupport=1 EnableRecoverablePageFaults=1 \
    PYTHONUNBUFFERED=1

# The kernel library loads next to torch the way the server loads it, and the server module imports (CPU only).
RUN cd /opt/glm53/b70tier && python3 -c "import os, sys, importlib.util; sys.path.insert(0, '.'); \
spec = importlib.util.spec_from_file_location('b70srv', 'b70srv.py'); m = importlib.util.module_from_spec(spec); \
spec.loader.exec_module(m); m.moe_offload.ops(); \
assert os.environ['EXL3_MOE_LIB'] == os.environ['GLM53_B70_MOE_LIB'], os.environ['EXL3_MOE_LIB']; \
assert m.make_perm(4096, 2048, 3).numel() == 2368512; print('glm53 B70 expert server: library loads, module imports')"

WORKDIR /opt/glm53/b70tier
ENTRYPOINT ["python3", "/opt/glm53/b70tier/b70srv.py"]
