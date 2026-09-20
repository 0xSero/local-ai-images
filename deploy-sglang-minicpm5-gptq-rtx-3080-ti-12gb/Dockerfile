# R05 ghcr.io/0xsero/deploy-sglang-minicpm5-gptq-rtx-3080-ti-12gb:v1
# Owned SGLang deploy image: engine (digest-pinned) + pinned model acquisition.
# Canonical interface: env OMP_SSH_FIRST=1 -> ENTRYPOINT execs the passed sshd bootstrap immediately
# (diagnostic SSH first, no fetch); unset -> blocking /opt/omp-acquire (fail-closed, rev-validated
# .omp-ready marker) then the base /opt/entrypoint.sh shim + baked serve CMD. Build-status only.
FROM ghcr.io/0xsero/sglang@sha256:787461f889674cdd9339be2409a6e8b46ec2f5aac71669481c319f87c2beca13
LABEL org.opencontainers.image.source="https://github.com/0xSero/local-ai-images"
LABEL ai.omarchy.row="R05"
LABEL ai.omarchy.model.repo="openbmb/MiniCPM5-2B-GPTQ"
LABEL ai.omarchy.model.revision="48d57c14fba6b1f86109e1a54fafb0848c163f04"
LABEL ai.omarchy.model.local_name="MiniCPM5-2B-GPTQ"
LABEL ai.omarchy.model.mount="/opt/models/MiniCPM5-2B-GPTQ"
LABEL ai.omarchy.model.quant="awq-4bit"
LABEL ai.omarchy.hardware="rtx-3080-ti-12gb"
LABEL ai.omarchy.target_context_tokens="131072"
LABEL ai.omarchy.engine_base_cuda_floor="13.0"
LABEL ai.omarchy.pinned_startup="sglang.launch_server --context-length 131072 --mem-fraction-static 0.85"
COPY deploy-preflight.sh /opt/deploy-preflight.sh
COPY omp-acquire.sh /opt/omp-acquire
RUN chmod 0755 /opt/deploy-preflight.sh /opt/omp-acquire
ENTRYPOINT ["/opt/deploy-preflight.sh"]
CMD ["/opt/sglang/bin/python3", "-m", "sglang.launch_server", "--model-path", "/opt/models/MiniCPM5-2B-GPTQ", "--served-model-name", "MiniCPM5-2B-GPTQ", "--host", "0.0.0.0", "--port", "30000", "--context-length", "131072", "--mem-fraction-static", "0.85", "--tool-call-parser", "minicpm5", "--reasoning-parser", "qwen3"]
