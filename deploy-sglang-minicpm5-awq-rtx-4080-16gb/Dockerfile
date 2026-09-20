# R12 ghcr.io/0xsero/deploy-sglang-minicpm5-awq-rtx-4080-16gb:v1
# Owned SGLang deploy image: engine (digest-pinned) + blocking revision-pinned model preflight.
# ENTRYPOINT fetches cyankiwi/MiniCPM5-2B-AWQ-INT4@bc59eebcb2e1 into /opt/models/MiniCPM5-2B-AWQ-INT4, writes .omp-ready, then delegates
# to the base /opt/entrypoint.sh shim ("$@" = passed --docker-args OR the serve CMD below). Build-status only.
FROM ghcr.io/0xsero/sglang@sha256:787461f889674cdd9339be2409a6e8b46ec2f5aac71669481c319f87c2beca13
LABEL org.opencontainers.image.source="https://github.com/0xSero/local-ai-images"
LABEL ai.omarchy.row="R12"
LABEL ai.omarchy.model.repo="cyankiwi/MiniCPM5-2B-AWQ-INT4"
LABEL ai.omarchy.model.revision="bc59eebcb2e12f5e1e714c57a4b6b7527f6cd845"
LABEL ai.omarchy.model.local_name="MiniCPM5-2B-AWQ-INT4"
LABEL ai.omarchy.model.mount="/opt/models/MiniCPM5-2B-AWQ-INT4"
LABEL ai.omarchy.model.quant="compressed-tensors-int4"
LABEL ai.omarchy.hardware="rtx-4080-16gb"
LABEL ai.omarchy.target_context_tokens="131072"
LABEL ai.omarchy.engine_base_cuda_floor="13.0"
LABEL ai.omarchy.pinned_startup="sglang.launch_server --enable-cache-report --context-length 131072 --mem-fraction-static 0.85"
COPY deploy-preflight.sh /opt/deploy-preflight.sh
COPY omp-acquire.sh /opt/omp-acquire
RUN chmod 0755 /opt/deploy-preflight.sh /opt/omp-acquire
ENTRYPOINT ["/opt/deploy-preflight.sh"]
CMD ["/opt/sglang/bin/python3", "-m", "sglang.launch_server", "--enable-cache-report", "--model-path", "/opt/models/MiniCPM5-2B-AWQ-INT4", "--served-model-name", "MiniCPM5-2B-AWQ-INT4", "--host", "0.0.0.0", "--port", "30000", "--context-length", "131072", "--mem-fraction-static", "0.85", "--tool-call-parser", "minicpm5", "--reasoning-parser", "qwen3"]
