# R30 ghcr.io/0xsero/deploy-pkg-24-sglang-qwen38-awq-int4:v1
# no measured tool parser -> capabilities.tools=unverified. Owned sglang deploy image: engine (digest-pinned) + blocking revision-pinned preflight; serve local /opt/models path. Build-status only.
FROM ghcr.io/0xsero/sglang@sha256:787461f889674cdd9339be2409a6e8b46ec2f5aac71669481c319f87c2beca13
LABEL org.opencontainers.image.source="https://github.com/0xSero/local-ai-images"
LABEL ai.omarchy.row="R30"
LABEL ai.omarchy.package="PKG-24"
LABEL ai.omarchy.model.repo="cyankiwi/Qwen3.8-27B-AWQ-INT4"
LABEL ai.omarchy.model.revision="63768c10df38c0395e12ef49edac1bd539eaeeea"
LABEL ai.omarchy.model.local_name="Qwen3.8-27B-AWQ-INT4"
LABEL ai.omarchy.model.mount="/opt/models/Qwen3.8-27B-AWQ-INT4"
LABEL ai.omarchy.model.quant="awq-int4"
LABEL ai.omarchy.hardware="rtx-pro-4500-blackwell-32gb"
LABEL ai.omarchy.target_context_tokens="40960"
LABEL ai.omarchy.engine_base_cuda_floor="13.0"
LABEL ai.omarchy.capabilities.tools="unverified"
LABEL ai.omarchy.model.served_name="Qwen3.8-27B"
LABEL ai.omarchy.pinned_startup="/opt/sglang/bin/python3 -m sglang.launch_server --model-path /opt/models/Qwen3.8-27B-AWQ-INT4 --served-model-name Qwen3.8-27B --host 0.0.0.0 --port 30000 --context-length 40960 --mem-fraction-static 0.85 --reasoning-parser qwen3"
COPY deploy-preflight.sh /opt/deploy-preflight.sh
COPY omp-acquire.sh /opt/omp-acquire
RUN chmod 0755 /opt/deploy-preflight.sh /opt/omp-acquire
ENTRYPOINT ["/opt/deploy-preflight.sh"]
CMD ["/opt/sglang/bin/python3", "-m", "sglang.launch_server", "--enable-cache-report", "--model-path", "/opt/models/Qwen3.8-27B-AWQ-INT4", "--served-model-name", "Qwen3.8-27B", "--host", "0.0.0.0", "--port", "30000", "--context-length", "40960", "--mem-fraction-static", "0.85", "--reasoning-parser", "qwen3"]
