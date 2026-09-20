# R17 ghcr.io/0xsero/deploy-sglang-ornith15-9b-l4-24gb:v1
# Owned SGLang deploy image: engine (digest-pinned) + blocking revision-pinned model preflight.
# ENTRYPOINT fetches ornith-ai/Ornith-1.5-9B@489cb97981b8 into /opt/models/Ornith-1.5-9B, writes .omp-ready, then delegates
# to the base /opt/entrypoint.sh shim ("$@" = passed --docker-args OR the serve CMD below). Build-status only.
FROM ghcr.io/0xsero/sglang@sha256:787461f889674cdd9339be2409a6e8b46ec2f5aac71669481c319f87c2beca13
LABEL org.opencontainers.image.source="https://github.com/0xSero/local-ai-images"
LABEL ai.omarchy.row="R17"
LABEL ai.omarchy.model.repo="ornith-ai/Ornith-1.5-9B"
LABEL ai.omarchy.model.revision="489cb97981b8654bcfcf30ce1f94ed1b62e07b53"
LABEL ai.omarchy.model.local_name="Ornith-1.5-9B"
LABEL ai.omarchy.model.mount="/opt/models/Ornith-1.5-9B"
LABEL ai.omarchy.model.quant="bf16"
LABEL ai.omarchy.hardware="l4-24gb"
LABEL ai.omarchy.target_context_tokens="131072"
LABEL ai.omarchy.engine_base_cuda_floor="13.0"
LABEL ai.omarchy.pinned_startup="sglang.launch_server --context-length 131072 --mem-fraction-static 0.85 --tool-call-parser qwen3_coder --reasoning-parser qwen3"
COPY deploy-preflight.sh /opt/deploy-preflight.sh
RUN chmod 0755 /opt/deploy-preflight.sh
ENTRYPOINT ["/opt/deploy-preflight.sh"]
CMD ["/opt/sglang/bin/python3", "-m", "sglang.launch_server", "--model-path", "/opt/models/Ornith-1.5-9B", "--served-model-name", "Ornith-1.5-9B", "--host", "0.0.0.0", "--port", "30000", "--context-length", "131072", "--mem-fraction-static", "0.85", "--tool-call-parser", "qwen3_coder", "--reasoning-parser", "qwen3"]
