# R18 ghcr.io/0xsero/deploy-vllm-ornith15-9b-l4-24gb:v1
# Owned vLLM deploy image: engine (digest-pinned) + blocking revision-pinned model preflight.
# ENTRYPOINT fetches ornith-ai/Ornith-1.5-9B@489cb97981b8 into /opt/models/Ornith-1.5-9B, then delegates to base shim.
# Build-status only; runtime untested here.
FROM ghcr.io/0xsero/vllm-openai@sha256:1de88ff218282ce71da7811f62ea527556648e1ad05dd36e5fc82d0f9ff07424
LABEL org.opencontainers.image.source="https://github.com/0xSero/local-ai-images"
LABEL ai.omarchy.row="R18"
LABEL ai.omarchy.model.repo="ornith-ai/Ornith-1.5-9B"
LABEL ai.omarchy.model.revision="489cb97981b8654bcfcf30ce1f94ed1b62e07b53"
LABEL ai.omarchy.model.local_name="Ornith-1.5-9B"
LABEL ai.omarchy.model.mount="/opt/models/Ornith-1.5-9B"
LABEL ai.omarchy.model.quant="bf16"
LABEL ai.omarchy.hardware="l4-24gb"
LABEL ai.omarchy.target_context_tokens="131072"
LABEL ai.omarchy.engine_base_cuda_floor="13.0"
LABEL ai.omarchy.pinned_startup="vllm serve --max-model-len 131072 --kv-cache-dtype fp8 --gpu-memory-utilization 0.92 --tool-call-parser qwen3_xml"
COPY deploy-preflight.sh /opt/deploy-preflight.sh
RUN chmod 0755 /opt/deploy-preflight.sh
ENTRYPOINT ["/opt/deploy-preflight.sh"]
CMD ["vllm", "serve", "/opt/models/Ornith-1.5-9B", "--served-model-name", "Ornith-1.5-9B", "--host", "0.0.0.0", "--port", "8000", "--max-model-len", "131072", "--kv-cache-dtype", "fp8", "--gpu-memory-utilization", "0.92", "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_xml", "--reasoning-parser", "qwen3", "--default-chat-template-kwargs", "{\"enable_thinking\": true}", "--trust-remote-code"]
