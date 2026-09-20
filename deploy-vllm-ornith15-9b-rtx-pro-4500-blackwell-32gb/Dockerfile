# R22 ghcr.io/0xsero/deploy-vllm-ornith15-9b-rtx-pro-4500-blackwell-32gb:v1
# Pinned vLLM deploy layer over the campaign vllm-openai engine image (digest-pinned).
# Model fetched from HF at first launch by pinned revision. Build-status only; runtime untested.
FROM ghcr.io/0xsero/vllm-openai@sha256:1de88ff218282ce71da7811f62ea527556648e1ad05dd36e5fc82d0f9ff07424
LABEL org.opencontainers.image.source="https://github.com/0xSero/local-ai-images"
LABEL ai.omarchy.row="R22"
LABEL ai.omarchy.model.repo="ornith-ai/Ornith-1.5-9B"
LABEL ai.omarchy.model.revision="489cb97981b8654bcfcf30ce1f94ed1b62e07b53"
LABEL ai.omarchy.model.branch="main"
LABEL ai.omarchy.hardware="rtx-pro-4500-blackwell-32gb"
LABEL ai.omarchy.target_context_tokens="131072"
LABEL ai.omarchy.engine_base_cuda_floor="13.0"
LABEL ai.omarchy.pinned_startup="vllm serve --max-model-len 131072 --kv-cache-dtype fp8 --gpu-memory-utilization 0.92 --tool-call-parser qwen3_xml --reasoning-parser qwen3"
CMD ["vllm","serve","ornith-ai/Ornith-1.5-9B","--revision","489cb97981b8654bcfcf30ce1f94ed1b62e07b53","--max-model-len","131072","--kv-cache-dtype","fp8","--gpu-memory-utilization","0.92","--tool-call-parser","qwen3_xml","--reasoning-parser","qwen3","--host","0.0.0.0","--port","8000"]
