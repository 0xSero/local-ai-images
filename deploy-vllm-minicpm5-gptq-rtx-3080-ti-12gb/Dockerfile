# R06 ghcr.io/0xsero/deploy-vllm-minicpm5-gptq-rtx-3080-ti-12gb:v1
# Pinned vLLM deploy layer over the campaign vllm-openai engine image (digest-pinned).
# Model fetched from HF at first launch by pinned revision. Build-status only; runtime untested.
FROM ghcr.io/0xsero/vllm-openai@sha256:1de88ff218282ce71da7811f62ea527556648e1ad05dd36e5fc82d0f9ff07424
LABEL org.opencontainers.image.source="https://github.com/0xSero/local-ai-images"
LABEL ai.omarchy.row="R06"
LABEL ai.omarchy.model.repo="openbmb/MiniCPM5-2B-GPTQ"
LABEL ai.omarchy.model.revision="48d57c14fba6b1f86109e1a54fafb0848c163f04"
LABEL ai.omarchy.model.branch="main"
LABEL ai.omarchy.hardware="rtx-3080-ti-12gb"
LABEL ai.omarchy.target_context_tokens="131072"
LABEL ai.omarchy.engine_base_cuda_floor="13.0"
LABEL ai.omarchy.pinned_startup="vllm serve --max-model-len 131072 --kv-cache-dtype fp8 --gpu-memory-utilization 0.92"
CMD ["vllm","serve","openbmb/MiniCPM5-2B-GPTQ","--revision","48d57c14fba6b1f86109e1a54fafb0848c163f04","--max-model-len","131072","--kv-cache-dtype","fp8","--gpu-memory-utilization","0.92","--host","0.0.0.0","--port","8000"]
