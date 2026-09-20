# R17 ghcr.io/0xsero/deploy-sglang-ornith15-9b-l4-24gb:v1
# Pinned SGLang deploy layer over the campaign sglang engine image (digest-pinned).
# Model fetched from HF at first launch by pinned revision. Build-status only; runtime untested.
FROM ghcr.io/0xsero/sglang@sha256:787461f889674cdd9339be2409a6e8b46ec2f5aac71669481c319f87c2beca13
LABEL org.opencontainers.image.source="https://github.com/0xSero/local-ai-images"
LABEL ai.omarchy.row="R17"
LABEL ai.omarchy.model.repo="ornith-ai/Ornith-1.5-9B"
LABEL ai.omarchy.model.revision="489cb97981b8654bcfcf30ce1f94ed1b62e07b53"
LABEL ai.omarchy.model.branch="main"
LABEL ai.omarchy.hardware="l4-24gb"
LABEL ai.omarchy.target_context_tokens="131072"
LABEL ai.omarchy.engine_base_cuda_floor="13.0"
LABEL ai.omarchy.pinned_startup="sglang.launch_server --context-length 131072 --mem-fraction-static 0.85 --tool-call-parser qwen3_coder --reasoning-parser qwen3"
CMD ["python3","-u","-m","sglang.launch_server","--model-path","ornith-ai/Ornith-1.5-9B","--revision","489cb97981b8654bcfcf30ce1f94ed1b62e07b53","--context-length","131072","--mem-fraction-static","0.85","--tool-call-parser","qwen3_coder","--reasoning-parser","qwen3","--host","0.0.0.0","--port","30000"]
