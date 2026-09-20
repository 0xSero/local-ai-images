# R12 ghcr.io/0xsero/deploy-sglang-minicpm5-awq-rtx-4080-16gb:v1
# Pinned SGLang deploy layer over the campaign sglang engine image (digest-pinned).
# Model fetched from HF at first launch by pinned revision. Build-status only; runtime untested.
FROM ghcr.io/0xsero/sglang@sha256:787461f889674cdd9339be2409a6e8b46ec2f5aac71669481c319f87c2beca13
LABEL org.opencontainers.image.source="https://github.com/0xSero/local-ai-images"
LABEL ai.omarchy.row="R12"
LABEL ai.omarchy.model.repo="cyankiwi/MiniCPM5-2B-AWQ-INT4"
LABEL ai.omarchy.model.revision="bc59eebcb2e12f5e1e714c57a4b6b7527f6cd845"
LABEL ai.omarchy.model.branch="main"
LABEL ai.omarchy.hardware="rtx-4080-16gb"
LABEL ai.omarchy.target_context_tokens="131072"
LABEL ai.omarchy.engine_base_cuda_floor="13.0"
LABEL ai.omarchy.pinned_startup="sglang.launch_server --context-length 131072 --mem-fraction-static 0.85"
CMD ["python3","-u","-m","sglang.launch_server","--model-path","cyankiwi/MiniCPM5-2B-AWQ-INT4","--revision","bc59eebcb2e12f5e1e714c57a4b6b7527f6cd845","--context-length","131072","--mem-fraction-static","0.85","--host","0.0.0.0","--port","30000"]
