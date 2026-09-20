# R02 ghcr.io/0xsero/deploy-sglang-minicpm5-gptq-rtx-3070-8gb:v1
# Pinned SGLang deploy layer over the campaign sglang engine image (digest-pinned).
# Model fetched from HF at first launch by pinned revision. Build-status only; runtime untested.
FROM ghcr.io/0xsero/sglang@sha256:787461f889674cdd9339be2409a6e8b46ec2f5aac71669481c319f87c2beca13
LABEL org.opencontainers.image.source="https://github.com/0xSero/local-ai-images"
LABEL ai.omarchy.row="R02"
LABEL ai.omarchy.model.repo="openbmb/MiniCPM5-2B-GPTQ"
LABEL ai.omarchy.model.revision="48d57c14fba6b1f86109e1a54fafb0848c163f04"
LABEL ai.omarchy.model.branch="main"
LABEL ai.omarchy.hardware="rtx-3070-8gb"
LABEL ai.omarchy.target_context_tokens="32768"
LABEL ai.omarchy.engine_base_cuda_floor="13.0"
LABEL ai.omarchy.pinned_startup="sglang.launch_server --context-length 32768 --mem-fraction-static 0.85 --tool-call-parser minicpm5 --reasoning-parser qwen3"
CMD ["python3","-u","-m","sglang.launch_server","--model-path","openbmb/MiniCPM5-2B-GPTQ","--revision","48d57c14fba6b1f86109e1a54fafb0848c163f04","--context-length","32768","--mem-fraction-static","0.85","--tool-call-parser","minicpm5","--reasoning-parser","qwen3","--host","0.0.0.0","--port","30000"]
