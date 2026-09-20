# R14 ghcr.io/0xsero/deploy-tabbyapi-qwen38-4bpw-rtx-pro-4000-blackwell-24gb:v1
# Pinned TabbyAPI deploy layer: engine image (digest-pinned) + baked /app/config.yml.
# Model weights fetched by the recipe into /models; config pins the serving window. Build-status only; runtime untested.
FROM ghcr.io/theroyallab/tabbyapi@sha256:1c13dd17416660856e9962aeee8498827f0acd604eb36795128a03ddfe03c6ef
LABEL org.opencontainers.image.source="https://github.com/0xSero/local-ai-images"
LABEL ai.omarchy.row="R14"
LABEL ai.omarchy.model.repo="turboderp/Qwen3.8-27B-exl3"
LABEL ai.omarchy.model.revision="4acd9ad5af224ec9e8815a54d71a033f378a7e9d"
LABEL ai.omarchy.model.branch="SC_4.00bpw_H5"
LABEL ai.omarchy.hardware="rtx-pro-4000-blackwell-24gb"
LABEL ai.omarchy.target_context_tokens="131072"
LABEL ai.omarchy.engine_base_cuda_floor="13.2"
LABEL ai.omarchy.pinned_startup="tabbyapi config.yml max_seq_len 131072, cache_size 131584, chunk_size 2048"
COPY config.yml /app/config.yml
