# R08 ghcr.io/0xsero/deploy-tabbyapi-qwen38-3bpw-v4-rtx-2000-ada-16gb:v1
# Pinned TabbyAPI deploy layer: engine image (digest-pinned) + baked /app/config.yml.
# Model weights fetched by the recipe into /models; config pins the serving window. Build-status only; runtime untested.
FROM ghcr.io/theroyallab/tabbyapi@sha256:1c13dd17416660856e9962aeee8498827f0acd604eb36795128a03ddfe03c6ef
LABEL org.opencontainers.image.source="https://github.com/0xSero/local-ai-images"
LABEL ai.omarchy.row="R08"
LABEL ai.omarchy.model.repo="turboderp/Qwen3.8-27B-exl3"
LABEL ai.omarchy.model.revision="004a887127d8304ca2d5475d3a3c41f1761fdd27"
LABEL ai.omarchy.model.branch="SC_3.00bpw_H4_V4"
LABEL ai.omarchy.hardware="rtx-2000-ada-16gb"
LABEL ai.omarchy.target_context_tokens="131072"
LABEL ai.omarchy.engine_base_cuda_floor="13.2"
LABEL ai.omarchy.pinned_startup="tabbyapi config.yml max_seq_len 131072, cache_size 131584, chunk_size 1024"
COPY config.yml /app/config.yml
