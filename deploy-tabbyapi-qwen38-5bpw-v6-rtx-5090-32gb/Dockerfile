# R24 ghcr.io/0xsero/deploy-tabbyapi-qwen38-5bpw-v6-rtx-5090-32gb:v1
# Pinned TabbyAPI deploy layer: engine image (digest-pinned) + baked /app/config.yml.
# Model weights fetched by the recipe into /models; config pins the serving window. Build-status only; runtime untested.
FROM ghcr.io/0xsero/tabbyapi-exl3-cu12@sha256:c0850ea09fcf8bb21a289ae0c5fe8ba29a1658a77257cf80a2308da3e4a3ec2c
LABEL org.opencontainers.image.source="https://github.com/0xSero/local-ai-images"
LABEL ai.omarchy.row="R24"
LABEL ai.omarchy.model.repo="turboderp/Qwen3.8-27B-exl3"
LABEL ai.omarchy.model.revision="f33f26d929e2b20ef21361145d582f5239e3831f"
LABEL ai.omarchy.model.branch="SC_5.00bpw_H6_V6"
LABEL ai.omarchy.hardware="rtx-5090-32gb"
LABEL ai.omarchy.target_context_tokens="262144"
LABEL ai.omarchy.engine_base_cuda_floor="12.8"
LABEL ai.omarchy.pinned_startup="tabbyapi config.yml max_seq_len 262144, cache_size 263168, chunk_size 2048"
COPY config.yml /app/config.yml
