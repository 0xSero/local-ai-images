# R04 ghcr.io/0xsero/deploy-tabbyapi-qwen38-2bpw-rtx-3080-ti-12gb:v1
# Pinned TabbyAPI deploy layer: engine image (digest-pinned) + baked /app/config.yml.
# Model weights fetched by the recipe into /models; config pins the serving window. Build-status only; runtime untested.
FROM ghcr.io/0xsero/tabbyapi-exl3-cu12@sha256:c0850ea09fcf8bb21a289ae0c5fe8ba29a1658a77257cf80a2308da3e4a3ec2c
LABEL org.opencontainers.image.source="https://github.com/0xSero/local-ai-images"
LABEL ai.omarchy.row="R04"
LABEL ai.omarchy.model.repo="turboderp/Qwen3.8-27B-exl3"
LABEL ai.omarchy.model.revision="e5e1f4b37c77641e465e24adf5f44d01bdd48de2"
LABEL ai.omarchy.model.branch="SC_2.00bpw_H3_V3"
LABEL ai.omarchy.hardware="rtx-3080-ti-12gb"
LABEL ai.omarchy.target_context_tokens="32768"
LABEL ai.omarchy.engine_base_cuda_floor="12.8"
LABEL ai.omarchy.pinned_startup="tabbyapi config.yml max_seq_len 32768, cache_size 33024, chunk_size 1024"
COPY config.yml /app/config.yml
