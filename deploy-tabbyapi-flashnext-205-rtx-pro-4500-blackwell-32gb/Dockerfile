# R20 ghcr.io/0xsero/deploy-tabbyapi-flashnext-205-rtx-pro-4500-blackwell-32gb:v1
# Pinned TabbyAPI deploy layer: engine image (digest-pinned) + baked /app/config.yml.
# Model weights fetched by the recipe into /models; config pins the serving window. Build-status only; runtime untested.
FROM ghcr.io/theroyallab/tabbyapi@sha256:1c13dd17416660856e9962aeee8498827f0acd604eb36795128a03ddfe03c6ef
LABEL org.opencontainers.image.source="https://github.com/0xSero/local-ai-images"
LABEL ai.omarchy.row="R20"
LABEL ai.omarchy.model.repo="turboderp/Qwen3.8-Flash-Next-exl3"
LABEL ai.omarchy.model.revision="65c895314393431c09050b2e04e250836b3a6eb4"
LABEL ai.omarchy.model.branch="2.05bpw_h4_ng4"
LABEL ai.omarchy.hardware="rtx-pro-4500-blackwell-32gb"
LABEL ai.omarchy.target_context_tokens="262144"
LABEL ai.omarchy.engine_base_cuda_floor="13.2"
LABEL ai.omarchy.pinned_startup="tabbyapi config.yml draft_mode mtp, cpu_moe_split_experts 128, ngram_ram true, max_seq_len 262144, cache_size 262656"
COPY config.yml /app/config.yml
