# R28 ghcr.io/0xsero/deploy-pkg-22-tabbyapi-qwen38-6bpw-v6:v1
# top of the quant ladder; fit failure is a valid result. Owned TabbyAPI deploy image: engine (digest-pinned) + baked config.yml + blocking revision-pinned preflight.
FROM ghcr.io/theroyallab/tabbyapi@sha256:1c13dd17416660856e9962aeee8498827f0acd604eb36795128a03ddfe03c6ef
LABEL org.opencontainers.image.source="https://github.com/0xSero/local-ai-images"
LABEL ai.omarchy.row="R28"
LABEL ai.omarchy.model.repo="turboderp/Qwen3.8-27B-exl3"
LABEL ai.omarchy.model.revision="60d005a257b39ecb25e4ba23c1dd29df877d5c69"
LABEL ai.omarchy.model.local_name="Qwen3.8-27B-EXL3-SC6bpw-H6-V6"
LABEL ai.omarchy.model.mount="/opt/models/Qwen3.8-27B-EXL3-SC6bpw-H6-V6"
LABEL ai.omarchy.model.quant="exl3-6.00bpw"
LABEL ai.omarchy.hardware="rtx-pro-4500-blackwell-32gb"
LABEL ai.omarchy.target_context_tokens="262144"
LABEL ai.omarchy.engine_base_cuda_floor="13.2"
LABEL ai.omarchy.mtp="on"
LABEL ai.omarchy.pinned_startup="model_name Qwen3.8-27B-EXL3-SC6bpw-H6-V6 max_seq_len 262144 cache_size 263168 chunk_size 2048 draft_mode mtp"
COPY config.yml /app/config.yml
COPY deploy-preflight.sh /opt/deploy-preflight.sh
RUN chmod 0755 /opt/deploy-preflight.sh
ENTRYPOINT ["/opt/deploy-preflight.sh"]
