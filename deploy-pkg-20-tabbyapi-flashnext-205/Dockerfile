# R26 ghcr.io/0xsero/deploy-pkg-20-tabbyapi-flashnext-205:v1
# MTP OFF control vs R20; distinguishes inert/mis-detected/costly MTP. Owned TabbyAPI deploy image: engine (digest-pinned) + baked config.yml + blocking revision-pinned preflight.
FROM ghcr.io/theroyallab/tabbyapi@sha256:1c13dd17416660856e9962aeee8498827f0acd604eb36795128a03ddfe03c6ef
LABEL org.opencontainers.image.source="https://github.com/0xSero/local-ai-images"
LABEL ai.omarchy.row="R26"
LABEL ai.omarchy.package="PKG-20"
LABEL ai.omarchy.model.repo="turboderp/Qwen3.8-Flash-Next-exl3"
LABEL ai.omarchy.model.revision="65c895314393431c09050b2e04e250836b3a6eb4"
LABEL ai.omarchy.model.local_name="Qwen3.8-Flash-Next-EXL3-205bpw-H4"
LABEL ai.omarchy.model.mount="/opt/models/Qwen3.8-Flash-Next-EXL3-205bpw-H4"
LABEL ai.omarchy.model.quant="exl3-2.05bpw"
LABEL ai.omarchy.hardware="rtx-pro-4500-blackwell-32gb"
LABEL ai.omarchy.target_context_tokens="262144"
LABEL ai.omarchy.engine_base_cuda_floor="13.2"
LABEL ai.omarchy.mtp="off"
LABEL ai.omarchy.pinned_startup="model_name Qwen3.8-Flash-Next-EXL3-205bpw-H4 max_seq_len 262144 cache_size 262656 chunk_size 2048 draft_mode none"
COPY config.yml /app/config.yml
COPY deploy-preflight.sh /opt/deploy-preflight.sh
COPY omp-acquire.sh /opt/omp-acquire
RUN chmod 0755 /opt/deploy-preflight.sh /opt/omp-acquire
ENTRYPOINT ["/opt/deploy-preflight.sh"]
