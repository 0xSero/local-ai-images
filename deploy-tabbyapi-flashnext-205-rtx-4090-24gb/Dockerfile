# R16 ghcr.io/0xsero/deploy-tabbyapi-flashnext-205-rtx-4090-24gb:v1
# Owned TabbyAPI deploy image: engine (digest-pinned) + baked /app/config.yml + blocking model preflight.
# ENTRYPOINT fetches turboderp/Qwen3.8-Flash-Next-exl3@65c895314393 into /opt/models/Qwen3.8-Flash-Next-EXL3-205bpw-H4, then launches tabby with the
# baked config. Build-status only; runtime untested here.
FROM ghcr.io/theroyallab/tabbyapi@sha256:1c13dd17416660856e9962aeee8498827f0acd604eb36795128a03ddfe03c6ef
LABEL org.opencontainers.image.source="https://github.com/0xSero/local-ai-images"
LABEL ai.omarchy.row="R16"
LABEL ai.omarchy.model.repo="turboderp/Qwen3.8-Flash-Next-exl3"
LABEL ai.omarchy.model.revision="65c895314393431c09050b2e04e250836b3a6eb4"
LABEL ai.omarchy.model.local_name="Qwen3.8-Flash-Next-EXL3-205bpw-H4"
LABEL ai.omarchy.model.mount="/opt/models/Qwen3.8-Flash-Next-EXL3-205bpw-H4"
LABEL ai.omarchy.model.quant="exl3-2.05bpw"
LABEL ai.omarchy.hardware="rtx-4090-24gb"
LABEL ai.omarchy.target_context_tokens="131072"
LABEL ai.omarchy.engine_base_cuda_floor="13.2"
LABEL ai.omarchy.pinned_startup="tabbyapi config.yml cpu_moe_split_experts 128, ngram_ram true, max_seq_len 131072, cache_size 131584"
COPY config.yml /app/config.yml
COPY deploy-preflight.sh /opt/deploy-preflight.sh
RUN chmod 0755 /opt/deploy-preflight.sh
ENTRYPOINT ["/opt/deploy-preflight.sh"]
