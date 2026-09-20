# R24 ghcr.io/0xsero/deploy-tabbyapi-qwen38-5bpw-v6-rtx-5090-32gb:v1
# Owned TabbyAPI deploy image: engine (digest-pinned) + baked /app/config.yml + blocking model preflight.
# ENTRYPOINT fetches turboderp/Qwen3.8-27B-exl3@f33f26d929e2 into /opt/models/Qwen3.8-27B-EXL3-SC5bpw-H6-V6, then launches tabby with the
# baked config. Build-status only; runtime untested here.
FROM ghcr.io/0xsero/tabbyapi-exl3-cu12@sha256:c0850ea09fcf8bb21a289ae0c5fe8ba29a1658a77257cf80a2308da3e4a3ec2c
LABEL org.opencontainers.image.source="https://github.com/0xSero/local-ai-images"
LABEL ai.omarchy.row="R24"
LABEL ai.omarchy.model.repo="turboderp/Qwen3.8-27B-exl3"
LABEL ai.omarchy.model.revision="f33f26d929e2b20ef21361145d582f5239e3831f"
LABEL ai.omarchy.model.local_name="Qwen3.8-27B-EXL3-SC5bpw-H6-V6"
LABEL ai.omarchy.model.mount="/opt/models/Qwen3.8-27B-EXL3-SC5bpw-H6-V6"
LABEL ai.omarchy.model.quant="exl3-5.00bpw"
LABEL ai.omarchy.hardware="rtx-5090-32gb"
LABEL ai.omarchy.target_context_tokens="262144"
LABEL ai.omarchy.engine_base_cuda_floor="12.8"
LABEL ai.omarchy.pinned_startup="tabbyapi config.yml max_seq_len 262144, cache_size 263168, chunk_size 2048"
COPY config.yml /app/config.yml
COPY deploy-preflight.sh /opt/deploy-preflight.sh
RUN chmod 0755 /opt/deploy-preflight.sh
ENTRYPOINT ["/opt/deploy-preflight.sh"]
