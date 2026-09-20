# R25 ghcr.io/0xsero/deploy-pkg-19-tabbyapi-qwen38-4bpw:v1
# MTP OFF control vs R13 (PKG-11); config byte-identical except draft_model removed. Owned TabbyAPI deploy image: engine (digest-pinned) + baked config.yml + blocking revision-pinned preflight.
FROM ghcr.io/theroyallab/tabbyapi@sha256:1c13dd17416660856e9962aeee8498827f0acd604eb36795128a03ddfe03c6ef
LABEL org.opencontainers.image.source="https://github.com/0xSero/local-ai-images"
LABEL ai.omarchy.row="R25"
LABEL ai.omarchy.model.repo="turboderp/Qwen3.8-27B-exl3"
LABEL ai.omarchy.model.revision="4acd9ad5af224ec9e8815a54d71a033f378a7e9d"
LABEL ai.omarchy.model.local_name="Qwen3.8-27B-EXL3-SC4bpw-H5"
LABEL ai.omarchy.model.mount="/opt/models/Qwen3.8-27B-EXL3-SC4bpw-H5"
LABEL ai.omarchy.model.quant="exl3-4.00bpw"
LABEL ai.omarchy.hardware="rtx-4090-24gb"
LABEL ai.omarchy.target_context_tokens="131072"
LABEL ai.omarchy.engine_base_cuda_floor="13.2"
LABEL ai.omarchy.mtp="off"
LABEL ai.omarchy.pinned_startup="model_name Qwen3.8-27B-EXL3-SC4bpw-H5 max_seq_len 131072 cache_size 131584 chunk_size 2048 draft_mode none"
COPY config.yml /app/config.yml
COPY deploy-preflight.sh /opt/deploy-preflight.sh
RUN chmod 0755 /opt/deploy-preflight.sh
ENTRYPOINT ["/opt/deploy-preflight.sh"]
