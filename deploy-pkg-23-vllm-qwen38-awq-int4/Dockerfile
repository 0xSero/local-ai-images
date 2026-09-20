# R29 ghcr.io/0xsero/deploy-pkg-23-vllm-qwen38-awq-int4:v1
# landed recipe qwen38-awq-int4-rtx3090-vllm-tp1 config; 21.04GB on 24GB is tight (landed config). Owned vllm deploy image: engine (digest-pinned) + blocking revision-pinned preflight; serve local /opt/models path. Build-status only.
FROM ghcr.io/0xsero/vllm-openai@sha256:1de88ff218282ce71da7811f62ea527556648e1ad05dd36e5fc82d0f9ff07424
LABEL org.opencontainers.image.source="https://github.com/0xSero/local-ai-images"
LABEL ai.omarchy.row="R29"
LABEL ai.omarchy.model.repo="cyankiwi/Qwen3.8-27B-AWQ-INT4"
LABEL ai.omarchy.model.revision="63768c10df38c0395e12ef49edac1bd539eaeeea"
LABEL ai.omarchy.model.local_name="Qwen3.8-27B-AWQ-INT4"
LABEL ai.omarchy.model.mount="/opt/models/Qwen3.8-27B-AWQ-INT4"
LABEL ai.omarchy.model.quant="awq-int4"
LABEL ai.omarchy.hardware="rtx-3090-24gb"
LABEL ai.omarchy.target_context_tokens="40960"
LABEL ai.omarchy.engine_base_cuda_floor="13.0"
LABEL ai.omarchy.model.served_name="Qwen3.8-27B"
LABEL ai.omarchy.pinned_startup="vllm serve /opt/models/Qwen3.8-27B-AWQ-INT4 --served-model-name Qwen3.8-27B --host 0.0.0.0 --port 8000 --max-model-len 40960 --max-num-seqs 2 --kv-cache-dtype fp8 --tensor-parallel-size 1 --trust-remote-code"
COPY deploy-preflight.sh /opt/deploy-preflight.sh
RUN chmod 0755 /opt/deploy-preflight.sh
ENTRYPOINT ["/opt/deploy-preflight.sh"]
CMD ["vllm", "serve", "/opt/models/Qwen3.8-27B-AWQ-INT4", "--served-model-name", "Qwen3.8-27B", "--host", "0.0.0.0", "--port", "8000", "--max-model-len", "40960", "--max-num-seqs", "2", "--kv-cache-dtype", "fp8", "--tensor-parallel-size", "1", "--trust-remote-code"]
