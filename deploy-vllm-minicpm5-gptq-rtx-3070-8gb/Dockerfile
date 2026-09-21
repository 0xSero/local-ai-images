# R03 ghcr.io/0xsero/deploy-vllm-minicpm5-gptq-rtx-3070-8gb:v1
# Owned vLLM deploy image: engine (digest-pinned) + blocking revision-pinned model preflight.
# ENTRYPOINT fetches openbmb/MiniCPM5-2B-GPTQ@48d57c14fba6 into /opt/models/MiniCPM5-2B-GPTQ, then delegates to base shim.
# Served WITHOUT tool-calling per DELIVERY-24 R03/R06 contingency; vLLM MiniCPM5 tool parser unverified. capabilities.tools=engine-unsupported.
# Build-status only; runtime untested here.
FROM ghcr.io/0xsero/vllm-openai@sha256:1de88ff218282ce71da7811f62ea527556648e1ad05dd36e5fc82d0f9ff07424
LABEL org.opencontainers.image.source="https://github.com/0xSero/local-ai-images"
LABEL ai.omarchy.row="R03"
LABEL ai.omarchy.model.repo="openbmb/MiniCPM5-2B-GPTQ"
LABEL ai.omarchy.model.revision="48d57c14fba6b1f86109e1a54fafb0848c163f04"
LABEL ai.omarchy.model.local_name="MiniCPM5-2B-GPTQ"
LABEL ai.omarchy.model.mount="/opt/models/MiniCPM5-2B-GPTQ"
LABEL ai.omarchy.model.quant="awq-4bit"
LABEL ai.omarchy.hardware="rtx-3070-8gb"
LABEL ai.omarchy.target_context_tokens="32768"
LABEL ai.omarchy.engine_base_cuda_floor="13.0"
LABEL ai.omarchy.pinned_startup.note="actual baked CMD; DELIVERY --enable-auto-tool-choice stripped (no vLLM MiniCPM5 tool parser)"
LABEL ai.omarchy.pinned_startup="vllm serve /opt/models/MiniCPM5-2B-GPTQ --served-model-name MiniCPM5-2B-GPTQ --host 0.0.0.0 --port 8000 --max-model-len 32768 --gpu-memory-utilization 0.90 --trust-remote-code"
LABEL ai.omarchy.capabilities.tools="engine-unsupported"
COPY deploy-preflight.sh /opt/deploy-preflight.sh
COPY omp-acquire.sh /opt/omp-acquire
RUN chmod 0755 /opt/deploy-preflight.sh /opt/omp-acquire
ENTRYPOINT ["/opt/deploy-preflight.sh"]
CMD ["vllm", "serve", "/opt/models/MiniCPM5-2B-GPTQ", "--served-model-name", "MiniCPM5-2B-GPTQ", "--host", "0.0.0.0", "--port", "8000", "--max-model-len", "32768", "--gpu-memory-utilization", "0.90", "--trust-remote-code"]
