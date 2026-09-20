# R02 ghcr.io/0xsero/deploy-sglang-minicpm5-gptq-rtx-3070-8gb:v1
# Owned SGLang deploy image: engine (digest-pinned) + blocking revision-pinned model preflight.
# ENTRYPOINT fetches openbmb/MiniCPM5-2B-GPTQ@48d57c14fba6 into /opt/models/MiniCPM5-2B-GPTQ, writes .omp-ready, then delegates
# to the base /opt/entrypoint.sh shim ("$@" = passed --docker-args OR the serve CMD below). Build-status only.
FROM ghcr.io/0xsero/sglang@sha256:787461f889674cdd9339be2409a6e8b46ec2f5aac71669481c319f87c2beca13
LABEL org.opencontainers.image.source="https://github.com/0xSero/local-ai-images"
LABEL ai.omarchy.row="R02"
LABEL ai.omarchy.model.repo="openbmb/MiniCPM5-2B-GPTQ"
LABEL ai.omarchy.model.revision="48d57c14fba6b1f86109e1a54fafb0848c163f04"
LABEL ai.omarchy.model.local_name="MiniCPM5-2B-GPTQ"
LABEL ai.omarchy.model.mount="/opt/models/MiniCPM5-2B-GPTQ"
LABEL ai.omarchy.model.quant="awq-4bit"
LABEL ai.omarchy.hardware="rtx-3070-8gb"
LABEL ai.omarchy.target_context_tokens="32768"
LABEL ai.omarchy.engine_base_cuda_floor="13.0"
LABEL ai.omarchy.pinned_startup="sglang.launch_server --context-length 32768 --mem-fraction-static 0.85 --tool-call-parser minicpm5 --reasoning-parser qwen3"
COPY deploy-preflight.sh /opt/deploy-preflight.sh
COPY omp-acquire.sh /opt/omp-acquire
RUN chmod 0755 /opt/deploy-preflight.sh /opt/omp-acquire
ENTRYPOINT ["/opt/deploy-preflight.sh"]
CMD ["/opt/sglang/bin/python3", "-m", "sglang.launch_server", "--model-path", "/opt/models/MiniCPM5-2B-GPTQ", "--served-model-name", "MiniCPM5-2B-GPTQ", "--host", "0.0.0.0", "--port", "30000", "--context-length", "32768", "--mem-fraction-static", "0.85", "--tool-call-parser", "minicpm5", "--reasoning-parser", "qwen3"]
