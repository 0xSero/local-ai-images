# R01 ghcr.io/0xsero/deploy-llamacpp-bonsai-ptq1-rtx-3070-8gb:v1
# Pinned Prism llama.cpp deploy layer over the campaign llamacpp-prism engine image (digest-pinned).
# GGUF + mmproj fetched from HF (prism-ml/Ternary-Bonsai-2-27B-gguf) at launch. mmproj included for the PTQ1_0+mmproj package.
# Build-status only; runtime untested.
FROM ghcr.io/0xsero/llamacpp-prism@sha256:d9efaf72a177156667879c795217f5e98a6041930ddb29778399edc1ef57b33c
LABEL org.opencontainers.image.source="https://github.com/0xSero/local-ai-images"
LABEL ai.omarchy.row="R01"
LABEL ai.omarchy.model.repo="prism-ml/Ternary-Bonsai-2-27B-gguf"
LABEL ai.omarchy.model.revision="6ed5e12bf84b7a63069882c91dd9e9218647d17b"
LABEL ai.omarchy.model.branch="main"
LABEL ai.omarchy.hardware="rtx-3070-8gb"
LABEL ai.omarchy.target_context_tokens="65536"
LABEL ai.omarchy.engine_base_cuda_floor="12.8"
LABEL ai.omarchy.pinned_startup="llama-server --ctx-size 65536 --cache-type-k q4_0 --cache-type-v q4_0 --n-gpu-layers 99 --jinja --mmproj"
CMD ["/opt/llama/llama-server","-hf","prism-ml/Ternary-Bonsai-2-27B-gguf","--hf-file","Ternary-Bonsai-2-27B-PTQ1_0.gguf","--ctx-size","65536","--cache-type-k","q4_0","--cache-type-v","q4_0","--n-gpu-layers","99","--jinja","--mmproj","Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf","--host","0.0.0.0","--port","8080"]
