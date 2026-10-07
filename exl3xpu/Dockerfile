# exl3xpu: EXL3 inference on Intel Arc (Battlemage) via vLLM XPU.
FROM intel/llm-scaler-vllm:0.26.0-b2@sha256:52218ad85513ab6686d4c090c83c2bd8c5b02423c63aa4dabd41837fe641fe3b
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/exl3xpu
ARG EXL3XPU_COMMIT=15ded2f3add148c4db3c900cba7de878238f53bc
RUN git init . && git remote add origin https://github.com/sybil-solutions/exl3xpu.git \
 && git fetch --depth 1 origin "$EXL3XPU_COMMIT" && git checkout --detach FETCH_HEAD \
 && test "$(git rev-parse HEAD)" = "$EXL3XPU_COMMIT"
LABEL org.opencontainers.image.source="https://github.com/sybil-solutions/local-ai-images"
RUN bash scripts/build_ext.sh && pip install --no-deps --no-build-isolation -e . \
 && python3 -c "from importlib.metadata import entry_points as e; assert any(x.name=='exl3xpu' for x in e(group='vllm.general_plugins'))"
ENV MODELS_DIR=/models
ENTRYPOINT ["python3", "scripts/serve.py"]
CMD ["models/qwen3.8-27b-exl3-4.00bpw", "--gpu", "0"]
