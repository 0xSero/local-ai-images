# Step-5-Preview-Spark serving image for NVIDIA DGX Spark (GB10, sm_121, arm64): StepFun Step-5-Preview with EXL3
# (turboderp exllamav3, MUL1) routed experts and dense body, served by vLLM with tensor parallel across 4 Sparks.
#
# Base: the CI-built, SLSA-attested DeepSeek-V4.1-Flash-Spark runtime from this repository (local-inference-lab vLLM
# r38 for sm_121, b12x incl. the RoCE one-shot all-reduce, turboderp exllamav3 1.5.3, the MIT st_moe_ext EXL3
# prefill GEMM). This layer only adds the pure-Python step5_spark vLLM plugin (entry point vllm.general_plugins):
# Step-5 text model (vLLM step3p5 + partial rotary + SSMax query scale + fp32 residual), Step-3.7-style vision
# wrapper with video-as-frames, Step-5 MTP drafter, and the step5_exl3 quantization method.
# Weights are NOT in the image. Entrypoint stays `vllm serve`.
FROM ghcr.io/0xsero/deepseek-v4.1-flash-spark@sha256:3cbc8ec016f5fbfc82eba3480de12399cbce31e7b76aa3108b8fe8246c42a79d
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]
COPY plugin /opt/step5/plugin
COPY validate.py /opt/step5/validate.py
RUN python3 -m pip install --no-deps --no-build-isolation /opt/step5/plugin && rm -rf /root/.cache/pip \
 && python3 /opt/step5/validate.py --static
# Step-5 dims for the EXL3 MoE kernels; EXL3 GEMV mode 1 (int8 activations + fp16 residual: ~fp16 accuracy at ~mode-2
# speed; mode 2 alone loses accuracy at 1-2 rows, mode 0 is exact but ~1.6x slower);
# the shared expert must not overlap the persistent exl3_moe kernel on a side stream (spin deadlock).
ENV ST_EXL3_HID=4096 ST_EXL3_INTER=1536 ST_EXL3_ACT_LIMIT=7.0 ST_EXL3_PREFILL=st ST_EXL3_PREFILL_MIN_ROWS=1 \
    EXL3_INT8_GEMV=1 VLLM_DISABLE_SHARED_EXPERTS_STREAM=1
