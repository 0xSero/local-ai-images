# Thin overlay on the built image: re-apply the (idempotent) patch with the DeepSeek-V4.1 hooks (quant config +
# text weight loop in vllm/models/deepseek_v4_1). Used when the full build context is unavailable.
ARG BASE=sovereign-trellis/ds41-exl3-spark:dev
FROM ${BASE}
COPY patch_vllm.py st_exl3_moe.py st_exl3_prefill.py /opt/st/patch/
# prebuilt st_moe_ext (same v0.30.0 base: torch 2.13+cu130, py3.12, sm_121) when the kernel signatures change
COPY st_moe_ext*.so /opt/st/patch/st_moe_ext/
RUN VD=$(python3 -c "import importlib.util,os;print(os.path.dirname(importlib.util.find_spec('vllm').origin))") \
 && python3 /opt/st/patch/patch_vllm.py "$VD" \
 && grep -q st_exl3 "$VD/models/deepseek_v4_1/quant_config.py" && grep -q skip_checkpoint_tensor "$VD/models/deepseek_v4_1/nvidia/model.py"
