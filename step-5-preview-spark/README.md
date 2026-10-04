# step-5-preview-spark

`ghcr.io/0xsero/step-5-preview-spark`: StepFun **Step-5-Preview** (600B MoE, 27B active, vision + video input) with
EXL3 (turboderp exllamav3, MUL1 codebook) routed experts and dense body, served by vLLM with tensor parallel 4 across
**four NVIDIA DGX Spark** (GB10, `sm_121`, arm64, 128 GB unified memory each) on their RoCE fabric. OpenAI-compatible
API on :8000, `step3p5` reasoning and tool-call parsers, image and video input, 262,144-token context, MTP
speculative decoding. Weights are not in the image: [0xSero/Step-5-Preview-Spark](https://huggingface.co/0xSero/Step-5-Preview-Spark).
Launch scripts and Pi integration: [0xSero/Step-5-Preview-Four-Sparks](https://github.com/0xSero/Step-5-Preview-Four-Sparks).

**linux/arm64 only**, for DGX Spark.

## Build

One layer on top of the CI-built, SLSA-attested DeepSeek-V4.1-Flash-Spark runtime from this repository (same vLLM,
b12x, exllamav3 and `st_moe_ext` binaries, see [`../deepseek-v4.1-flash-spark`](../deepseek-v4.1-flash-spark)):

| component | pin | licence |
|---|---|---|
| base `ghcr.io/0xsero/deepseek-v4.1-flash-spark` (tag `s016-ci`) | `sha256:3cbc8ec016f5fbfc82eba3480de12399cbce31e7b76aa3108b8fe8246c42a79d` | see base README (Apache-2.0 vLLM/b12x, MIT exllamav3) |
| `plugin/` — `step5_spark` vLLM general plugin | this commit | Apache-2.0 (model files derived from vLLM `step3p5.py`, `step3p5_mtp.py`, `step3p7.py`) / MIT (EXL3 integration) |

`step5_spark` registers (entry point `vllm.general_plugins`):
- configs `step5` / `step5_text` / `step5_vision` (no trust-remote-code);
- `Step5ForCausalLM`: vLLM Step-3.5 model + partial rotary 1/3 on full-attention layers + SSMax query scaling
  (`q *= log(pos+1) * ssmax_s[head]`) + fp32 residual stream; the checkpoint's sparse-indexer weights are not used
  (attention is dense);
- `Step5ForConditionalGeneration`: Step-3.7-style perception-encoder VL wrapper, video input as sampled frames through
  the image path (171 tokens per frame);
- `Step5MTP`: the three Step-5 MTP predictors for vLLM speculative decoding (`method: mtp`, V1 model runner);
- quantization method `step5_exl3`: EXL3 routed experts (expert-parallel inside the TP group, `exl3_moe` decode
  kernel, `st_moe_ext` prefill GEMM) and EXL3 body linears (lossless 128-aligned TP slicing, fused q/k/v and gate/up),
  optional BF16 body for prefill-sized batches (`body_format: hybrid`).

Environment baked in: `ST_EXL3_HID=4096 ST_EXL3_INTER=1536 ST_EXL3_ACT_LIMIT=7.0 ST_EXL3_PREFILL=st
ST_EXL3_PREFILL_MIN_ROWS=1 EXL3_INT8_GEMV=1 VLLM_DISABLE_SHARED_EXPERTS_STREAM=1`. The last one is required: the
persistent `exl3_moe` kernel can deadlock if the shared expert runs on a side stream at the same time.

`validate.py --static` (no GPU) runs in the build; `validate.py` on a Spark also checks CUDA/GB10.

Release (GitHub-hosted arm64 runner, signed provenance):

    gh workflow run release-image.yml -f image=step-5-preview-spark -f tag=<tag> -f platform=linux/arm64

## Runtime qualification

Measured on 4x DGX Spark, TP4 (see the model card for method): numbers are filled in at release from the final
configuration on the published digest.
