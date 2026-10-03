# deepseek-v4.1-flash-spark

`ghcr.io/0xsero/deepseek-v4.1-flash-spark`: DeepSeek-V4.1-Flash-Spark, which is DeepSeek-V4.1-Flash with EXL3
trellis routed experts, served by vLLM with tensor parallel 2 across **two NVIDIA DGX Spark** (GB10, `sm_121`,
arm64, 128 GB unified memory each) linked by their RoCE fabric. It exposes an OpenAI-compatible API on :8000 with
deepseek_v41 reasoning and tool-call parsing, image input (1 image per prompt), and a 262,144-token context.

The image is **linux/arm64 only**. Its architecture-specific vLLM and EXL3 kernels target GB10; FlashAttention 2 retains its upstream `8.0+PTX` fallback. The image is intended for DGX Spark.

## Published image

| | |
|---|---|
| image | `ghcr.io/0xsero/deepseek-v4.1-flash-spark@sha256:3cbc8ec016f5fbfc82eba3480de12399cbce31e7b76aa3108b8fe8246c42a79d` (tag `s016-ci`) |
| visibility | **public**; anonymous OCI manifest retrieval and its SHA256 verified |
| built by | GitHub-hosted ARM64 [release-image run 37138684406](https://github.com/0xSero/local-ai-images/actions/runs/37138684406), source `93e001cf4ecafeb71c6e737eeefad9a9b4ce9235` on `main` |
| attestation | Signed SLSA provenance verified against `release-image.yml@refs/heads/main`, with self-hosted runners denied |
| runtime qualification | Fresh six-gate acceptance on this exact image and final public weight layout is still required in [registry PR #154](https://github.com/0xSero/local-ai-registry/pull/154) |

The earlier host-built image was `sha256:5668e35e5ee021da4b9ce88a1964caf0e082e8ca01d81d1d64b6213a55c6add5`
(`s016` / historical `latest`), produced by `spark/host-build.sh` and `spark/overlay.Dockerfile` without a CI
attestation. Its measurements below remain historical evidence; they have not been relabeled for the new digest.

## Stack

| component | pin | licence |
|---|---|---|
| base `vllm/vllm-openai:v0.30.0` (arm64 from the multi-arch index) | `sha256:8a69ffad…4b90` | Apache-2.0 (vLLM); NVIDIA CUDA/NCCL runtime under NVIDIA's licences |
| vLLM: [local-inference-lab/vllm](https://github.com/local-inference-lab/vllm) r38 | `66c293578412417476f842c1da5805d3a3d959a8` | Apache-2.0 |
| [local-inference-lab/b12x](https://github.com/local-inference-lab/b12x) (B12X attention / linear / MoE, RoCE all-reduce) | `ce419b52681b7922bb0972d4b58b590a3fd005b2` | Apache-2.0 |
| [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) v1.5.3 | `d3739fd393337b1ff4d6c2a342b12f0c87a9592f` | MIT (turboderp) |
| `patch/` (this directory: `st_exl3_moe`, `st_exl3_prefill`, `st_moe_ext`, `patch_vllm.py`) | this commit | MIT |
| model: DeepSeek-V4.1-Flash (weights are not in the image) | `deepseek-ai/DeepSeek-V4.1-Flash@fb2764a5cf321eaa5070ca8f9e892818f477c16d` | MIT (DeepSeek) |

Licence note: other DGX Spark images, the local-inference-lab "jovian" images among them, bundle the
brandonmmusic fork of exllamav3. This image does **not** contain that fork or any of its code. It builds upstream
turboderp exllamav3 1.5.3 from source, and `st_moe_ext` compiles only against that release's MIT headers
(`quant/exl3_dq.cuh`, `quant/hadamard_inner.cuh`, `ptx.cuh`).

What the current source builds (the published image is pinned separately above):

- vLLM r38 is rebuilt with `TORCH_CUDA_ARCH_LIST=12.1a`. The build makes one CMake source
  edit: r38's CUDA>=13.0 supported-arch list omits `12.1`, which narrows `12.1a` kernels to `sm_120a` cubins that do
  not load on GB10. `build_wheels.sh` adds `12.1`, and the edit is idempotent.
- r38 unconditionally requests the Hopper-only FA3 extension even when `FA3_ARCHS` is empty. Its runtime only
  supports FA3 on compute capability 9.x, so `build_wheels.sh` omits that extension from this GB10 build. FA2 and
  all GB10-compatible vLLM/EXL3 targets remain. The source edit checks its exact anchor and is idempotent. This
  optimization landed after the published `93e001c` build, which still contains the unused FA3 extension.
- The base image's vLLM 0.30.0 is uninstalled completely, so its `.so` files never mix with r38's Python.
- CuTe-DSL 4.6.2 and quack 0.6.4 are installed `--no-deps` to keep the base NCCL 2.30.7.
- `patch_vllm.py` wires the EXL3 routed-expert method into `deepseek_v4` / `deepseek_v4_1` (quant config plus the
  text weight loop). Each edit asserts that its anchor appears exactly once, so a different vLLM revision fails the
  build instead of serving the wrong model.
- `st_moe_ext` is a prefill grouped GEMM over EXL3 experts that decodes once per 128/256 rows. Selected with
  `ST_EXL3_PREFILL=st`, it gives 1.3-1.65x on the routed MoE.

## Weights and mounts

The image ships no weights. The final public checkpoint is
`0xSero/DeepSeek-V4.1-Flash-Spark@08ac8b3defc9a239ba0baf51059687c689406519` (296 files, about 462 GB).
The pending registry launch mounts its repository root at `/model`, uses `--workdir /model/exl3`, and sets
`ST_EXL3_PLAN=/model/exl3/plan/J268-ho-v31-K5n.json`. Run the pinned repository's `reassemble.sh` on each node
first to reconstruct and hash-check the two Engram shards. Allow about 665 GB during download/reassembly,
plus image and cache space. This final layout still needs acceptance with the CI image.

The historical S016 launcher `spark/run_2spark.sh` uses separate local directories instead, read-only except the cache:

| container path | content |
|---|---|
| `/model` | native base `deepseek-ai/DeepSeek-V4.1-Flash@fb2764a5…` (config, tokenizer, attention / shared / Engram tensors) |
| `/banks` | EXL3 routed-expert banks (`qn`, `q31`, `q31s`) from the HF repo `0xSero/DeepSeek-V4.1-Flash-Spark` |
| `/plans` | the per-layer expert plan JSON (`ST_EXL3_PLAN`) from the same HF repo |
| `/cache` | b12x / Triton / RoCE JIT caches (read-write) |

## Serving config (campaign run S016)

`spark/run_2spark.sh` runs on the head node. It starts rank 1 headless on the worker over ssh, then starts rank 0
with the API. The flags are:

- `--tensor-parallel-size 2 --nnodes 2` with `--master-addr` set to the head's fabric IP, and `VLLM_HOST_IP` pinned
  to each node's fabric IP.
- `--kv-cache-dtype fp8 --block-size 256 --swa-block-size 128 --kv-cache-memory-bytes 2700000000`. This gives
  2,023,717 KV tokens.
- `--max-model-len 262144 --max-num-seqs 2 --max-num-batched-tokens 2048`.
- DSpark speculative decoding, k=7 adaptive.
- Engram `table_memory=disk`.
- `--attention-backend B12X --linear-backend b12x --moe-backend b12x`.
- `FULL_DECODE_ONLY` CUDA graphs and `VLLM_USE_BREAKABLE_CUDAGRAPH=0`.
- RoCE all-reduce (`VLLM_ENABLE_ROCE_ALLREDUCE=1`).
- `ST_EXL3_PREFILL=st` and `MALLOC_ARENA_MAX=2`.
- `--reasoning-parser deepseek_v41 --tool-call-parser deepseek_v41`.

The containers need `--network host --ipc host --privileged` and RDMA devices, because NCCL and B12X RoCE run over
the ConnectX fabric. A host `spark/memguard.sh` stops the model containers before a unified-memory OOM can wedge
the host. The launcher never caps output length and never touches GPU power or clocks.

Measured on 2x DGX Spark with the earlier host image `5668e35e` (sovereign-trellis `campaigns/dsv41-2spark/serve2spark/LEDGER.md` @
`a9ea365`, captured 2026-10-03):

- **S016**: prefill 8k **2,050.1** tok/s, 32k **2,060.6** tok/s; code decode C1 39.26 tok/s.
- **S012**: same stack minus the no-host-sync prefill route. Code C1 40.38 / 40.58, prose C1 29.00, code C2
  aggregate 60.15 tok/s.

The registry recipe's acceptance run is the authoritative evidence.

## Build

**CI publication.** Run `release-image.yml` with `image=deepseek-v4.1-flash-spark`, a new tag and
`platform=linux/arm64`. It builds natively on `ubuntu-24.04-arm` with a 6 h timeout. `./Dockerfile` is self-contained:

- A `wheels` stage fetches the three pinned sources as GitHub archives with `--checksum=sha256:`.
- That stage runs `build_wheels.sh` CPU-only with `MAX_JOBS=4` (build-arg); the workflow adds 24 GB swap for compiler memory peaks.
- The final stage installs the results.

The published main build completed vLLM in 6,607 s, exllamav3 in 672 s and the prefill extension in 67 s, passed
the patch/import checks, and published the image and provenance. These CPU build checks do not replace GPU
model acceptance. The registry requires the GitHub-hosted main-workflow attestation for publication.

**Historical host build.** `spark/host-build.sh` was used for the earlier unattested image. It:

1. fetches the pinned sources;
2. builds the wheels in the base container (`MAX_JOBS=10`, memory-capped);
3. runs `docker build` with `spark/Dockerfile`;
4. runs `validate.py` on the GPU.

Its build context is `spark/Dockerfile`, `build_wheels.sh`, `validate.py` and `patch/`. `spark/overlay.Dockerfile`
then re-applies the patch and adds the prebuilt `st_moe_ext` `.so` (built from the `st_moe_ext.cu` in `patch/`),
producing `v41fix3`. In that historical image, the `.cu` copy at
`/opt/st/patch/st_moe_ext/` is an older revision, and only the `.so` is used. The CI image ships its pinned source.

Validate on a Spark (no weights needed):

    docker run --rm --gpus all --ipc host --ulimit memlock=-1 --entrypoint python3 IMAGE /opt/st/validate.py
