# Step-5 Spark runtime foundation

Experimental Linux ARM64 dependency image for two DGX Sparks. This is not a validated Step-5 server or a downloadable checkpoint. Full-model fidelity, memory fit, vision and end-to-end serving speed are still acceptance gates.

The image contains public upstream code and Python dependencies only. No checkpoint, tokenizer, calibration examples, evaluation prompts, images, responses, traces, credentials, host inventory or private runtime code enters the build context. `.dockerignore` permits exactly the Dockerfile, dependency list and launcher.

`--check` reports installed package versions without loading a model. `--serve` requires an independently supplied read-only `/runtime` bundle with `release.json`: an `entrypoint` filename and a `files` map of relative filenames to SHA-256 values. Every listed file is verified before execution. This is integrity checking, not a sandbox for untrusted Python. Mount only code you trust, and supply model storage separately. A serving bundle is not included in this experimental image.

The public ExLlamaV3 revision is pinned. Python direct model dependencies match the research environment; ancillary packages and OS packages are not a complete reproducibility lock. CUDA extensions JIT-compile on the target GPU and need a writable cache. No speed or graph acceptance is inferred from a successful image build or dependency check.

Build through `release-image` with `image=step5-spark-runtime`, `platform=linux/arm64`, and a development tag. Pin the resulting attested digest only after target-device container validation. Default launches do not open a network port or download any model data. Preserve graphs and the declared source termination markers in the separately admitted server; do not add output limits to benchmark requests.
