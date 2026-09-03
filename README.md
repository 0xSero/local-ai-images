# local-ai-images

Container images the [local-ai registry](https://github.com/0xSero/local-ai-registry)
pins when an upstream image cannot run a recipe as published. One directory per
image, one Dockerfile each, built only by the `release-image` workflow so every
published digest carries BuildKit provenance, an SBOM, and a GitHub build
attestation:

    gh attestation verify oci://ghcr.io/0xsero/<image>@sha256:<digest> -o 0xSero

| Image | Base | Why it exists |
|---|---|---|
| `gateway` | `python:3.12-slim` by digest | the plugin's one endpoint: OpenAI chat passes through, Anthropic Messages and OpenAI Responses are translated to the engine's chat completions, streaming and tool calls included; enforces the share key when one is set. `python3 gateway/test.py` runs its tests against a fake engine. |
| `tabbyapi-exl3` | `ghcr.io/theroyallab/tabbyapi:cu13` by digest | adds `python3-dev` and `build-essential`; without Python headers Triton cannot JIT ExLlamaV3's gated-delta-net kernels, so Qwen3.5 and Qwen3.8 fail to load |

Recipes reference the digest the workflow prints, never a tag. To publish:
run the workflow with the image directory and a tag, take the digest from the
run summary, and pin it in the registry recipe together with a link to the run.
