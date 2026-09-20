# TabbyAPI source contract — bound to immutable image digests (parity-only)

These bind the ACTUAL installed TabbyAPI source (read from the pinned image via `docker create` +
`docker cp /app`, no weights, no GPU) to the immutable image digest. This is a SOURCE/tokenizer-parity
contract — it is NOT proof of server usage==sent, which needs a real /v1/completions request adapter
on the exact source at rental (BenchmarkRepair/Opus).

## cu12 — deploy-tabbyapi-qwen38-2bpw-rtx-3080-ti-12gb @ sha256:47c4c6eb402267cd8d70b2954111cc37214887afa7168c2759b6cd73f93f8623
- Source-bundle artifact: `tabby-source-deploy-tabbyapi-qwen38-2bpw-rtx-3080-ti-12gb` (244 KB, source only), run 35517774025 (SUCCESS), artifact id 10607062924.
  Fetch: `gh api repos/0xSero/local-ai-images/actions/artifacts/10607062924/zip > s.zip && unzip s.zip`
- Commit provenance (no guess): tabbyapi-0.0.1 dist-info direct_url.json = `{"url":"file:///app"}` — local install,
  NOT a git checkout, so NO recoverable commit_id. In-image source BYTES are authoritative; compare to 53da7919 by content.
  exllamav3 dist-info direct_url.json = release wheel `.../v1.5.0/exllamav3-1.5.0+cu128.torch2.9.0-cp312-cp312-linux_x86_64.whl` (provably v1.5.0, cu128, torch2.9.0).
- Contract-file sha256 (bound to @47c4c6eb):
  - common/errors.py            = a5f6db04b928a904edea5415dd6011ce5407541980aac2767b5679d5eab820ad
  - endpoints/OAI/types/completion.py = b8b672299e550a38ee25394e116acf43f22e4343e68150cd95e2445c427f59cb
  - common/sampling.py          = f39e831f40aee7a22cc0098d2d2e7ac74814045e535eedeef5dd828a247e8c2b
  - common/config_models.py     = 49eb7b9f6d5fd7aff83ba5b54d80b8b60b127bdd5ebca04dee5694d746e92034
  - common/tabby_config.py      = 98ed3b609bd48740c8acc24d20cefa9820f66aeef66847470e1fed4e4d206d43
  - backends/exllamav3/model.py = 2d9308a34852d6c5d6786b96ffacd6eda4dbfb5e3946cb4b6880b9ee72020176
  - backends/exllamav3/tokenizer.py = ABSENT (exllamav3 tokenizer lives in the exllamav3 wheel, not tabby backends)
- Contract fields (line-cited from the bundle):
  - CompletionRequest: endpoints/OAI/types/completion.py:57 `class CompletionRequest(CommonCompletionRequest)`
  - add_bos_token: default True (endpoints/core/types/token.py:13); chat-completions FORCE off
    (endpoints/OAI/types/chat_completion.py:167 `@field_validator("add_bos_token", mode="after")` "Always disable
    add_bos_token with chat completions") + double-BOS guard (chat_completion.py:529)
  - ignore_eos: ALIAS of ban_eos_token — common/sampling.py:249 `validation_alias=AliasChoices("ban_eos_token","ignore_eos")`
  - validate_context_requirements: common/errors.py:40
  - exllamav3 encode passes bos through: backends/exllamav3/model.py:916 `add_bos=unwrap(kwargs.get("add_bos_token"), self.hf_model.add_bos_token())`

## cu13 — deploy-tabbyapi-qwen38-4bpw-rtx-4090-24gb @ sha256:80f9e2befda50e4bb1c0ae6797a39f1cf285b54f2a519458095167bb0562bd65
- Source-bundle artifact `tabby-source-deploy-tabbyapi-qwen38-4bpw-rtx-4090-24gb` (244 KB), run 35518118820 (SUCCESS), artifact id 10607921308.
- tabby dist-info direct_url.json = `{"url":"file:///app"}` (local install, no git — same as cu12). exllamav3 = release wheel v1.5.0+cu132.torch2.11.0.
- cu13 commit LABELED 53da7919 by the 09-18 engine.json provenance (a provenance label, not a byte comparison).

## Contract-file byte scope (precise — SIX files only, NOT full repo / full commit identity)
The SIX contract-relevant files are byte-identical across cu12 (@47c4c6eb), cu13 (@80f9e2be), AND upstream
theroyallab/tabbyAPI @53da7919 — verified by sha256 of the actual installed bytes and of upstream content
fetched at that ref (all six MATCH):
  common/errors.py=a5f6db04, endpoints/OAI/types/completion.py=b8b67229, common/sampling.py=f39e831f,
  common/config_models.py=49eb7b9f, common/tabby_config.py=98ed3b60, backends/exllamav3/model.py=2d9308a3.
This scopes to the contract surface only. It does NOT establish that the cu12 (or cu13) full tree equals
upstream 53da7919, and it does NOT recover cu12's commit: cu12 tabby is version 0.0.1, dist-info
direct_url=file:///app, no VCS metadata => cu12 commit remains UNKNOWN. The contract is verified by the
actual installed bytes bound to the image digests + the upstream byte match on those six files — that is
sufficient; the commit stays truthfully unknown. exllamav3 differs by build only (cu12 v1.5.0+cu128.torch2.9.0
vs cu13 v1.5.0+cu132.torch2.11.0; same release v1.5.0).

## Long-context adapter tokenizer/template dep-check (BenchmarkRepair, in-image, fail-closed)
Probe engine=tabbytmpl: per EXL3 model dir, snapshot_download tokenizer/config ONLY (no weights), then
AutoTokenizer.from_pretrained(dir, local_files_only=True, trust_remote_code=False). Ran inside BOTH images.
- cu12 image (transformers 4.57.6, run 35521000062) and cu13 image (transformers 5.17.0, run 35521002667):
  ALL 7 distinct EXL3 dirs => template=PRESENT, chat_template_sha=c3cf9e34, vocab=248044,
  enable_thinking=True renders '...<|im_start|>assistant\n<think>\n'. ALL_TABBY_TEMPLATE_LOCAL_OK=True both runs.
  Dirs: turboderp/Qwen3.8-27B-exl3 @ {e5e1f4b3 2bpw, 004a8871 3bpw, 4acd9ad5 4bpw, 516bf129 4bpwv6,
  f33f26d9 5bpw, 60d005a2 6bpw} + turboderp/Qwen3.8-Flash-Next-exl3 @ 65c89531 2.05bpw.
=> (1) .chat_template loads local-only (no network) + trust_remote_code=False for every dir; (2) render is
  template-driven and identical across both pinned transformers versions (version-invariant). No tabby package
  fails the long-context gate on this dep. A future revision that drops the template exits nonzero (rc1 fail-closed).

## Token endpoints (/v1/token/encode + /v1/token/decode) present, not stripped
Installed endpoints/core/router.py: "/v1/token/encode" (line 400 -> encode_tokens -> TokenEncodeResponse),
"/v1/token/decode" (line 458 -> decode_tokens -> TokenDecodeResponse). router.py byte-identical across cu12
(@47c4c6eb) + cu13 (@80f9e2be): sha256 100427688dbf1508123d9753a6b0677ccffe8ae54a593eae87c910fce6eb7ac4.
No package strips them: every tabby deploy Dockerfile's only mutation over the engine base is
`COPY config.yml /app/config.yml` (config + labels), never a /app source rewrite -> endpoints inherited
unchanged in all tabby packages. Supports BenchmarkRepair's encode->truncate->decode bounded-length path.
