"""SGLang general plugin (`sglang.srt.plugins` entry point): registers the `exl3` quantization method and installs
the two model-side shims EXL3 checkpoints need. Runs in the launcher, the engine and every scheduler subprocess."""
from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)
_done = False


def activate() -> None:
    global _done
    if _done:
        return
    _done = True
    from sglang.srt.layers.quantization import QUANTIZATION_METHODS
    from sglang.srt.arg_groups.choices import QUANTIZATION_CHOICES, add_quantization_method_choices
    from .sglang_glue.config import Exl3Config
    QUANTIZATION_METHODS["exl3"] = Exl3Config
    if "exl3" not in QUANTIZATION_CHOICES:
        add_quantization_method_choices(["exl3"])
    _patch_mtp_fc()
    logger.info("sglang-exl3: registered quantization method 'exl3'")


def _patch_mtp_fc() -> None:
    """The Qwen3.5 MTP draft builds `self.fc = nn.Linear(2h, h)`; EXL3 checkpoints quantize `mtp.fc`. Replace it
    with an EXL3 dense module after construction and finalize it after loading (nn.Linear has no quant hook)."""
    try:
        from sglang.srt.models import qwen3_5_mtp as m
    except Exception as e:  # pragma: no cover
        logger.warning("sglang-exl3: MTP shim not installed (%s)", e)
        return
    cls = m.Qwen3_5ForCausalLMMTP
    if getattr(cls, "_exl3_patched", False):
        return
    orig_init, orig_load = cls.__init__, cls.load_weights

    def __init__(self, config, quant_config=None, prefix="", *a, **k):
        orig_init(self, config, quant_config, prefix, *a, **k)
        from .sglang_glue.config import Exl3Config
        from .sglang_glue.linear import Exl3Dense
        qc = getattr(self, "quant_config", quant_config)
        if isinstance(qc, Exl3Config) and os.environ.get("SGLANG_EXL3_DRAFT_SHARE_EMBED", "1") == "1":
            # The draft receives the target's embed_tokens via set_embed_and_head (init_lm_head); on a 24 GB card the
            # draft's own 2.5 GB bf16 copy would otherwise sit in memory when the KV pool is sized. Keep a 0-row
            # placeholder until the target's tensor is shared.
            emb = self.model.embed_tokens
            w = emb.weight
            emb.weight = torch.nn.Parameter(torch.empty((0, w.shape[1]), dtype=w.dtype, device=w.device), requires_grad=False)
            for name, val in vars(w).items():
                if name != "data" and not hasattr(emb.weight, name):
                    setattr(emb.weight, name, val)
            self._exl3_skip_embed = True
            del w
            torch.cuda.empty_cache()
        if isinstance(qc, Exl3Config):
            info = qc.lookup("mtp.fc")
            if info is not None:
                self.fc = Exl3Dense(self.fc.in_features, self.fc.out_features, info, "mtp.fc",
                                    params_dtype=self.fc.weight.dtype)
                logger.info("sglang-exl3: MTP fc replaced by an EXL3 linear (%s)", info[:4])

    def load_weights(self, weights, *a, **k):
        if getattr(self, "_exl3_skip_embed", False):
            weights = ((n, w) for n, w in weights if not n.endswith("embed_tokens.weight"))
        out = orig_load(self, weights, *a, **k)
        fc = getattr(self, "fc", None)
        if fc is not None and hasattr(fc, "process_weights_after_loading") and hasattr(fc, "exl3_shards"):
            fc.process_weights_after_loading()
        return out

    cls.__init__, cls.load_weights, cls._exl3_patched = __init__, load_weights, True
