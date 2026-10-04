# SPDX-License-Identifier: Apache-2.0
"""vLLM general plugin for Step-5 on DGX Spark (entry point: vllm.general_plugins -> step5_spark:register).

Registers the step5 / step5_text / step5_vision configs, the Step5ForConditionalGeneration / Step5ForCausalLM / Step5MTP
architectures and the "step5_exl3" quantization method, and teaches the speculative config to build the Step-5 MTP
drafter (method "mtp").
"""
_DONE = False


def register():
    global _DONE
    if _DONE:
        return
    _DONE = True
    from transformers import AutoConfig
    from vllm import ModelRegistry
    from vllm.transformers_utils import config as vcfg
    from step5_spark.config import Step5Config, Step5TextConfig, Step5VisionConfig

    for cls in (Step5Config, Step5TextConfig, Step5VisionConfig):
        AutoConfig.register(cls.model_type, cls, exist_ok=True)
        vcfg._CONFIG_REGISTRY[cls.model_type] = cls
    ModelRegistry.register_model("Step5ForConditionalGeneration", "step5_spark.step5_vl:Step5ForConditionalGeneration")
    ModelRegistry.register_model("Step5ForCausalLM", "step5_spark.step5:Step5ForCausalLM")
    ModelRegistry.register_model("Step5MTP", "step5_spark.step5_mtp:Step5MTP")
    import step5_spark.exl3_quant  # noqa: F401  (registers quant_method "step5_exl3")
    _patch_speculative()
    _patch_chat_video_parts()


def _patch_chat_video_parts():
    """Step-5's chat template renders only image parts. For models that define STEP5_VIDEO_PLACEHOLDER, turn
    OpenAI-format video parts ({"type": "video"}) into a text part carrying that placeholder so the template keeps
    it in place (the video data itself is still tracked by vLLM's multimodal parser)."""
    try:
        from vllm.entrypoints import chat_utils as cu
    except Exception as e:  # pragma: no cover
        import logging
        logging.getLogger(__name__).warning("step5_spark: chat video patch skipped (%s)", e)
        return
    orig = cu._parse_chat_message_content_part
    if getattr(orig, "_step5", False):
        return

    def _parse_chat_message_content_part(part, mm_parser, *, wrap_dicts, interleave_strings):
        res = orig(part, mm_parser, wrap_dicts=wrap_dicts, interleave_strings=interleave_strings)
        if wrap_dicts and isinstance(res, dict) and res.get("type") == "video":
            try:
                ph = getattr(mm_parser._tracker.model_cls, "STEP5_VIDEO_PLACEHOLDER", None)
            except Exception:
                ph = None
            if ph:
                return {"type": "text", "text": ph}
        return res

    _parse_chat_message_content_part._step5 = True
    cu._parse_chat_message_content_part = _parse_chat_message_content_part


_ORIG_HF_CONFIG_OVERRIDE = None


def _step5_hf_config_override(hf_config):
    """Module level so it pickles (the speculative config travels to every worker): Step-5 MTP drafter config."""
    if getattr(hf_config, "model_type", None) in ("step5", "step5_text"):
        qc = getattr(hf_config, "quantization_config", None)
        hf_config = getattr(hf_config, "text_config", hf_config)
        if qc is not None and getattr(hf_config, "quantization_config", None) is None:
            hf_config.update({"quantization_config": qc})
        hf_config.model_type = "step3p5_mtp"
        hf_config.update({"n_predict": getattr(hf_config, "num_nextn_predict_layers", 1),
                          "architectures": ["Step5MTP"]})
        return hf_config
    return _ORIG_HF_CONFIG_OVERRIDE(hf_config)


def _patch_speculative():
    global _ORIG_HF_CONFIG_OVERRIDE
    from vllm.config import speculative as spec

    cur = spec.SpeculativeConfig.__dict__.get("hf_config_override")
    if getattr(cur, "__func__", cur) is _step5_hf_config_override:
        return
    _ORIG_HF_CONFIG_OVERRIDE = spec.SpeculativeConfig.hf_config_override
    spec.SpeculativeConfig.hf_config_override = staticmethod(_step5_hf_config_override)
