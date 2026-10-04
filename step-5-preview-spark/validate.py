"""Image checks for step-5-preview-spark (no weights, no GPU needed for --static)."""
import importlib.metadata as m, sys


def main():
    import step5_spark
    step5_spark.register()
    from vllm import ModelRegistry
    from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
    from transformers import AutoConfig
    archs = ModelRegistry.get_supported_archs()
    for a in ("Step5ForConditionalGeneration", "Step5ForCausalLM", "Step5MTP"):
        assert a in archs, a
    assert "step5_exl3" in QUANTIZATION_METHODS
    for t in ("step5", "step5_text", "step5_vision"):
        AutoConfig.for_model(t)
    eps = [e.name for e in m.entry_points(group="vllm.general_plugins")]
    assert "step5_spark" in eps, eps
    print("ok step5_spark", m.version("step5-spark"), "vllm", m.version("vllm"), "exllamav3", m.version("exllamav3"))
    if "--static" not in sys.argv:
        import torch, exllamav3_ext  # noqa: F401
        assert torch.cuda.is_available(), "GPU check requested but no CUDA device"
        print("cuda", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))


if __name__ == "__main__":
    main()
