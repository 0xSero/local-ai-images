"""Build-time patcher (Dockerfile): wires st_exl3_moe into the jovian vLLM source tree. Each edit asserts that its
anchor exists exactly once, so a different vLLM revision fails the build instead of serving the wrong thing."""
import shutil, sys
from pathlib import Path

V = Path(sys.argv[1] if len(sys.argv) > 1 else "/opt/glm53-flash/vllm/vllm")
D = V / "models" / "deepseek_v4"
shutil.copy(Path(__file__).with_name("st_exl3_moe.py"), D / "st_exl3_moe.py")
shutil.copy(Path(__file__).with_name("st_exl3_prefill.py"), D / "st_exl3_prefill.py")


def edit(path, anchor, new):
    s = path.read_text()
    if new in s:                                   # idempotent: already applied (overlay rebuilds re-run this)
        print("already", path.relative_to(V.parent))
        return
    assert s.count(anchor) == 1, f"{path}: anchor found {s.count(anchor)}x: {anchor[:80]!r}"
    path.write_text(s.replace(anchor, new))
    print("patched", path.relative_to(V.parent), "|", anchor.strip().splitlines()[0][:70])


# 1. backbone routed experts -> EXL3 method (only when ST_EXL3_PLAN is set; draft/mtp experts untouched)
edit(D / "quant_config.py",
     "        if isinstance(layer, RoutedExperts):\n",
     "        if isinstance(layer, RoutedExperts):\n"
     "            from vllm.models.deepseek_v4 import st_exl3_moe as _st\n"
     "            _st_layer = _st.backbone_layer(prefix)\n"
     "            if _st_layer is not None:\n"
     "                return _st.Exl3RoutedMoEMethod(layer.moe_config, _st_layer)\n")
# 2. the loader skips the native routed-expert tensors of backbone layers (their params do not exist)
edit(D / "nvidia" / "model.py",
     "        for name, loaded_weight in weights:\n            if pad_shared_expert and \".shared_experts.\" in name:\n",
     "        from vllm.models.deepseek_v4 import st_exl3_moe as _st\n"
     "        for name, loaded_weight in weights:\n"
     "            if _st.skip_checkpoint_tensor(name):\n"
     "                continue\n"
     "            if pad_shared_expert and \".shared_experts.\" in name:\n")
# 3./4. DeepSeek-V4.1 has its own quant config and text-model weight loop (deepseek_v4_1/); without these the backbone
# routed experts load as native MXFP4 (~134 GiB/rank on TP2) and GB10 runs out of memory. The DSpark draft layers are
# named layers.40-42 and stay native (backbone_layer only takes ids < 40).
D41 = V / "models" / "deepseek_v4_1"
if D41.exists():
    edit(D41 / "quant_config.py",
         "        if isinstance(layer, RoutedExperts):\n            return Mxfp4MoEMethod(layer.moe_config)\n",
         "        if isinstance(layer, RoutedExperts):\n"
         "            from vllm.models.deepseek_v4 import st_exl3_moe as _st\n"
         "            _st_layer = _st.backbone_layer(prefix)\n"
         "            if _st_layer is not None:\n"
         "                return _st.Exl3RoutedMoEMethod(layer.moe_config, _st_layer)\n"
         "            return Mxfp4MoEMethod(layer.moe_config)\n")
    edit(D41 / "nvidia" / "model.py",
         "        for name, loaded_weight in weights:\n            if name.startswith((\"vision.\", \"aligner.\", \"image_\")):\n",
         "        from vllm.models.deepseek_v4 import st_exl3_moe as _st\n"
         "        for name, loaded_weight in weights:\n"
         "            if _st.skip_checkpoint_tensor(name):\n"
         "                continue\n"
         "            if name.startswith((\"vision.\", \"aligner.\", \"image_\")):\n")
print("ok")
