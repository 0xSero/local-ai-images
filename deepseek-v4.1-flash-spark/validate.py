"""In-image validation for ghcr.io/0xsero/deepseek-v4.1-flash-spark (no model server, no weights).
  docker run --rm --gpus all --ipc host --ulimit memlock=-1 --entrypoint python3 IMG /opt/st/validate.py
Each check prints one line; exit code is the number of failed checks."""
import os
import re
import subprocess
import sys
import time
import traceback

FAILS = []


def check(name):
    def wrap(fn):
        t0 = time.time()
        try:
            out = fn()
            print(f"[ok]   {name}: {out} ({time.time() - t0:.1f}s)", flush=True)
        except Exception as e:  # noqa: BLE001
            FAILS.append(name)
            print(f"[FAIL] {name}: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
        return fn
    return wrap


@check("torch/gpu")
def _():
    import torch
    p = torch.cuda.get_device_properties(0)
    return f"torch {torch.__version__} cuda {torch.version.cuda} {p.name} sm_{p.major}{p.minor} " \
           f"arch_list={torch.cuda.get_arch_list()} TORCH_CUDA_ARCH_LIST={os.environ.get('TORCH_CUDA_ARCH_LIST')}"


@check("vllm version")
def _():
    import vllm
    v = vllm.__version__
    assert "r38" in v, v
    return f"{v} at {os.path.dirname(vllm.__file__)}"


@check("vllm compiled ops sm_121")
def _():
    import glob
    import importlib
    import vllm
    d = os.path.dirname(vllm.__file__)
    sos = sorted(glob.glob(f"{d}/*.so") + glob.glob(f"{d}/vllm_flash_attn/*.so"))
    per = {}
    for so in sos:
        r = subprocess.run(["cuobjdump", "--list-elf", so], capture_output=True, text=True)
        p = subprocess.run(["cuobjdump", "--list-ptx", so], capture_output=True, text=True)
        per[os.path.basename(so).split(".")[0]] = sorted(set(re.findall(r"\.(sm_\d+[af]?)\.", r.stdout))) + \
            [f"ptx:{a}" for a in sorted(set(re.findall(r"\.(sm_\d+[af]?)\.ptx", p.stdout)))]
    importlib.import_module("vllm._C_stable_libtorch")
    importlib.import_module("vllm._moe_C_stable_libtorch")
    # sm_121a SASS for the vLLM/MoE op libraries; the sm_80/89/90 entries come from kernels whose lists carry
    # "+PTX" or a fixed older arch (Marlin etc.). sm_120a would be a mis-targeted build (does not load on GB10).
    assert "sm_121a" in per["_C_stable_libtorch"] and "sm_121a" in per["_moe_C_stable_libtorch"], per
    assert not any("sm_120a" in v for v in per.values()), per
    return per


@check("EngramConfig table_memory=disk")
def _():
    from vllm.config import EngramConfig
    c = EngramConfig(cpu_offload=False, table_memory="disk", disk_resident_scales=False,
                     disk_prefetch_max_tokens=0)
    return f"table_memory={c.table_memory} disk_resident_scales={c.disk_resident_scales}"


@check("patch anchors applied")
def _():
    import vllm
    d = os.path.join(os.path.dirname(vllm.__file__), "models", "deepseek_v4")
    q = open(os.path.join(d, "quant_config.py")).read()
    m = open(os.path.join(d, "nvidia", "model.py")).read()
    assert "_st.Exl3RoutedMoEMethod" in q and "_st.skip_checkpoint_tensor(name)" in m
    from vllm.models.deepseek_v4 import st_exl3_moe as st
    assert st.backbone_layer("model.layers.3.ffn.experts") is None  # no ST_EXL3_PLAN -> native path
    return f"quant_config + nvidia/model.py patched, st_exl3_moe at {st.__file__}"


@check("exllamav3_ext")
def _():
    import exllamav3
    import exllamav3_ext
    from exllamav3.modules.quant.exl3 import LinearEXL3  # noqa: F401
    n = exllamav3_ext.exl3_moe_max_concurrency(0)
    return f"exllamav3 {getattr(exllamav3, '__version__', '?')} ext {exllamav3_ext.__file__} " \
           f"exl3_moe_max_concurrency(0)={n}"


@check("st_exl3_prefill ST_EXL3_PREFILL=st + st_moe_ext")
def _():
    os.environ["ST_EXL3_PREFILL"] = "st"
    import importlib
    from vllm.models.deepseek_v4 import st_exl3_prefill as sp
    sp = importlib.reload(sp)
    assert sp.PREFILL == "st", sp.PREFILL
    stx = sp._stx()
    fns = [n for n in dir(stx) if not n.startswith("_")]
    assert fns, "st_moe_ext exports nothing"
    return f"PREFILL={sp.PREFILL} st_moe_ext={stx.__file__} fns={fns}"


@check("b12x CuTe-DSL JIT (bf16_gemv) on GPU")
def _():
    import torch
    import b12x
    from b12x.gemm import bf16_gemv
    torch.manual_seed(0)
    res = []
    for m, n, k in [(4, 64, 5120), (300, 384, 5120)]:
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.125
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.125
        t0 = time.time()
        y = bf16_gemv.mm(x, w)
        torch.cuda.synchronize()
        err = ((y.float() - x.float() @ w.float().t()).abs().max().item())
        assert err < 5e-2, err
        res.append(f"[{m}x{k}]@[{n}x{k}] max_err={err:.2e} first_call={time.time() - t0:.1f}s")
    return f"b12x {getattr(b12x, '__version__', '')} CUTE_DSL_ARCH={os.environ.get('CUTE_DSL_ARCH')} " + "; ".join(res)


@check("b12x C loader + liburing")
def _():
    from b12x.loader import _native
    mod = _native.load()
    from b12x.sequence._shared import disk_table  # noqa: F401  (Engram disk staging)
    r = subprocess.run(["pkg-config", "--modversion", "liburing"], capture_output=True, text=True)
    return f"{mod.__file__} liburing {r.stdout.strip()}"


@check("b12x vllm plugins (entry points)")
def _():
    from importlib.metadata import entry_points
    eps = sorted(e.name for e in entry_points(group="vllm.general_plugins"))
    assert "b12x_loader" in eps, eps
    return eps


@check("vllm serve --help flags")
def _():
    r = subprocess.run(["vllm", "serve", "--help=all"], capture_output=True, text=True, timeout=600)
    txt = r.stdout + r.stderr
    need = ["--nnodes", "--node-rank", "--master-addr", "--engram-config", "--speculative-config",
            "--attention-backend", "--moe-backend", "--kv-cache-dtype", "--swa-block-size"]
    have = {f: f in txt for f in need}
    assert all(have[f] for f in need[:5]), have
    return have


sys.exit(len(FAILS))
