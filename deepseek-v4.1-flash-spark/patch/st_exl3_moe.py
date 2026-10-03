# SPDX-License-Identifier: MIT
# sovereign-trellis: EXL3 (trellis, mul1) routed experts for DeepSeek-V4.1-Flash inside vLLM.
#
# The routed experts of the 40 backbone MoE layers come from per-expert EXL3 banks chosen by a plan
# (pipeline/dsq/planstate.py schema: layers / keep_native / expert_state, qdir = colon-separated bank roots).
# Everything else (attention, shared experts, router, Engram, DSpark draft experts, head) stays on the
# checkpoint's native path. Only the quant method of the backbone RoutedExperts is replaced:
#   * create_weights registers no tensors (the loader skips the native routed-expert tensors, see patch_vllm.py)
#   * process_weights_after_loading reads this TP rank's experts [rank*E/tp, (rank+1)*E/tp) from the banks
#   * apply() gets the router's top-k (ids, weights) and returns this rank's partial routed sum; the MoE runner's
#     TP all-reduce adds the ranks up (expert-parallel inside a TP group: a sum either way)
# Kernels: exllamav3 1.5.3 exllamav3_ext.exl3_moe for decode/CUDA-graph shapes (all rows fused, no host sync). Prefill
# (host-synced) goes through st_exl3_prefill.run_bucket, chosen by ST_EXL3_PREFILL:
#   off      exl3_moe for experts with <= ST_EXL3_FUSED_ROWS rows, reconstruct + hgemm above (the original route)
#   st       st_moe_ext grouped GEMM (decode once per 32-256 rows) for experts with >= ST_EXL3_PREFILL_MIN_ROWS rows
#   dq_fp16  batched reconstruct_had + torch._grouped_mm (measured slower than off on GB10; kept for comparison)
# see campaigns/dsv41-2spark/serve2spark/kern/RESULTS.md
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import FusedMoEMethodBase

logger = init_logger(__name__)

try:                                    # copied next to this file into vllm/models/deepseek_v4 by patch_vllm.py
    from . import st_exl3_prefill as _pf
except ImportError:                     # tests / PYTHONPATH=/opt/st/patch
    import st_exl3_prefill as _pf
ACT_LIMIT, FUSED_ROWS, HID, INTER = _pf.ACT_LIMIT, _pf.FUSED_ROWS, _pf.HID, _pf.INTER

N_EXPERTS = 384
_PLAN = None
_BUFS: dict[int, tuple] = {}
_PREFIX_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.ffn\.experts(?:$|\.)")


def plan():
    global _PLAN
    if _PLAN is None:
        p = os.environ.get("ST_EXL3_PLAN")
        _PLAN = json.load(open(p)) if p else None
    return _PLAN


def backbone_layer(prefix: str) -> int | None:
    """Layer id if prefix names a backbone (not mtp/draft) routed-experts module under an active plan."""
    if plan() is None or "mtp" in prefix:
        return None
    m = _PREFIX_RE.search(prefix)
    return int(m.group(1)) if m and int(m.group(1)) < 40 else None


_SKIP_RE = re.compile(r"^(?:model\.)?layers\.(\d+)\.ffn\.experts\.\d+\.")


def skip_checkpoint_tensor(name: str) -> bool:
    """True for native routed-expert tensors of backbone layers (replaced by EXL3 banks)."""
    if plan() is None:
        return False
    m = _SKIP_RE.match(name)
    return bool(m) and int(m.group(1)) < 40


def _ext():
    import exllamav3_ext
    return exllamav3_ext


def _states(L: int) -> list[str]:
    p = plan()
    base = str(p["layers"][str(L)])
    st = [base] * N_EXPERTS
    for e in p.get("keep_native", {}).get(str(L), []):
        st[int(e)] = "native"
    for e, k in p.get("expert_state", {}).get(str(L), {}).items():
        st[int(e)] = k
    return st


def _bank_files(K: int, L: int) -> list[Path]:
    out = []
    for root in str(plan()["qdir"]).split(":"):
        for nm in (f"L{L:02d}.safetensors", f"L{L:02d}.part.safetensors"):
            f = Path(root) / f"K{K}" / nm
            if f.exists():
                out.append(f)
    return out


def _buffers(dev: torch.device):
    i = dev.index
    if i not in _BUFS:
        C = int(_ext().exl3_moe_max_concurrency(i))
        R = FUSED_ROWS
        _BUFS[i] = (torch.empty((C, R, HID), dtype=torch.half, device=dev),
                    torch.empty((C, R, HID), dtype=torch.half, device=dev),
                    torch.empty((C, R, INTER), dtype=torch.half, device=dev),
                    torch.empty((C, R, INTER), dtype=torch.half, device=dev))
        logger.info("st-exl3: fused buffers on cuda:%d C=%d R=%d (%.2f GiB)", i, C, R,
                    sum(t.numel() * 2 for t in _BUFS[i]) / 2**30)
    return _BUFS[i]


_PRIMED: set[int] = set()


def _release_host_heap() -> None:
    """Return freed bank-staging memory to the OS. GB10 memory is unified: glibc arenas that keep the CPU copies of
    the banks (freed after .to(dev)) count against the same pool the GPU allocates from."""
    import ctypes, gc
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def _prime(dev: torch.device, buckets) -> None:
    """One empty exl3_moe launch per device at load time (eager): exllamav3 allocates its lock / scheduler buffer
    (cudaMalloc) on the first exl3_moe call, which fails if that first call is inside CUDA graph capture."""
    if dev.index in _PRIMED or not buckets or torch.cuda.is_current_stream_capturing():
        return
    b = buckets[0]
    y = torch.zeros((1, HID), dtype=torch.half, device=dev)
    out = torch.zeros((1, HID), dtype=torch.float, device=dev)
    cnt = torch.zeros((b.E + 1,), dtype=torch.long, device=dev)          # no expert has rows: nothing to compute
    _pf.fused(_ext(), b, y, out, cnt, torch.zeros((1,), dtype=torch.long, device=dev),
              torch.zeros((1,), dtype=torch.half, device=dev), _buffers(dev), -1)
    torch.cuda.synchronize(dev)
    _PRIMED.add(dev.index)


class _Bucket:
    """All local experts of one layer that share one K: pointer tables for exl3_moe + LinearEXL3 modules."""

    def __init__(self, K: int, experts: list[int], lin: dict, dev):
        self.K, self.experts, self.E = K, experts, len(experts)
        self.lin = lin                                  # e -> (w1, w3, w2) LinearEXL3
        ptr = lambda w, a: torch.tensor([getattr(lin[e][w], a).data_ptr() for e in experts], dtype=torch.long, device=dev)
        self.g = (ptr(0, "trellis"), ptr(0, "suh"), ptr(0, "svh"))
        self.u = (ptr(1, "trellis"), ptr(1, "suh"), ptr(1, "svh"))
        self.d = (ptr(2, "trellis"), ptr(2, "suh"), ptr(2, "svh"))
        l0 = lin[experts[0]][0]
        self.mcg, self.mul1 = bool(l0.mcg), bool(l0.mul1)


class Exl3RoutedMoEMethod(FusedMoEMethodBase):
    def __init__(self, moe, layer_id: int):
        super().__init__(moe)
        self.layer_id = layer_id
        self.buckets: list[_Bucket] = []
        # tests only (ST_EXL3_SKIP_NATIVE=1): leave native experts out (zero contribution) instead of refusing the plan
        self.skip_native = os.environ.get("ST_EXL3_SKIP_NATIVE") == "1"

    # ---- weights
    def create_weights(self, layer, num_experts, hidden_size, intermediate_size_per_partition, params_dtype,
                       **extra_weight_attrs):
        layer.st_exl3_layer = self.layer_id          # no tensors: experts are read from the banks after loading

    def get_fused_moe_quant_config(self, layer):
        return None

    def process_weights_after_loading(self, layer) -> None:
        from safetensors import safe_open
        from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
        from exllamav3.modules.quant.exl3 import LinearEXL3
        t0 = time.time()
        L = self.layer_id
        tp, rk = get_tensor_model_parallel_world_size(), get_tensor_model_parallel_rank()
        dev = torch.device("cuda", torch.cuda.current_device())
        nloc = N_EXPERTS // tp
        lo, hi = rk * nloc, (rk + 1) * nloc
        st = _states(L)
        byk: dict[int, list[int]] = {}
        for e in range(lo, hi):
            if st[e] == "native":
                if self.skip_native:
                    continue
                raise RuntimeError(f"st-exl3: layer {L} expert {e} is native; serving plans must be all-EXL3 (K5n)")
            byk.setdefault(int(st[e][1:]), []).append(e)
        nbytes = 0
        for K, ids in sorted(byk.items()):
            want, tens = set(ids), {}
            for f in _bank_files(K, L):
                left = want - set(tens)
                if not left:
                    break
                with safe_open(str(f), framework="pt", device="cpu") as h:
                    for k in h.keys():
                        p = k.split(".")
                        if len(p) > 5 and p[1] == str(L) and int(p[4]) in left:
                            tens.setdefault(int(p[4]), {})[f"{p[5]}.{p[6]}"] = h.get_tensor(k)
            miss = [e for e in ids if e not in tens]
            if miss:
                raise RuntimeError(f"st-exl3: layer {L} K{K}: {len(miss)} experts missing in banks {plan()['qdir']} (e.g. {miss[:6]})")
            lin = {}
            for e in ids:
                t = {k: v.to(dev) for k, v in tens[e].items()}
                nbytes += sum(v.numel() * v.element_size() for v in t.values())
                mk = lambda w, i, o: LinearEXL3(config=None, in_features=i, out_features=o, scale=None, su=None, sv=None,
                                                suh=t.get(f"{w}.suh"), svh=t.get(f"{w}.svh"), trellis=t[f"{w}.trellis"],
                                                mcg=t.get(f"{w}.mcg"), mul1=t.get(f"{w}.mul1"), bias=None,
                                                out_dtype=torch.half, key=f"layers.{L}.ffn.experts.{e}.{w}")
                lin[e] = (mk("w1", HID, INTER), mk("w3", HID, INTER), mk("w2", INTER, HID))
                if lin[e][0].K != K:
                    raise RuntimeError(f"st-exl3: layer {L} expert {e}: bank K{lin[e][0].K} != plan K{K}")
            self.buckets.append(_Bucket(K, ids, lin, dev))
        # global id -> local slot inside its bucket; -1 for experts of other ranks
        maps = []
        for b in self.buckets:
            m = torch.full((N_EXPERTS,), b.E, dtype=torch.long)    # sentinel slot E (ignored by the kernel)
            m[torch.tensor(b.experts)] = torch.arange(b.E)
            maps.append(m.to(dev))
        self.maps = maps
        self.st_tables = _pf.LayerTables(self.buckets, N_EXPERTS, dev) if self.buckets else None
        _buffers(dev)
        _prime(dev, self.buckets)
        _release_host_heap()
        if L in (0, 39) or os.environ.get("ST_EXL3_LOG"):
            logger.info("st-exl3: layer %d rank %d/%d experts %d-%d buckets %s, %.2f GiB, %.1fs", L, rk, tp, lo, hi - 1,
                        {f"K{b.K}": b.E for b in self.buckets}, nbytes / 2**30, time.time() - t0)

    # ---- forward
    def prepare_workspace(self, hidden_states, shared_workspace_size):
        # the routed path needs no arena; give the shared experts their own byte scratch (same spec as the
        # modular kernel's _allocate_buffers: uint8, shared_workspace_size bytes)
        sw = torch.empty((shared_workspace_size,), dtype=torch.uint8, device=hidden_states.device) \
            if shared_workspace_size else None
        return (None, None, None), sw

    def apply_with_workspace(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input, workspace):
        return self.apply(layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input)

    def apply(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
        ext = _ext()
        T, H = x.shape
        y = x.to(torch.half).contiguous()
        out = torch.zeros((T, H), dtype=torch.float, device=x.device)
        if T == 0:
            return out.to(x.dtype)
        tsg, tsu, tig, tiu = _buffers(x.device)
        top_k = topk_ids.shape[-1]
        flat_ids = topk_ids.reshape(-1).long()
        flat_w = topk_weights.reshape(-1).to(torch.half)
        flat_tok = torch.arange(T, device=x.device).repeat_interleave(top_k)
        all_fused = T * top_k <= FUSED_ROWS
        if not all_fused and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(f"st-exl3: CUDA graph capture at {T} tokens x top-{top_k} > ST_EXL3_FUSED_ROWS={FUSED_ROWS}; "
                               "the prefill route host-syncs. Keep capture sizes small and VLLM_USE_BREAKABLE_CUDAGRAPH=0.")
        bufs = (tsg, tsu, tig, tiu)
        if not all_fused and _pf.PREFILL == "st" and _pf.MIN_ROWS <= 1 and _pf.ST_GPU and self.st_tables is not None:
            # prefill, st route, all buckets at once with no host sync (GPU-built tile lists)
            _pf.run_st_layer(self.st_tables, y, out, flat_tok, flat_w, flat_ids, top_k)
            return out.to(x.dtype)
        for b, m in zip(self.buckets, self.maps):
            loc = m[flat_ids]
            if not all_fused:
                _pf.run_bucket(b, y, out, flat_tok, flat_w, loc, bufs)     # prefill: host-synced routes
                continue
            # graph-safe: no host sync, every expert fits the fused kernel's row capacity
            order = loc.argsort()
            # bincount reads max(loc) back to the host (not capture-safe); scatter_add gives the same int64 counts
            cnt = torch.zeros(b.E + 1, dtype=torch.long, device=loc.device).scatter_add_(0, loc, torch.ones_like(loc))
            _pf.fused(ext, b, y, out, cnt, flat_tok[order], flat_w[order], bufs, -1)
        return out.to(x.dtype)
