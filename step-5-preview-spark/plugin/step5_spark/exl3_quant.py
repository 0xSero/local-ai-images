# SPDX-License-Identifier: MIT
"""EXL3 (stock turboderp exllamav3 trellis, MUL1 codebook) weights for Step-5 inside vLLM.

quant_method "step5_exl3". The checkpoint keeps BF16 for everything that is not quantized (embeddings, norms, router,
g_proj, lm_head, vision, MTP layers) and stores EXL3 tensors under <model>/exl3/:
  exl3/experts/L{LL}.safetensors  model.layers.{L}.moe.experts.{E}.{gate,up,down}_proj.{trellis,suh,svh,mul1}
  exl3/body/L{LL}.safetensors     model.layers.{L}.{self_attn.{q,k,v,o}_proj | share_expert.{gate,up,down}_proj |
                                   mlp.{gate,up,down}_proj}.{trellis,suh,svh,mul1}
Routed experts: expert parallel inside the TP group (rank r owns experts [r*E/tp, (r+1)*E/tp)); decode / graph shapes
run exllamav3 exl3_moe, prefill runs the st_moe_ext grouped GEMM (see exl3_prefill.py).
Body: lossless TP slicing of the trellis at 128-multiples (column-parallel along outputs, row-parallel along inputs);
each part runs exllamav3 LinearEXL3 (GEMV for <=144 rows, reconstruct + hgemm above) inside a torch custom op.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import FusedMoEMethodBase
from vllm.model_executor.layers.linear import (LinearBase, LinearMethodBase, RowParallelLinear,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

from step5_spark import exl3_prefill as _pf

logger = init_logger(__name__)

HID, INTER = _pf.HID, _pf.INTER
# Expert tensor parallelism (STEP5_EXPERT_TP=1): every rank holds a 1/tp column slice of every expert (gate/up outputs,
# down inputs; 128-aligned, lossless) instead of whole experts for 1/tp of the expert ids. Balanced per-layer work and all
# routed experts in flight on every rank. Requires ST_EXL3_INTER = 1536 / tp (the launcher sets it).
EXPERT_TP = os.environ.get("STEP5_EXPERT_TP", "0") == "1"
FUSED_ROWS = _pf.FUSED_ROWS
_BODY_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.(self_attn\.(?:qkv_proj|o_proj)|moe\.share_expert\.(?:gate_up_proj|down_proj)"
                      r"|mlp\.(?:gate_up_proj|down_proj))$")
_EXPERTS_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.moe\.experts$")
_PARTS = {"qkv_proj": ("q_proj", "k_proj", "v_proj"), "gate_up_proj": ("gate_proj", "up_proj"),
          "o_proj": ("o_proj",), "down_proj": ("down_proj",)}


def _ext():
    import exllamav3_ext
    return exllamav3_ext


_MODEL_DIR: Path | None = None


def model_dir() -> Path:
    global _MODEL_DIR
    if _MODEL_DIR is None:
        d = os.environ.get("STEP5_MODEL_DIR")
        if not d:
            from vllm.config import get_current_vllm_config
            d = get_current_vllm_config().model_config.model
        p = Path(d)
        if not p.is_dir():
            from huggingface_hub import snapshot_download
            p = Path(snapshot_download(d, local_files_only=True))
        _MODEL_DIR = p
    return _MODEL_DIR


def _tp():
    from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
    return get_tensor_model_parallel_world_size(), get_tensor_model_parallel_rank()


def _read(path: Path, keys: list[str]) -> dict[str, torch.Tensor]:
    from safetensors import safe_open
    with safe_open(str(path), framework="pt", device="cpu") as h:
        have = set(h.keys())
        return {k: h.get_tensor(k) for k in keys if k in have}


def _make_linear(t: dict, prefix: str, ni: int, no: int, dev, key: str):
    from exllamav3.modules.quant.exl3 import LinearEXL3
    g = lambda n: t.get(f"{prefix}.{n}").to(dev) if f"{prefix}.{n}" in t else None
    return LinearEXL3(config=None, in_features=ni, out_features=no, scale=None, su=None, sv=None,
                      suh=g("suh"), svh=g("svh"), trellis=g("trellis"), mcg=g("mcg"), mul1=g("mul1"),
                      bias=None, out_dtype=torch.half, key=key)


def _release_host_heap() -> None:
    import ctypes, gc
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


# ----------------------------------------------------------------------------------------------------------- body
_LINEARS: list = []


@torch.library.custom_op("step5_spark::exl3_linear", mutates_args=())
def exl3_linear(x: torch.Tensor, handle: int, out_features: int) -> torch.Tensor:
    return _LINEARS[handle].forward(x, {})


@exl3_linear.register_fake
def _(x, handle, out_features):
    return x.new_empty((*x.shape[:-1], out_features))


_HYBRID: list = []           # per hybrid layer: [(handle, out_features, slice|None), ...]
HYBRID_ROWS = int(os.environ.get("STEP5_HYBRID_ROWS", "256"))
FUSE = os.environ.get("STEP5_FUSE_BODY", "1") != "0"   # fuse q/k/v and gate/up EXL3 GEMVs when they share su


@torch.library.custom_op("step5_spark::hybrid_linear", mutates_args=())
def hybrid_linear(x: torch.Tensor, weight: torch.Tensor, key: int, out_features: int) -> torch.Tensor:
    """Dense BF16 GEMM for prefill-sized inputs (> STEP5_HYBRID_ROWS rows), EXL3 GEMV path for decode-sized inputs."""
    if x.shape[0] > HYBRID_ROWS:
        return torch.nn.functional.linear(x, weight)
    y = x.to(torch.half).contiguous()
    outs = []
    for h, no, ranges in _HYBRID[key]:
        o = _LINEARS[h].forward(y, {})
        outs += [o if (a == 0 and b == no) else o[:, a:b] for a, b in ranges]
    out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=-1)
    return out.to(x.dtype)


@hybrid_linear.register_fake
def _(x, weight, key, out_features):
    return x.new_empty((*x.shape[:-1], out_features))


class Exl3LinearMethod(LinearMethodBase):
    def __init__(self, layer_id: int, module: str, hybrid: bool = False):
        self.layer_id, self.module = layer_id, module        # e.g. 5, "self_attn.qkv_proj"
        self.hybrid = hybrid                                 # also keep the BF16 weight for prefill-sized inputs

    def create_weights(self, layer, input_size_per_partition, output_partition_sizes, input_size, output_size,
                       params_dtype, **extra_weight_attrs):
        layer.st5_in, layer.st5_outs = input_size_per_partition, list(output_partition_sizes)
        layer.st5_row = isinstance(layer, RowParallelLinear)
        if self.hybrid:
            UnquantizedLinearMethod().create_weights(layer, input_size_per_partition, output_partition_sizes,
                                                     input_size, output_size, params_dtype, **extra_weight_attrs)

    def process_weights_after_loading(self, layer) -> None:
        tp, rk = _tp()
        dev = torch.device("cuda", torch.cuda.current_device())
        L = self.layer_id
        base, leaf = self.module.rsplit(".", 1)
        base = base.replace("moe.share_expert", "share_expert")
        parts = _PARTS[leaf]
        keys = [f"model.layers.{L}.{base}.{p}.{n}" for p in parts for n in ("trellis", "suh", "svh", "mul1", "mcg")]
        t = _read(model_dir() / "exl3" / "body" / f"L{L:02d}.safetensors", keys)
        prepared = []
        for p, no_loc in zip(parts, layer.st5_outs):
            pre = f"model.layers.{L}.{base}.{p}"
            tr, suh, svh = t[f"{pre}.trellis"], t[f"{pre}.suh"], t[f"{pre}.svh"]
            ni_full, no_full = tr.shape[0] * 16, tr.shape[1] * 16
            if layer.st5_row:            # split inputs
                ni = ni_full // tp
                assert ni == layer.st5_in and ni % 128 == 0, (pre, ni)
                tr, suh, no = tr[rk * ni // 16:(rk + 1) * ni // 16], suh[rk * ni:(rk + 1) * ni], no_full
            elif (no_full // tp) % 128 == 0:   # split outputs at 128-multiples (Hadamard blocks stay whole)
                no = no_full // tp
                assert no == no_loc, (pre, no, no_loc)
                tr, svh, ni = tr[:, rk * no // 16:(rk + 1) * no // 16], svh[rk * no:(rk + 1) * no], ni_full
            else:                        # small part (e.g. k/v at TP4: 192 per rank): keep it whole, slice the output
                no, ni = no_full, ni_full
                assert no_full // tp == no_loc, (pre, no_full, no_loc)
            sl = (rk * no_loc, (rk + 1) * no_loc) if (not layer.st5_row and no != no_loc) else (0, no)
            prepared.append((pre, tr.contiguous(), suh.contiguous(), svh.contiguous(), ni, no, sl))
        # (handle, out_features, output column ranges to keep, in order)
        lins = []
        fuse = (len(prepared) > 1 and not layer.st5_row and FUSE
                and all(torch.equal(q[2], prepared[0][2]) for q in prepared[1:]))
        if fuse:   # shared input signs (one su): one GEMV over the concatenated output tiles
            pre0 = prepared[0][0]
            ft = {f"{pre0}.trellis": torch.cat([q[1] for q in prepared], dim=1).contiguous(),
                  f"{pre0}.suh": prepared[0][2], f"{pre0}.svh": torch.cat([q[3] for q in prepared]).contiguous()}
            for n in ("mul1", "mcg"):
                if f"{pre0}.{n}" in t:
                    ft[f"{pre0}.{n}"] = t[f"{pre0}.{n}"]
            no_tot, ranges, off = sum(q[5] for q in prepared), [], 0
            for q in prepared:
                ranges.append((off + q[6][0], off + q[6][1])); off += q[5]
            merged = [ranges[0]]
            for r in ranges[1:]:
                merged[-1:] = [(merged[-1][0], r[1])] if r[0] == merged[-1][1] else [merged[-1], r]
            _LINEARS.append(_make_linear(ft, pre0, prepared[0][4], no_tot, dev, pre0 + "+fused"))
            lins.append((len(_LINEARS) - 1, no_tot, merged))
        else:
            for pre, tr, suh, svh, ni, no, sl in prepared:
                t[f"{pre}.trellis"], t[f"{pre}.suh"], t[f"{pre}.svh"] = tr, suh, svh
                _LINEARS.append(_make_linear(t, pre, ni, no, dev, pre))
                lins.append((len(_LINEARS) - 1, no, [sl]))
        layer.st5_handles = lins
        if self.hybrid:
            _HYBRID.append(lins)
            layer.st5_hybrid_key = len(_HYBRID) - 1
            # plain-tensor alias: passing vLLM's Parameter subclass into the custom op routes every call through
            # Parameter.__torch_function__ (seen in the stuck V1+MTP capture stack)
            layer.st5_dense_w = layer.weight.data
        _release_host_heap()

    def apply(self, layer, x, bias=None):
        if self.hybrid:
            shape = x.shape
            out = hybrid_linear(x.reshape(-1, shape[-1]), layer.st5_dense_w, layer.st5_hybrid_key, sum(layer.st5_outs))
            out = out.reshape(*shape[:-1], out.shape[-1])
            return out if bias is None else out + bias
        shape = x.shape
        y = x.reshape(-1, shape[-1]).to(torch.half).contiguous()
        outs = []
        for h, no, ranges in layer.st5_handles:
            o = exl3_linear(y, h, no)
            outs += [o if (a == 0 and b == no) else o[:, a:b] for a, b in ranges]
        out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=-1)
        out = out.to(x.dtype).reshape(*shape[:-1], out.shape[-1])
        if bias is not None:
            out = out + bias
        return out


# -------------------------------------------------------------------------------------------------------- experts
_BUFS: dict[int, tuple] = {}
_PRIMED: set[int] = set()


def _buffers(dev):
    i = dev.index
    if i not in _BUFS:
        C = int(_ext().exl3_moe_max_concurrency(i))
        R = FUSED_ROWS
        _BUFS[i] = (torch.empty((C, R, HID), dtype=torch.half, device=dev),
                    torch.empty((C, R, HID), dtype=torch.half, device=dev),
                    torch.empty((C, R, INTER), dtype=torch.half, device=dev),
                    torch.empty((C, R, INTER), dtype=torch.half, device=dev))
    return _BUFS[i]


def _prime(dev, buckets) -> None:
    """exllamav3 allocates its exl3_moe scheduler buffer on the first call; do it eagerly, outside graph capture."""
    if dev.index in _PRIMED or not buckets or torch.cuda.is_current_stream_capturing():
        return
    b = buckets[0]
    y = torch.zeros((1, HID), dtype=torch.half, device=dev)
    out = torch.zeros((1, HID), dtype=torch.float, device=dev)
    cnt = torch.zeros((b.E + 1,), dtype=torch.long, device=dev)
    _pf.fused(_ext(), b, y, out, cnt, torch.zeros((1,), dtype=torch.long, device=dev),
              torch.zeros((1,), dtype=torch.half, device=dev), _buffers(dev), -1)
    torch.cuda.synchronize(dev)
    _PRIMED.add(dev.index)


class _Bucket:
    def __init__(self, K, experts, lin, dev):
        self.K, self.experts, self.E, self.lin = K, experts, len(experts), lin
        ptr = lambda w, a: torch.tensor([getattr(lin[e][w], a).data_ptr() for e in experts], dtype=torch.long, device=dev)
        self.g = (ptr(0, "trellis"), ptr(0, "suh"), ptr(0, "svh"))
        self.u = (ptr(1, "trellis"), ptr(1, "suh"), ptr(1, "svh"))
        self.d = (ptr(2, "trellis"), ptr(2, "suh"), ptr(2, "svh"))
        l0 = lin[experts[0]][0]
        self.mcg, self.mul1 = bool(l0.mcg), bool(l0.mul1)


class Exl3MoEMethod(FusedMoEMethodBase):
    def __init__(self, moe, layer_id: int, n_experts: int):
        super().__init__(moe)
        self.layer_id, self.n_experts = layer_id, n_experts
        self.buckets: list[_Bucket] = []

    def create_weights(self, layer, num_experts, hidden_size, intermediate_size_per_partition, params_dtype,
                       **extra_weight_attrs):
        layer.st5_layer = self.layer_id

    def get_fused_moe_quant_config(self, layer):
        return None

    def process_weights_after_loading(self, layer) -> None:
        t0 = time.time()
        L, N = self.layer_id, self.n_experts
        tp, rk = _tp()
        dev = torch.device("cuda", torch.cuda.current_device())
        if EXPERT_TP:
            lo, hi = 0, N
            assert INTER * tp == 1536 and INTER % 128 == 0, ("set ST_EXL3_INTER=1536/tp", INTER, tp)
        else:
            nloc = N // tp
            lo, hi = rk * nloc, (rk + 1) * nloc
        projs = (("gate_proj", HID, INTER), ("up_proj", HID, INTER), ("down_proj", INTER, HID))
        keys = [f"model.layers.{L}.moe.experts.{e}.{p}.{n}" for e in range(lo, hi) for p, _, _ in projs
                for n in ("trellis", "suh", "svh", "mul1", "mcg")]
        t = _read(model_dir() / "exl3" / "experts" / f"L{L:02d}.safetensors", keys)
        if EXPERT_TP:   # slice gate/up along outputs, down along inputs (tiles of 16, Hadamard blocks of 128 stay whole)
            a, b = rk * INTER, (rk + 1) * INTER
            for e in range(lo, hi):
                pre = f"model.layers.{L}.moe.experts.{e}"
                for p in ("gate_proj", "up_proj"):
                    t[f"{pre}.{p}.trellis"] = t[f"{pre}.{p}.trellis"][:, a // 16:b // 16].contiguous()
                    t[f"{pre}.{p}.svh"] = t[f"{pre}.{p}.svh"][a:b].contiguous()
                t[f"{pre}.down_proj.trellis"] = t[f"{pre}.down_proj.trellis"][a // 16:b // 16].contiguous()
                t[f"{pre}.down_proj.suh"] = t[f"{pre}.down_proj.suh"][a:b].contiguous()
        lin, byk, nbytes = {}, {}, 0
        for e in range(lo, hi):
            pre = f"model.layers.{L}.moe.experts.{e}"
            if f"{pre}.gate_proj.trellis" not in t:
                raise RuntimeError(f"step5-exl3: layer {L} expert {e} missing in exl3/experts/L{L:02d}.safetensors")
            lin[e] = tuple(_make_linear(t, f"{pre}.{p}", ni, no, dev, f"{pre}.{p}") for p, ni, no in projs)
            nbytes += sum(v.numel() * v.element_size() for k, v in t.items() if k.startswith(pre + "."))
            K = lin[e][0].K
            if not (lin[e][1].K == lin[e][2].K == K):
                raise RuntimeError(f"step5-exl3: layer {L} expert {e}: mixed K within one expert is not supported")
            byk.setdefault(K, []).append(e)
        del t
        self.buckets = [_Bucket(K, ids, lin, dev) for K, ids in sorted(byk.items())]
        maps = []
        for b in self.buckets:
            m = torch.full((N,), b.E, dtype=torch.long)
            m[torch.tensor(b.experts)] = torch.arange(b.E)
            maps.append(m.to(dev))
        self.maps = maps
        self.st_tables = _pf.LayerTables(self.buckets, N, dev) if self.buckets else None
        _buffers(dev)
        _prime(dev, self.buckets)
        _release_host_heap()
        if L in (3, 90) or os.environ.get("ST_EXL3_LOG"):
            logger.info("step5-exl3: layer %d rank %d/%d experts %d-%d buckets %s, %.2f GiB, %.1fs", L, rk, tp, lo,
                        hi - 1, {f"K{b.K}": b.E for b in self.buckets}, nbytes / 2**30, time.time() - t0)

    def prepare_workspace(self, hidden_states, shared_workspace_size):
        sw = torch.empty((shared_workspace_size,), dtype=torch.uint8, device=hidden_states.device) \
            if shared_workspace_size else None
        return (None, None, None), sw

    def apply_with_workspace(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input, workspace):
        return self.apply(layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input)

    def apply(self, layer, x, topk_weights, topk_ids, shared_experts=None, shared_experts_input=None):
        ext = _ext()
        T, H = x.shape
        y = x.to(torch.half).contiguous()
        out = torch.zeros((T, H), dtype=torch.float, device=x.device)
        if T == 0:
            return out.to(x.dtype)
        bufs = _buffers(x.device)
        top_k = topk_ids.shape[-1]
        flat_ids = topk_ids.reshape(-1).long()
        flat_w = topk_weights.reshape(-1).to(torch.half)
        flat_tok = torch.arange(T, device=x.device).repeat_interleave(top_k)
        all_fused = T * top_k <= FUSED_ROWS
        if not all_fused and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(f"step5-exl3: CUDA graph capture at {T} tokens x top-{top_k} > ST_EXL3_FUSED_ROWS={FUSED_ROWS}")
        if not all_fused and _pf.PREFILL == "st" and _pf.MIN_ROWS <= 1 and _pf.ST_GPU and self.st_tables is not None:
            _pf.run_st_layer(self.st_tables, y, out, flat_tok, flat_w, flat_ids, top_k)
            return out.to(x.dtype)
        for b, m in zip(self.buckets, self.maps):
            loc = m[flat_ids]
            if not all_fused:
                _pf.run_bucket(b, y, out, flat_tok, flat_w, loc, bufs)
                continue
            order = loc.argsort()
            cnt = torch.zeros(b.E + 1, dtype=torch.long, device=loc.device).scatter_add_(0, loc, torch.ones_like(loc))
            _pf.fused(ext, b, y, out, cnt, flat_tok[order], flat_w[order], bufs, -1)
        return out.to(x.dtype)


# --------------------------------------------------------------------------------------------------------- config
@register_quantization_config("step5_exl3")
class Step5Exl3Config(QuantizationConfig):
    def __init__(self, body_layers: list[int] | None = None, expert_layers: list[int] | None = None,
                 n_experts: int = 352, extra: dict[str, Any] | None = None):
        super().__init__()
        self.body_layers = set(body_layers or [])
        self.expert_layers = set(expert_layers or [])
        self.n_experts = n_experts
        self.extra = extra or {}
        # body_format: "exl3" (exl3/body banks), "bf16" (checkpoint BF16 body), "fp8" (BF16 checkpoint, online FP8)
        self.body_format = os.environ.get("STEP5_BODY_FORMAT") or self.extra.get("body_format", "exl3")
        self._fp8 = None

    def get_name(self):
        return "step5_exl3"

    def get_supported_act_dtypes(self):
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Step5Exl3Config":
        return cls(config.get("body_layers"), config.get("expert_layers"), config.get("n_experts", 352), config)

    def get_quant_method(self, layer, prefix: str):
        from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
        if isinstance(layer, RoutedExperts):
            m = _EXPERTS_RE.search(prefix)
            if m and int(m.group(1)) in self.expert_layers:
                return Exl3MoEMethod(layer.moe_config, int(m.group(1)), self.n_experts)
            return None
        if isinstance(layer, LinearBase):
            m = _BODY_RE.search(prefix)
            if m and ".mtp_block." not in prefix and "vision" not in prefix:
                if self.body_format in ("exl3", "hybrid") and int(m.group(1)) in self.body_layers:
                    return Exl3LinearMethod(int(m.group(1)), m.group(2), hybrid=self.body_format == "hybrid")
                if self.body_format == "fp8":
                    if self._fp8 is None:
                        from vllm.model_executor.layers.quantization.fp8 import Fp8Config
                        self._fp8 = Fp8Config(is_checkpoint_fp8_serialized=False, activation_scheme="dynamic")
                    return self._fp8.get_quant_method(layer, prefix)
            return UnquantizedLinearMethod()
        return None
