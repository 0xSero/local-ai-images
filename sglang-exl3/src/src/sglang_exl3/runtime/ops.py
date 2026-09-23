"""Custom ops. One opaque op per layer kind: vLLM traces `apply()` once with Dynamo and freezes it,
so every row-count branch, dtype cast, pad and trim lives in here (docs/ARCHITECTURE.md, L3)."""
from __future__ import annotations

import torch

from ..kernels import hopper, reference

_lib = torch.library.Library("sglang_exl3", "FRAGMENT")  # must stay alive at module scope
import os

# Row-count switch points, measured on H200 (docs/STATUS.md): ExLlamaV3's trellis kernel costs 11.8 / 16.4 / 18.2 /
# 34.8 / 67.9 ms per step at 1 / 4 / 16 / 32 / 64 rows; a resident decoded weight + cuBLAS is flat at ~16 ms; decoding
# the weight per call adds roughly one rows=1 kernel pass (~12 ms). ExLlamaV3's own switch point (144) is tuned for
# bandwidth-starved consumer GPUs. To be replaced by the per-shape dispatch plan.
DENSE_ROWS = int(os.environ.get("SGLANG_EXL3_DENSE_ROWS", "144"))   # 3090: kernel wins to ~128 rows (ExLlamaV3 switch 144)                 # transient decoded weight
DENSE_CACHED_ROWS = int(os.environ.get("SGLANG_EXL3_DENSE_CACHED_ROWS", "16"))   # resident decoded weight
# Below the dense switch, matrices that share an input run as one sliced multi-matrix launch (what ExLlamaV3 itself
# does for q/k/v). Measured on H200: q/k/v 72 -> 47 us at 8 rows, gate/up 98 -> 81 us; at 1-2 rows the separate
# int8 GEMV launches still win for wide groups, so each group carries its own minimum row count.


# Switch points of OUR kernel, measured on H200 (research/11 section 18, all 27B linears per forward, ms): kernel
# 10.3 / 12.5 / 18.1 / 28.8 / 51.6 / 99.6 at 16 / 32 / 64 / 128 / 256 / 512 rows against 43 .. 69 for decode + cuBLAS
# (the per-call reconstruct alone is 28 ms) -> the kernel serves everything below 256 rows (speculative verification
# batches, chunked-prefill tails); with resident decoded weights the dense path has no reconstruct and wins from ~48.
HOPPER_DENSE_ROWS = int(os.environ.get("SGLANG_EXL3_HOPPER_DENSE_ROWS", "256"))
HOPPER_DENSE_CACHED_ROWS = int(os.environ.get("SGLANG_EXL3_HOPPER_DENSE_CACHED_ROWS", "48"))
_HOPPER_DENSE = os.environ.get("SGLANG_EXL3_HOPPER_DENSE", "1") == "1"
_HOPPER_DENSE_MAX_N = 65536     # lm_head-sized outputs keep the column-sliced reference path (bounded transients)


def dense_group(x: torch.Tensor, trellis: list[torch.Tensor], suh_cat: torch.Tensor, svh: list[torch.Tensor],
                codebook: int, out: torch.Tensor, weights: list[torch.Tensor] | None = None) -> torch.Tensor:
    """Many-row (prefill) path of a fused group on our Hadamard launches: x fp16 | bf16 [rows, k] goes in as is, ONE
    batched input transform for all matrices, per matrix decode (ExLlamaV3's reconstruct, or the resident `weights`)
    + cuBLAS fp16 GEMM, and an output transform that writes straight into the matrix's column span of `out` in
    out's dtype. Same arithmetic as ExLlamaV3's reconstruct path (04 section 2.3) and as the explicit-cast path it
    replaces (the conversions are torch's value conversions, kernels/hopper.py), minus 2 casts, 1 copy and
    len(trellis) - 1 input-transform launches per call."""
    rows = x.shape[0]
    xh = hopper.had_in_group(x, suh_cat)
    col0 = 0
    for i, (t, sv) in enumerate(zip(trellis, svh)):
        w = weights[i] if weights is not None else reference.reconstruct(t, codebook)
        y = torch.mm(xh[i * rows:(i + 1) * rows], w)
        del w
        hopper.had_out_into(y, sv, out, col0)
        col0 += y.shape[1]
    return out


def _linear(x: torch.Tensor, trellis: list[torch.Tensor], suh: list[torch.Tensor], svh: list[torch.Tensor],
            dense: list[torch.Tensor], sliced: list[torch.Tensor], sliced_min_rows: int,
            hopper_w: list[torch.Tensor], hopper_ends: list[int],
            codebooks: list[int], out_widths: list[int]) -> torch.Tensor:
    """Fused EXL3 linear: every matrix reads the same x and writes an adjacent output span.
    Each matrix has its own input scales, bitrate and codebook, so they are launched one by one."""
    k = (trellis[0].shape[0] if trellis else hopper_w[0].shape[0]) * 16
    rows = x.reshape(-1, x.shape[-1])
    if hopper_w and not trellis:
        # memory-lean layout (3090): only the repacked form is resident; rebuild int16 trellis per shard on demand
        b = hopper_w[0]
        bounds = [0, *[e // 64 for e in hopper_ends], b.shape[1]]
        trellis = [hopper.unprepare_matrix(b[:, a:c].contiguous()) for a, c in zip(bounds[:-1], bounds[1:])]
    if (hopper_w and rows.shape[0] < (HOPPER_DENSE_CACHED_ROWS if dense else HOPPER_DENSE_ROWS) and rows.shape[1] == k
            and rows.dtype in (torch.float16, torch.bfloat16)):
        # Our Marlin-template kernel: one launch set for the whole fused group, written straight into the fused
        # output. bf16 goes in and comes out as is: the kernel converts bf16 -> fp16 on load and fp16 -> bf16 on
        # store with torch's value conversions, i.e. exactly what `.to(float16)` ... `.to(x.dtype)` did here before
        # (incl. bf16 magnitudes > 65504 -> inf), without the two cast launches and their temporaries.
        y = hopper.run(rows.contiguous(), hopper_w[0], hopper_w[1], hopper_w[2], hopper_ends, codebooks[0])
        return y.view(*x.shape[:-1], y.shape[1])
    if (hopper_w and rows.shape[1] == k and rows.dtype in (torch.float16, torch.bfloat16)
            and max(out_widths) <= _HOPPER_DENSE_MAX_N and _HOPPER_DENSE):
        # Many rows (prefill, big verification batches): decoded weight + cuBLAS between OUR transforms, bf16 in/out,
        # one input-transform launch per fused group, results written straight into the fused output.
        out = torch.empty((rows.shape[0], sum(out_widths)), dtype=x.dtype, device=x.device)
        ends = [0, *hopper_ends, out.shape[1]]
        svs = [hopper_w[2][a:b] for a, b in zip(ends[:-1], ends[1:])]
        dense_group(rows.contiguous(), trellis, hopper_w[1], svs, codebooks[0], out, dense if dense else None)
        return out.view(*x.shape[:-1], out.shape[1])
    h = rows.to(torch.float16)
    if h.shape[1] < k:                               # stored input width is padded to 128
        h = torch.nn.functional.pad(h, (0, k - h.shape[1]))
    h = h.contiguous()
    padded = any(w > t.shape[1] * 16 for w, t in zip(out_widths, trellis))
    out = (torch.zeros if padded else torch.empty)((h.shape[0], sum(out_widths)), dtype=x.dtype, device=x.device)
    offset = 0
    n_rows = h.shape[0]
    dense_switch = DENSE_CACHED_ROWS if dense else DENSE_ROWS
    if hopper_w and n_rows < dense_switch:       # padded input width or an fp32 caller: explicit casts
        y = hopper.run(h, hopper_w[0], hopper_w[1], hopper_w[2], hopper_ends, codebooks[0])
        return y.to(x.dtype).view(*x.shape[:-1], y.shape[1])
    if sliced and sliced_min_rows <= n_rows < dense_switch and n_rows <= sliced[9].shape[0]:   # tables[9] = first out buffer
        ys = reference.run_sliced(h, sliced, trellis[0].shape[2] / 16, codebooks[0] == 1, codebooks[0] == 2)
        for y, width in zip(ys, out_widths):
            keep = min(width, y.shape[1])
            out[:, offset:offset + keep] = y[:, :keep]
            offset += width
        return out.view(*x.shape[:-1], out.shape[1])
    for i, (t, su, sv, cb, width) in enumerate(zip(trellis, suh, svh, codebooks, out_widths)):
        if dense and n_rows >= DENSE_CACHED_ROWS:
            y = reference.dense_cached(h, dense[i], su, sv)
        elif n_rows >= DENSE_ROWS:
            y = reference.dense_forward(h, t, su, sv, cb)
        else:
            y = reference.gemm(h, t, su, sv, cb)
        keep = min(width, y.shape[1])                # stored output columns beyond the true width are noise
        out[:, offset:offset + keep] = y[:, :keep]
        offset += width
    return out.view(*x.shape[:-1], out.shape[1])


def _linear_fake(x, trellis, suh, svh, dense, sliced, sliced_min_rows, hopper_w, hopper_ends, codebooks, out_widths):
    return x.new_empty((*x.shape[:-1], sum(out_widths)))


_lib.define("linear(Tensor x, Tensor[] trellis, Tensor[] suh, Tensor[] svh, Tensor[] dense, Tensor[] sliced, int sliced_min_rows, Tensor[] hopper_w, int[] hopper_ends, int[] codebooks, int[] out_widths) -> Tensor")
_lib.impl("linear", _linear, "CUDA")
torch.library.register_fake("sglang_exl3::linear", _linear_fake, lib=_lib)

linear = torch.ops.sglang_exl3.linear
