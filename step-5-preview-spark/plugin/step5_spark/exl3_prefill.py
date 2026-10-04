# SPDX-License-Identifier: MIT
# sovereign-trellis: prefill routes for EXL3 (trellis, mul1) routed experts (copied from the DeepSeek-V4.1-Flash-Spark
# plugin, dims/limit parameterised for Step-5: hidden 4096, expert intermediate 1536, SwiGLU limit 7).
#
# vLLM-free (the microbench imports it directly). st_exl3_moe.Exl3RoutedMoEMethod.apply calls run_bucket() once per
# K bucket for host-synced (prefill) batches; decode / CUDA-graph batches stay on the all-fused exl3_moe launch.
# ST_EXL3_PREFILL selects the route for experts with >= ST_EXL3_PREFILL_MIN_ROWS rows (the rest use exl3_moe):
#   off      (default) exl3_moe 16/32/64-row tiles up to ST_EXL3_FUSED_ROWS rows, reconstruct + hgemm above
#   st       st_moe_ext (st_moe_ext/st_moe_ext.cu): grouped GEMM over 32/64/128/256-row tiles of one expert; each
#            32-deep k stage of trellis is decoded once per block into shared memory and shared by 8 warps of fp16
#            m16n8k16 MMAs; gather + input Hadamard, out-Hadamard + SiLU*up + in-Hadamard, and the down projection's
#            out-Hadamard * router weight (fp32 vector atomics) are separate / fused passes. Use MIN_ROWS=1.
#   dq_fp16  batched exllamav3 reconstruct_had into a double-buffered fp16 scratch + torch._grouped_mm. Measured
#            slower than off on GB10 (decode-to-DRAM is bandwidth/ALU bound), kept for comparison only.
# Measurements and fidelity: campaigns/dsv41-2spark/serve2spark/kern/RESULTS.md
from __future__ import annotations

import os

import torch

HID = int(os.environ.get("ST_EXL3_HID", "4096"))
INTER = int(os.environ.get("ST_EXL3_INTER", "1536"))
FUSED_ROWS = int(os.environ.get("ST_EXL3_FUSED_ROWS", "512"))
ACT_LIMIT = float(os.environ.get("ST_EXL3_ACT_LIMIT", "7.0"))
PREFILL = os.environ.get("ST_EXL3_PREFILL", "off")
MIN_ROWS = int(os.environ.get("ST_EXL3_PREFILL_MIN_ROWS", "64"))
GROUP = int(os.environ.get("ST_EXL3_PREFILL_GROUP", "8"))
ST_BM = int(os.environ.get("ST_EXL3_ST_BM", "0"))              # 0 = 256-row tiles + 128-row remainders
ST_CHUNK_ROWS = int(os.environ.get("ST_EXL3_ST_CHUNK_ROWS", "8192"))
MTILE_T1, MTILE_T2 = 16, 32
FLOP_PER_ROW = 2 * 3 * HID * INTER

_SCRATCH: dict[tuple, tuple] = {}
_STREAMS: dict[int, torch.cuda.Stream] = {}


def _ext():
    import exllamav3_ext
    return exllamav3_ext


_STX = None


def _stx():
    """sovereign-trellis grouped-GEMM extension (st_moe_ext/), prebuilt .so on sys.path or next to this file."""
    global _STX
    if _STX is None:
        import importlib, sys
        for d in (os.environ.get("ST_MOE_EXT_DIR"), os.path.join(os.path.dirname(os.path.abspath(__file__)), "st_moe_ext"),
                  "/opt/st/patch/st_moe_ext"):
            if d and os.path.isdir(d) and d not in sys.path:
                sys.path.append(d)
        _STX = importlib.import_module("st_moe_ext")
    return _STX


def _scratch(dev: torch.device, G: int):
    key = (dev.index, G)
    if key not in _SCRATCH:
        mk = lambda k, n: torch.empty((G, k, n), dtype=torch.half, device=dev)
        # two slots (double buffer) x (w1, w3, w2): 2 * G * 70.8 MB
        _SCRATCH[key] = tuple((mk(HID, INTER), mk(HID, INTER), mk(INTER, HID)) for _ in range(2))
    return _SCRATCH[key]


def _side(dev: torch.device) -> torch.cuda.Stream:
    if dev.index not in _STREAMS:
        _STREAMS[dev.index] = torch.cuda.Stream(device=dev)
    return _STREAMS[dev.index]


def fused(ext, b, y, out, cnt, tok_s, w_s, bufs, num_active, lo=1, hi=None, mt=16):
    tsg, tsu, tig, tiu = bufs
    ext.exl3_moe(y, out, cnt, tok_s, w_s, tsg, tsu, tig, tiu, 0, b.K, b.K, b.K,
                 *b.g, *b.u, *b.d, b.mcg, b.mul1, b.mcg, b.mul1, b.mcg, b.mul1,
                 ACT_LIMIT, num_active, None, None, lo, FUSED_ROWS if hi is None else hi, mt)


def run_fused_classes(ext, b, y, out, cnt, tok_s, w_s, bufs, counts, hi):
    """exl3_moe over experts with 0 < rows <= hi, one launch per m-tile class (mul1) or one launch."""
    small = [c for c in counts if 0 < c <= hi]
    if not small:
        return
    if b.mul1:
        t1 = sum(1 for c in small if MTILE_T1 < c <= MTILE_T2)
        t2 = sum(1 for c in small if c > MTILE_T2)
        t0 = len(small) - t1 - t2
        if t2: fused(ext, b, y, out, cnt, tok_s, w_s, bufs, t2, MTILE_T2 + 1, hi, 64)
        if t1: fused(ext, b, y, out, cnt, tok_s, w_s, bufs, t1, MTILE_T1 + 1, min(MTILE_T2, hi), 32)
        if t0: fused(ext, b, y, out, cnt, tok_s, w_s, bufs, t0, 1, min(MTILE_T1, hi), 16)
    else:
        fused(ext, b, y, out, cnt, tok_s, w_s, bufs, len(small), 1, hi)


def heavy_recon(ext, b, y, out, tok_s, w_s, counts, starts, lo):
    """Original route for rows > lo: per-expert reconstruct + hgemm (exllamav3 LinearEXL3 reconstruct path)."""
    for i, c in enumerate(counts):
        if c > lo:
            s = starts[i]
            rows = tok_s[s:s + c]
            xe = y.index_select(0, rows)
            w1, w3, w2 = b.lin[b.experts[i]]
            g = w1.forward(xe, {"reconstruct": True})
            u = w3.forward(xe, {"reconstruct": True})
            a = torch.empty_like(g, dtype=torch.half)
            ext.silu_mul(g, u, a, ACT_LIMIT)
            o = w2.forward(a, {"reconstruct": True}).float()
            out.index_add_(0, rows, o * w_s[s:s + c].float().unsqueeze(1))


def _decode_group(ext, b, slots_dev, slot_bufs, G):
    w1, w3, w2 = (t[:G] for t in slot_bufs)
    for buf, (tp, sp, vp) in ((w1, b.g), (w3, b.u), (w2, b.d)):
        ext.reconstruct_had_batch(buf, tp.index_select(0, slots_dev), sp.index_select(0, slots_dev),
                                  vp.index_select(0, slots_dev), b.K, b.mcg, b.mul1)


def _gemm_group(ext, y, out, tok_s, w_s, slot_bufs, G, segs, gather_idx, offs):
    w1, w3, w2 = (t[:G] for t in slot_bufs)
    rows = tok_s.index_select(0, gather_idx)
    xg = y.index_select(0, rows)
    g = torch._grouped_mm(xg, w1, offs=offs)
    u = torch._grouped_mm(xg, w3, offs=offs)
    a = torch.empty_like(g)
    ext.silu_mul(g, u, a, ACT_LIMIT)
    o = torch._grouped_mm(a, w2, offs=offs)
    out.index_add_(0, rows, o.float() * w_s.index_select(0, gather_idx).float().unsqueeze(1))


def run_dq(ext, b, y, out, tok_s, w_s, counts, starts, sel, G=None):
    """Dequantize-once route for local slots `sel` (host list, each with counts[i] > 0)."""
    if not sel:
        return
    G = G or GROUP
    dev = y.device
    slots = _scratch(dev, G)
    side = _side(dev)
    main = torch.cuda.current_stream(dev)
    groups = [sel[i:i + G] for i in range(0, len(sel), G)]
    # host-built index tensors for all groups in one H2D copy
    meta = []
    for grp in groups:
        idx = torch.cat([torch.arange(starts[i], starts[i] + counts[i]) for i in grp])
        offs = torch.tensor(counts, dtype=torch.int32)[grp].cumsum(0, dtype=torch.int32)
        meta.append((torch.tensor(grp, dtype=torch.long), idx, offs))
    flat = torch.cat([torch.cat((m[0], m[1], m[2].long())) for m in meta]).to(dev, non_blocking=True)
    dmeta, p = [], 0
    for sl, idx, offs in meta:
        n0, n1, n2 = len(sl), len(idx), len(offs)
        dmeta.append((flat[p:p + n0], flat[p + n0:p + n0 + n1], flat[p + n0 + n1:p + n0 + n1 + n2].to(torch.int32)))
        p += n0 + n1 + n2
    ev_dec = [torch.cuda.Event() for _ in range(2)]
    ev_use = [torch.cuda.Event() for _ in range(2)]
    side.wait_stream(main)
    for gi, grp in enumerate(groups):
        s = gi & 1
        with torch.cuda.stream(side):
            if gi >= 2:
                side.wait_event(ev_use[s])            # GEMMs of group gi-2 done with this slot
            _decode_group(ext, b, dmeta[gi][0], slots[s], len(grp))
            ev_dec[s].record(side)
        main.wait_event(ev_dec[s])
        _gemm_group(ext, y, out, tok_s, w_s, slots[s], len(grp), None, dmeta[gi][1], dmeta[gi][2])
        ev_use[s].record(main)
    for t in flat, *[x for m in dmeta for x in m]:
        t.record_stream(side)


ST_HEIGHTS = (256, 128, 64, 32)


ST_H256 = int(os.environ.get("ST_EXL3_ST_H256", "192"))      # remainder rows from which a 256-row tile is used


def _st_height(rem):
    return 256 if rem >= ST_H256 else 128 if rem > 64 else 64 if rem > 32 else 32


def _st_tiles(counts, sel, bm):
    """Row tiles over the compacted rows of the selected experts: {height: (slots, row0s, rows)}."""
    out, r0 = {h: ([], [], []) for h in ST_HEIGHTS}, 0
    for i in sel:
        c, j = counts[i], 0
        while j < c:
            h = bm if bm else _st_height(c - j)
            n = min(h, c - j)
            dst = out[h]
            dst[0].append(i); dst[1].append(r0 + j); dst[2].append(n)
            j += n
        r0 += c
    return out


def _dbg(tag):
    if os.environ.get("ST_DEBUG"):
        torch.cuda.synchronize()
        print("st:", tag, flush=True)


def run_st(b, y, out, tok_s, w_s, loc_s, counts, starts, sel, bm=None):
    """Custom grouped GEMM route (st_moe_ext) for local slots `sel`, in chunks of whole experts."""
    if not sel:
        return
    stx = _stx()
    dev = y.device
    bm = ST_BM if bm is None else bm
    chunks, cur, rows = [], [], 0
    for i in sel:
        if cur and rows + counts[i] > ST_CHUNK_ROWS:
            chunks.append(cur); cur, rows = [], 0
        cur.append(i); rows += counts[i]
    if cur:
        chunks.append(cur)
    for grp in chunks:
        R = sum(counts[i] for i in grp)
        ridx = torch.cat([torch.arange(starts[i], starts[i] + counts[i]) for i in grp])
        tl = _st_tiles(counts, grp, bm)
        host = [ridx] + [torch.tensor(v, dtype=torch.long) for h in ST_HEIGHTS for v in tl[h]]
        flat = torch.cat(host).to(dev, non_blocking=True)
        parts, p = [], 0
        for h in host:
            parts.append(flat[p:p + len(h)]); p += len(h)
        ridx_d = parts[0]
        tiles = [(h, tuple(x.to(torch.int32) for x in parts[1 + 3 * k: 4 + 3 * k]))
                 for k, h in enumerate(ST_HEIGHTS) if len(tl[h][0])]
        tok_c, w_c, slot_c = tok_s[ridx_d], w_s[ridx_d], loc_s[ridx_d].to(torch.int32)
        xg = torch.empty((R, HID), dtype=torch.half, device=dev)
        xu = torch.empty((R, HID), dtype=torch.half, device=dev)
        _dbg("prep")
        stx.st_gather_had_gu(y, xg, xu, tok_c, slot_c, b.g[1], b.u[1], None)
        _dbg("gather")
        g = torch.empty((R, INTER), dtype=torch.half, device=dev)
        u = torch.empty((R, INTER), dtype=torch.half, device=dev)
        for h, (ts, tr0, trw) in tiles:
            stx.st_moe_gemm(xg, g, b.g[0], ts, tr0, trw, int(b.K), h, None)
            stx.st_moe_gemm(xu, u, b.u[0], ts, tr0, trw, int(b.K), h, None)
            _dbg(f"gemm gu {h}")
        del xg, xu
        a = torch.empty_like(g)
        stx.st_guad(g, u, a, slot_c, b.g[2], b.u[2], b.d[1], ACT_LIMIT, None)
        _dbg("guad")
        del g, u
        for h, (ts, tr0, trw) in tiles:          # down GEMM + output Hadamard * svh * weight -> out (fp32 atomics)
            stx.st_moe_gemm_down(a, out, b.d[0], b.d[2], tok_c, w_c, ts, tr0, trw, int(b.K), h, None)
            _dbg(f"gemm d {h}")


ST_GPU = os.environ.get("ST_EXL3_ST_GPU", "1") != "0"
ST_CHUNK_TOKENS = int(os.environ.get("ST_EXL3_ST_CHUNK_TOKENS", "2048"))


class LayerTables:
    """Layer-wide tables for the sync-free st route: all K buckets of one layer in one slot space."""

    def __init__(self, buckets, n_experts, dev):
        self.Ks = [int(b.K) for b in buckets]
        self.E = [b.E for b in buckets]
        self.S = sum(self.E)
        lmap = torch.full((n_experts,), self.S, dtype=torch.long)
        bucket, base = [], 0
        for i, b in enumerate(buckets):
            lmap[torch.tensor(b.experts)] = torch.arange(base, base + b.E)
            bucket += [i] * b.E
            base += b.E
        self.lmap = lmap.to(dev)
        self.bucket = torch.tensor(bucket, dtype=torch.int32, device=dev)
        cat = lambda w, i: torch.cat([getattr(b, w)[i] for b in buckets]).contiguous()
        self.g = tuple(cat("g", i) for i in range(3))
        self.u = tuple(cat("u", i) for i in range(3))
        self.d = tuple(cat("d", i) for i in range(3))


def run_st_layer(tb, y, out, flat_tok, flat_w, flat_ids, top_k):
    """st route for all K buckets of a layer with no host sync: per-expert counts, offsets and the tile lists are
    built on the GPU (st_build_tiles); kernels read the device row / tile counts and skip past them. Tokens are
    processed in chunks of ST_EXL3_ST_CHUNK_TOKENS so the scratch is bounded by chunk * top_k rows."""
    stx = _stx()
    dev = y.device
    T = flat_ids.numel() // top_k
    for t0 in range(0, T, ST_CHUNK_TOKENS):
        t1 = min(T, t0 + ST_CHUNK_TOKENS)
        lslot = tb.lmap[flat_ids[t0 * top_k: t1 * top_k]]
        Rmax = lslot.numel()
        order = lslot.argsort()
        tok_s = flat_tok[t0 * top_k: t1 * top_k][order]
        w_s = flat_w[t0 * top_k: t1 * top_k][order]
        slot_s = lslot[order].to(torch.int32)
        cnt = torch.zeros(tb.S + 1, dtype=torch.long, device=dev).scatter_add_(0, lslot, torch.ones_like(lslot))
        offs = torch.zeros(tb.S + 1, dtype=torch.int32, device=dev)
        offs[1:] = cnt[: tb.S].cumsum(0)
        nrows = offs[tb.S:]                                       # local rows (device); rows past it: other rank
        bound = max(Rmax // ST_H256 + 1, max(tb.E))
        tiles = torch.empty((len(tb.Ks), 4, 3, bound), dtype=torch.int32, device=dev)
        ntiles = torch.zeros((len(tb.Ks), 4), dtype=torch.int32, device=dev)
        stx.st_build_tiles(cnt, offs[: tb.S], tb.bucket, ST_H256, tiles, ntiles)
        views = [(K, h, tiles[i, k, 0], tiles[i, k, 1], tiles[i, k, 2], ntiles[i, k: k + 1])
                 for i, K in enumerate(tb.Ks) for k, h in enumerate(ST_HEIGHTS)]
        xg = torch.empty((Rmax, HID), dtype=torch.half, device=dev)
        xu = torch.empty((Rmax, HID), dtype=torch.half, device=dev)
        stx.st_gather_had_gu(y, xg, xu, tok_s, slot_s, tb.g[1], tb.u[1], nrows)
        g = torch.empty((Rmax, INTER), dtype=torch.half, device=dev)
        u = torch.empty((Rmax, INTER), dtype=torch.half, device=dev)
        for K, h, ts, tr0, trw, nt in views:
            stx.st_moe_gemm(xg, g, tb.g[0], ts, tr0, trw, K, h, nt)
            stx.st_moe_gemm(xu, u, tb.u[0], ts, tr0, trw, K, h, nt)
        del xg, xu
        a = torch.empty_like(g)
        stx.st_guad(g, u, a, slot_s, tb.g[2], tb.u[2], tb.d[1], ACT_LIMIT, nrows)
        del g, u
        for K, h, ts, tr0, trw, nt in views:
            stx.st_moe_gemm_down(a, out, tb.d[0], tb.d[2], tok_s, w_s, ts, tr0, trw, K, h, nt)


def run_bucket(b, y, out, flat_tok, flat_w, loc, bufs, mode=None, min_rows=None):
    """Prefill (host-synced) route for one K bucket. loc = local slot per (token, k) pair, b.E for other ranks."""
    ext = _ext()
    mode = PREFILL if mode is None else mode
    min_rows = MIN_ROWS if min_rows is None else min_rows
    order = loc.argsort()
    tok_s, w_s = flat_tok[order], flat_w[order]
    cnt = torch.bincount(loc, minlength=b.E + 1)
    counts = cnt.tolist()[: b.E]
    starts, acc = [], 0
    for c in counts:
        starts.append(acc); acc += c
    if mode == "off":
        # the original route: 16/32/64-row exl3_moe tiles up to FUSED_ROWS, reconstruct + hgemm above
        run_fused_classes(ext, b, y, out, cnt, tok_s, w_s, bufs, counts, FUSED_ROWS)
        heavy_recon(ext, b, y, out, tok_s, w_s, counts, starts, FUSED_ROWS)
        return
    if mode not in ("dq_fp16", "st"):
        raise ValueError(f"ST_EXL3_PREFILL={mode!r} (want off|st|dq_fp16)")
    # experts at or above min_rows -> dq / st; the rest stay fused. exl3_moe walks experts in slot order and takes
    # the first num_active with rows in [lo, hi], so the fused share must be the experts with rows < min_rows.
    hi = min(min_rows - 1, FUSED_ROWS)
    sel = [i for i, c in enumerate(counts) if c > hi]
    run_fused_classes(ext, b, y, out, cnt, tok_s, w_s, bufs, counts, hi)
    if mode == "st":
        run_st(b, y, out, tok_s, w_s, loc[order], counts, starts, sel)
    else:
        run_dq(ext, b, y, out, tok_s, w_s, counts, starts, sel)
