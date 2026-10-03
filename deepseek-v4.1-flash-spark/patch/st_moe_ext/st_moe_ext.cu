// SPDX-License-Identifier: MIT
// sovereign-trellis: prefill grouped GEMM for EXL3 (trellis, mul1) routed experts, decode-once per 128/256 rows.
//
// exllamav3's fused exl3_moe kernel decodes every weight fragment once per 16/32/64-row tile inside each warp, so on
// GB10 it runs at ~35 TFLOP/s at best (fp16 MMA peak ~95). Here a block owns a BM x 128 output tile (BM = 128 or
// 256 rows of ONE expert) and walks K in 32-deep stages: the 16 trellis tiles of a stage are decoded once by the
// block's 8 warps (exllamav3 dq_dispatch, MMA-fragment layout) into shared memory, then every warp runs its
// (BM/2) x 32 sub-tile of m16n8k16 MMAs against them. Decode of stage k+1 is issued in the same iteration as the
// MMAs of stage k (one barrier per stage) so ALU decode and tensor work overlap.
//
// The EXL3 Hadamard rotations stay on the activations (as in exl3_moe): gather_had_gu produces had(x * suh) per
// (row, expert) for gate and up, guad applies out-had + SiLU*up + down in-had, and the down GEMM's epilogue applies
// the down out-had, svh and router weight and adds into the fp32 token rows. Uses exllamav3 1.5.3 headers (MIT,
// turboderp): quant/exl3_dq.cuh (trellis decode), quant/hadamard_inner.cuh, ptx.cuh.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>

#include "util.h"
#include "util.cuh"
#include "ptx.cuh"
#include "quant/exl3_dq.cuh"
#include "quant/hadamard_inner.cuh"

#define ST_THREADS 256
#define ST_BN 128
#define ST_BK 32
#define ST_STAGES 4

// ---------------------------------------------------------------------------------------------------------------------
// Activation-side kernels: one warp per (row, 128-chunk), 1-D grids (the exllamav3 inner helpers index their scale
// vectors with blockIdx.y, which stays 0 here; the per-chunk offset is applied to the scale pointer instead)

__global__ void st_gather_had_gu_kernel(const half* __restrict__ x, half* __restrict__ out_g, half* __restrict__ out_u,
                                        const int64_t* __restrict__ tok, const int32_t* __restrict__ slot,
                                        const int64_t* __restrict__ suh_g, const int64_t* __restrict__ suh_u,
                                        int rows, int dim, const int32_t* __restrict__ nrows)
{
    const int chunks = dim / 128;
    if (nrows) rows = min(rows, *nrows);          // device-side row count (rows past it belong to other ranks)
    const int64_t w = (int64_t) blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (w >= (int64_t) rows * chunks) return;
    const int r = w / chunks, c = w % chunks;
    const int e = slot[r];
    const half* in = x + tok[r] * dim + c * 128;
    had_hf_r_128_inner<true, false>(in, out_g + (int64_t) r * dim + c * 128, ((const half*) suh_g[e]) + c * 128, 0.088388347648f);
    had_hf_r_128_inner<true, false>(in, out_u + (int64_t) r * dim + c * 128, ((const half*) suh_u[e]) + c * 128, 0.088388347648f);
}

__global__ void st_guad_kernel(const half* __restrict__ g, const half* __restrict__ u, half* __restrict__ out,
                               const int32_t* __restrict__ slot, const int64_t* __restrict__ svh_g,
                               const int64_t* __restrict__ svh_u, const int64_t* __restrict__ suh_d,
                               int rows, int dim, float act_limit, const int32_t* __restrict__ nrows)
{
    const int chunks = dim / 128;
    if (nrows) rows = min(rows, *nrows);
    const int64_t w = (int64_t) blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (w >= (int64_t) rows * chunks) return;
    const int r = w / chunks, c = w % chunks;
    const int e = slot[r];
    const int64_t o = (int64_t) r * dim + c * 128;
    had_hf_r_128_guad_inner(g + o, u + o, out + o, ((const half*) svh_g[e]) + c * 128, ((const half*) svh_u[e]) + c * 128,
                            ((const half*) suh_d[e]) + c * 128, 0.088388347648f, act_limit, ACT_SILU);
}

// ---------------------------------------------------------------------------------------------------------------------
// Grouped GEMM C[rows, N] = A[rows, K] @ W_e[K, N] over row tiles; W_e decoded from trellis [K/16, N/16, 16*bits]

// DOUT: down projection epilogue. Instead of storing C, each output row's 128-column chunk (= one Hadamard block) goes
// through the output Hadamard, the expert's svh signs and the router weight and is added into out[tok[row]] (fp32,
// 16-byte vector atomics): the d buffer and the separate d_out pass disappear.
template <int bits, int BM, bool DOUT>
__global__ __launch_bounds__(ST_THREADS, 1)
void st_moe_gemm_kernel(const half* __restrict__ A, half* __restrict__ C, const int64_t* __restrict__ trellis_ptrs,
                        const int32_t* __restrict__ tile_slot, const int32_t* __restrict__ tile_row0,
                        const int32_t* __restrict__ tile_rows, int size_k, int size_n,
                        float* __restrict__ out, const int64_t* __restrict__ tok, const half* __restrict__ wts,
                        const int64_t* __restrict__ svh_ptrs, const int32_t* __restrict__ ntiles)
{
    if (ntiles && (int) blockIdx.y >= *ntiles) return;   // GPU-built tile list: grid.y is an upper bound
    constexpr int TILE_U16 = 16 * bits;                  // one 16x16 trellis tile
    constexpr int MF = BM / 32;                          // m16 fragments per warp (warp tile (BM/2) x 32)
    constexpr int A_STAGE = BM * ST_BK;                  // halfs
    constexpr int T_STAGE = 2 * 8 * TILE_U16;            // uint16: k16 x2, n16 x8
    constexpr int A_INT4 = A_STAGE / 8;
    constexpr int T_INT4 = T_STAGE / 8;

    extern __shared__ int4 smem[];
    half* sA = (half*) smem;                                                 // [STAGES][BM][32] swizzled
    uint16_t* sT = (uint16_t*) (sA + ST_STAGES * A_STAGE);                   // [STAGES][2][8][TILE_U16]
    uint4* sB = (uint4*) (sT + ST_STAGES * T_STAGE);                         // [2][16 tiles][32 lanes]

    const int t = threadIdx.x, lane = t % 32, warp = t / 32;
    const int warp_m = warp / 4, warp_n = warp % 4;
    const int nt = blockIdx.x;
    const int tile = blockIdx.y;
    const int e = tile_slot[tile];
    const int row0 = tile_row0[tile];
    const int rows = tile_rows[tile];
    const uint16_t* T = (const uint16_t*) trellis_ptrs[e];
    const int blocks_n = size_n / 16;
    const int num_k = size_k / ST_BK;
    const half* Ab = A + (int64_t) row0 * size_k;

    auto load_stage = [&](int kt)
    {
        if (kt < num_k)
        {
            const int s = kt % ST_STAGES;
            int4* sa = (int4*) (sA + s * A_STAGE);
            #pragma unroll
            for (int i = 0; i < CEIL_DIVIDE(A_INT4, ST_THREADS); ++i)
            {
                int j = i * ST_THREADS + t;
                if (j < A_INT4)
                {
                    int m = j / 4, k = j % 4;                // 4 int4 per 32-half row
                    if (m < rows)
                        cp_async(sa + m * 4 + (k ^ ((m >> 1) & 3)), Ab + (int64_t) m * size_k + kt * ST_BK + k * 8);
                }
            }
            int4* st = (int4*) (sT + s * T_STAGE);
            #pragma unroll
            for (int i = 0; i < CEIL_DIVIDE(T_INT4, ST_THREADS); ++i)
            {
                int j = i * ST_THREADS + t;
                if (j < T_INT4)
                {
                    int kb = j / (T_INT4 / 2), o = j % (T_INT4 / 2);   // 8 consecutive n16 tiles per k16 row
                    const int4* g = (const int4*) (T + ((int64_t) (kt * 2 + kb) * blocks_n + nt * 8) * TILE_U16);
                    cp_async(st + j, g + o);
                }
            }
        }
        cp_async_fence();
    };

    // decode the stage's 16 tiles (2 per warp) into fragment-order shared memory
    auto decode_stage = [&](int kt)
    {
        const int s = kt % ST_STAGES;
        const uint16_t* st = sT + s * T_STAGE;
        uint4* sb = sB + (kt & 1) * 16 * 32;
        #pragma unroll
        for (int i = 0; i < 2; ++i)
        {
            const int tl = warp * 2 + i;                 // tile index kb * 8 + nb
            FragB f0, f1;
            dq_dispatch<bits, 2, false>((const uint32_t*) (st + tl * TILE_U16), lane << 3, f0, f1);
            uint4 v;
            v.x = *reinterpret_cast<uint32_t*>(&f0[0]);
            v.y = *reinterpret_cast<uint32_t*>(&f0[1]);
            v.z = *reinterpret_cast<uint32_t*>(&f1[0]);
            v.w = *reinterpret_cast<uint32_t*>(&f1[1]);
            sb[tl * 32 + lane] = v;
        }
    };

    FragC acc[MF][4];
    #pragma unroll
    for (int m = 0; m < MF; ++m)
        #pragma unroll
        for (int n = 0; n < 4; ++n) acc[m][n] = {};

    #pragma unroll
    for (int i = 0; i < ST_STAGES - 1; ++i) load_stage(i);
    cp_async_wait<ST_STAGES - 2>();
    __syncthreads();
    decode_stage(0);

    const int a_r = (lane % 8) + 8 * ((lane / 8) % 2);
    for (int kt = 0; kt < num_k; ++kt)
    {
        // stage kt + 1 landed; decode of kt visible; everyone is past the MMAs of kt - 1
        cp_async_wait<ST_STAGES - 3>();
        __syncthreads();
        load_stage(kt + ST_STAGES - 1);
        if (kt + 1 < num_k) decode_stage(kt + 1);

        const half* sa = sA + (kt % ST_STAGES) * A_STAGE;
        const uint4* sb = sB + (kt & 1) * 16 * 32;
        #pragma unroll
        for (int kb = 0; kb < 2; ++kb)
        {
            FragB fb[4];
            #pragma unroll
            for (int j = 0; j < 2; ++j)
            {
                uint4 v = sb[(kb * 8 + warp_n * 2 + j) * 32 + lane];
                *reinterpret_cast<uint32_t*>(&fb[2 * j][0]) = v.x;
                *reinterpret_cast<uint32_t*>(&fb[2 * j][1]) = v.y;
                *reinterpret_cast<uint32_t*>(&fb[2 * j + 1][0]) = v.z;
                *reinterpret_cast<uint32_t*>(&fb[2 * j + 1][1]) = v.w;
            }
            #pragma unroll
            for (int m = 0; m < MF; ++m)
            {
                const int R = warp_m * (BM / 2) + m * 16 + a_r;
                const int c = (lane / 16 + kb * 2) ^ ((R >> 1) & 3);
                FragA fa;
                ldsm4(fa, ((const int4*) sa) + R * 4 + c);
                #pragma unroll
                for (int n = 0; n < 4; ++n) ptx_mma_m16n8k16(fa, fb[n], acc[m][n]);
            }
        }
    }

    if constexpr (DOUT)
    {
        // stage the fp16 tile in shared memory (row stride 136 halfs), then one warp per row: had + scale + atomics
        constexpr int LD = ST_BN + 8;
        cp_async_wait<0>();
        __syncthreads();
        half* sC = (half*) smem;
        #pragma unroll
        for (int m = 0; m < MF; ++m)
        {
            const int r0 = warp_m * (BM / 2) + m * 16 + lane / 4;
            #pragma unroll
            for (int n = 0; n < 4; ++n)
            {
                const int col = warp_n * 32 + n * 8 + (lane % 4) * 2;
                *(half2*) (sC + r0 * LD + col) = __floats2half2_rn(acc[m][n][0], acc[m][n][1]);
                *(half2*) (sC + (r0 + 8) * LD + col) = __floats2half2_rn(acc[m][n][2], acc[m][n][3]);
            }
        }
        __syncthreads();
        const half4 s4 = ((const half4*) (((const half*) svh_ptrs[e]) + nt * ST_BN))[lane];
        const float sv0 = __low2float(s4.x), sv1 = __high2float(s4.x), sv2 = __low2float(s4.y), sv3 = __high2float(s4.y);
        for (int r = warp; r < rows; r += ST_THREADS / 32)
        {
            half4 v = *(const half4*) (sC + r * LD + lane * 4);
            float v0 = __low2float(v.x), v1 = __high2float(v.x), v2 = __low2float(v.y), v3 = __high2float(v.y);
            float a0 = v0 + v1, d0 = v0 - v1, a1 = v2 + v3, d1 = v2 - v3;
            float h0 = a0 + a1, h1 = d0 + d1, h2 = a0 - a1, h3 = d0 - d1;
            shuffle_had_f4x32(h0, h1, h2, h3, lane);
            const float sc = 0.088388347648f * __half2float(wts[row0 + r]);
            float4 o = make_float4(h0 * sc * sv0, h1 * sc * sv1, h2 * sc * sv2, h3 * sc * sv3);
            atomicAdd((float4*) (out + tok[row0 + r] * size_n + nt * ST_BN) + lane, o);
        }
        return;
    }

    // epilogue: fp16 rows < rows
    #pragma unroll
    for (int m = 0; m < MF; ++m)
    {
        const int r0 = warp_m * (BM / 2) + m * 16 + lane / 4;
        #pragma unroll
        for (int n = 0; n < 4; ++n)
        {
            const int col = nt * ST_BN + warp_n * 32 + n * 8 + (lane % 4) * 2;
            if (r0 < rows)
                *(half2*) (C + (int64_t) (row0 + r0) * size_n + col) = __floats2half2_rn(acc[m][n][0], acc[m][n][1]);
            if (r0 + 8 < rows)
                *(half2*) (C + (int64_t) (row0 + r0 + 8) * size_n + col) = __floats2half2_rn(acc[m][n][2], acc[m][n][3]);
        }
    }
}

template <int bits, int BM>
static int st_smem_bytes()
{
    const int pipe = ST_STAGES * (BM * ST_BK * 2 + 2 * 8 * 16 * bits * 2) + 2 * 16 * 32 * 16;
    const int ctile = BM * (ST_BN + 8) * 2;
    return pipe > ctile ? pipe : ctile;
}

template <int bits, int BM, bool DOUT>
static void launch_gemm(const at::Tensor& a, const at::Tensor& c, const at::Tensor& ptrs, const at::Tensor& ts,
                        const at::Tensor& tr0, const at::Tensor& trows, int size_n, float* out, const int64_t* tok,
                        const half* wts, const int64_t* svh, const int32_t* ntiles, cudaStream_t stream)
{
    auto kern = st_moe_gemm_kernel<bits, BM, DOUT>;
    const int smem = st_smem_bytes<bits, BM>();
    static bool attr = false;
    if (!attr)
    {
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        attr = true;
    }
    const int size_k = a.size(1);
    dim3 grid(size_n / ST_BN, ts.size(0));
    kern<<<grid, ST_THREADS, smem, stream>>>((const half*) a.data_ptr(), DOUT ? nullptr : (half*) c.data_ptr(),
        (const int64_t*) ptrs.data_ptr(), (const int32_t*) ts.data_ptr(), (const int32_t*) tr0.data_ptr(),
        (const int32_t*) trows.data_ptr(), size_k, size_n, out, tok, wts, svh, ntiles);
}

#define ST_DISPATCH(DOUT_) \
    ST_CASE(1, 32, DOUT_) ST_CASE(2, 32, DOUT_) ST_CASE(3, 32, DOUT_) ST_CASE(4, 32, DOUT_) ST_CASE(5, 32, DOUT_) ST_CASE(6, 32, DOUT_) ST_CASE(7, 32, DOUT_) ST_CASE(8, 32, DOUT_) \
    ST_CASE(1, 64, DOUT_) ST_CASE(2, 64, DOUT_) ST_CASE(3, 64, DOUT_) ST_CASE(4, 64, DOUT_) ST_CASE(5, 64, DOUT_) ST_CASE(6, 64, DOUT_) ST_CASE(7, 64, DOUT_) ST_CASE(8, 64, DOUT_) \
    ST_CASE(1, 128, DOUT_) ST_CASE(2, 128, DOUT_) ST_CASE(3, 128, DOUT_) ST_CASE(4, 128, DOUT_) ST_CASE(5, 128, DOUT_) ST_CASE(6, 128, DOUT_) ST_CASE(7, 128, DOUT_) ST_CASE(8, 128, DOUT_) \
    ST_CASE(1, 256, DOUT_) ST_CASE(2, 256, DOUT_) ST_CASE(3, 256, DOUT_) ST_CASE(4, 256, DOUT_) ST_CASE(5, 256, DOUT_) ST_CASE(6, 256, DOUT_) ST_CASE(7, 256, DOUT_) ST_CASE(8, 256, DOUT_)

static void check_tiles(const at::Tensor& a, const at::Tensor& ptrs, const at::Tensor& ts, const at::Tensor& tr0,
                        const at::Tensor& trows, int64_t size_n)
{
    TORCH_CHECK(a.dtype() == at::kHalf && a.is_contiguous(), "st_moe_gemm: contiguous fp16 a");
    TORCH_CHECK(a.size(1) % ST_BK == 0 && size_n % ST_BN == 0, "st_moe_gemm: K % 32, N % 128");
    TORCH_CHECK(ts.dtype() == at::kInt && tr0.dtype() == at::kInt && trows.dtype() == at::kInt, "int32 tiles");
    TORCH_CHECK(ptrs.dtype() == at::kLong, "int64 pointer tables");
}

// a: [R, K] fp16 (rows of all tiles, expert-contiguous), c: [R, N] fp16, trellis_ptrs: [E] int64,
// tile_slot / tile_row0 / tile_rows: [num_tiles] int32 (each tile <= bm rows of one expert)
static const int32_t* opt_i32(const c10::optional<at::Tensor>& t)
{
    if (!t.has_value()) return nullptr;
    TORCH_CHECK(t->dtype() == at::kInt, "st_moe_ext: device counts must be int32");
    return (const int32_t*) t->data_ptr();
}

// ntiles (optional, int32 [1] on device): number of valid tiles; grid.y = tile_slot.size(0) is then an upper bound
void st_moe_gemm(at::Tensor a, at::Tensor c, at::Tensor trellis_ptrs, at::Tensor tile_slot, at::Tensor tile_row0,
                 at::Tensor tile_rows, int64_t bits, int64_t bm, c10::optional<at::Tensor> ntiles)
{
    const at::cuda::OptionalCUDAGuard guard(a.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    check_tiles(a, trellis_ptrs, tile_slot, tile_row0, tile_rows, c.size(1));
    TORCH_CHECK(c.dtype() == at::kHalf && c.is_contiguous(), "st_moe_gemm: contiguous fp16 c");
    if (tile_slot.size(0) == 0) return;
    const int size_n = c.size(1);
    #define ST_CASE(B_, M_, D_) if (bits == B_ && bm == M_) { launch_gemm<B_, M_, D_>(a, c, trellis_ptrs, tile_slot, tile_row0, tile_rows, size_n, nullptr, nullptr, nullptr, nullptr, opt_i32(ntiles), stream); return; }
    ST_DISPATCH(false)
    #undef ST_CASE
    TORCH_CHECK(false, "st_moe_gemm: unsupported bits/bm ", bits, "/", bm);
}

// down projection with fused output Hadamard * svh * router weight, atomically added into out[tok] (fp32 [T, N])
void st_moe_gemm_down(at::Tensor a, at::Tensor out, at::Tensor trellis_ptrs, at::Tensor svh_ptrs, at::Tensor tok,
                      at::Tensor wts, at::Tensor tile_slot, at::Tensor tile_row0, at::Tensor tile_rows, int64_t bits,
                      int64_t bm, c10::optional<at::Tensor> ntiles)
{
    const at::cuda::OptionalCUDAGuard guard(a.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    check_tiles(a, trellis_ptrs, tile_slot, tile_row0, tile_rows, out.size(1));
    TORCH_CHECK(out.dtype() == at::kFloat && out.is_contiguous(), "st_moe_gemm_down: contiguous fp32 out");
    TORCH_CHECK(wts.dtype() == at::kHalf && tok.dtype() == at::kLong && svh_ptrs.dtype() == at::kLong, "st_moe_gemm_down: dtypes");
    if (tile_slot.size(0) == 0) return;
    const int size_n = out.size(1);
    float* o = (float*) out.data_ptr();
    const int64_t* tk = (const int64_t*) tok.data_ptr();
    const half* w = (const half*) wts.data_ptr();
    const int64_t* sv = (const int64_t*) svh_ptrs.data_ptr();
    #define ST_CASE(B_, M_, D_) if (bits == B_ && bm == M_) { launch_gemm<B_, M_, D_>(a, out, trellis_ptrs, tile_slot, tile_row0, tile_rows, size_n, o, tk, w, sv, opt_i32(ntiles), stream); return; }
    ST_DISPATCH(true)
    #undef ST_CASE
    TORCH_CHECK(false, "st_moe_gemm_down: unsupported bits/bm ", bits, "/", bm);
}

static int warp_grid(int64_t warps) { return (int) CEIL_DIVIDE(warps, 8); }

void st_gather_had_gu(at::Tensor x, at::Tensor out_g, at::Tensor out_u, at::Tensor tok, at::Tensor slot,
                      at::Tensor suh_g, at::Tensor suh_u, c10::optional<at::Tensor> nrows)
{
    const at::cuda::OptionalCUDAGuard guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const int rows = out_g.size(0), dim = x.size(1);
    if (!rows) return;
    st_gather_had_gu_kernel<<<warp_grid((int64_t) rows * (dim / 128)), 256, 0, stream>>>(
        (const half*) x.data_ptr(), (half*) out_g.data_ptr(), (half*) out_u.data_ptr(), (const int64_t*) tok.data_ptr(),
        (const int32_t*) slot.data_ptr(), (const int64_t*) suh_g.data_ptr(), (const int64_t*) suh_u.data_ptr(), rows, dim, opt_i32(nrows));
}

void st_guad(at::Tensor g, at::Tensor u, at::Tensor out, at::Tensor slot, at::Tensor svh_g, at::Tensor svh_u,
             at::Tensor suh_d, double act_limit, c10::optional<at::Tensor> nrows)
{
    const at::cuda::OptionalCUDAGuard guard(g.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const int rows = g.size(0), dim = g.size(1);
    if (!rows) return;
    st_guad_kernel<<<warp_grid((int64_t) rows * (dim / 128)), 256, 0, stream>>>(
        (const half*) g.data_ptr(), (const half*) u.data_ptr(), (half*) out.data_ptr(), (const int32_t*) slot.data_ptr(),
        (const int64_t*) svh_g.data_ptr(), (const int64_t*) svh_u.data_ptr(), (const int64_t*) suh_d.data_ptr(),
        rows, dim, (float) act_limit, opt_i32(nrows));
}

// GPU tile scheduling (no host sync): one thread per local expert slot splits its rows [offs[s], offs[s] + cnt[s])
// into 256 / 128 / 64 / 32-row tiles (same policy as st_exl3_prefill._st_height) and appends them to the list of its
// (bucket, height): tiles [nb][4][3][bound] int32 = (slot, row0, rows), ntiles [nb][4] int32 (zeroed by the caller).
__global__ void st_build_tiles_kernel(const int64_t* __restrict__ cnt, const int32_t* __restrict__ offs,
                                      const int32_t* __restrict__ bucket, int nslots, int h256, int bound,
                                      int32_t* __restrict__ tiles, int32_t* __restrict__ ntiles)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nslots) return;
    const int c = (int) cnt[s], o = offs[s], b = bucket[s];
    for (int j = 0; j < c;)
    {
        const int rem = c - j;
        const int hi = rem >= h256 ? 0 : rem > 64 ? 1 : rem > 32 ? 2 : 3;      // 256, 128, 64, 32
        const int h = 256 >> hi;
        const int n = min(h, rem);
        const int k = atomicAdd(ntiles + b * 4 + hi, 1);
        int32_t* t = tiles + ((int64_t) (b * 4 + hi) * 3) * bound;
        t[k] = s; t[bound + k] = o + j; t[2 * bound + k] = n;
        j += n;
    }
}

void st_build_tiles(at::Tensor cnt, at::Tensor offs, at::Tensor bucket, int64_t h256, at::Tensor tiles, at::Tensor ntiles)
{
    const at::cuda::OptionalCUDAGuard guard(cnt.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(cnt.dtype() == at::kLong && offs.dtype() == at::kInt && bucket.dtype() == at::kInt, "st_build_tiles dtypes");
    TORCH_CHECK(tiles.dtype() == at::kInt && tiles.dim() == 4 && tiles.size(1) == 4 && tiles.size(2) == 3, "tiles [nb,4,3,bound]");
    const int nslots = bucket.size(0);
    st_build_tiles_kernel<<<CEIL_DIVIDE(nslots, 128), 128, 0, stream>>>((const int64_t*) cnt.data_ptr(),
        (const int32_t*) offs.data_ptr(), (const int32_t*) bucket.data_ptr(), nslots, (int) h256, (int) tiles.size(3),
        (int32_t*) tiles.data_ptr(), (int32_t*) ntiles.data_ptr());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("st_moe_gemm", &st_moe_gemm, "EXL3 mul1 grouped GEMM, decode once per BM rows");
    m.def("st_moe_gemm_down", &st_moe_gemm_down, "grouped GEMM + output Hadamard * svh * weight, atomic into fp32 out");
    m.def("st_gather_had_gu", &st_gather_had_gu, "gather token rows + input Hadamard for gate and up");
    m.def("st_guad", &st_guad, "gate/up output Hadamard + SiLU*up + down input Hadamard");
    m.def("st_build_tiles", &st_build_tiles, "GPU tile list per (bucket, height) from per-slot counts/offsets");
}
