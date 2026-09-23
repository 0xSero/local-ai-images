// aikido-exl3: original-basis expert weights written STRAIGHT into vLLM's fused-MoE layout.
//
//   W = diag(suh) . H128 . W_hat . H128 . diag(svh)      (ExLlamaV3's own >= 1024-row arithmetic)
//
// `aikido_orig_batch_kernel` below is ExLlamaV3's `reconstruct_had_tile` + `reconstruct_had_batch_kernel` (exllamav3_ext/quant/reconstruct.cu, MIT, Copyright (c)
// 2025 Turboderp), K = 4, statement for statement - the same shared-memory tile, the same fp16 butterfly with the
// pre-applied 1/sqrt(128), the same fp32 first stage of the second transform, the same three fp16 multiplies (r_scale, suh,
// svh) in the same order - with these changes, all marked AIKIDO, none of which touches a computed value:
//   1. the K = 4 window decode is our `aikido_exl3::dq8_regs_4bits(prev_word, own_word)` (exl3_decode.cuh), the same
//      instructions as ExLlamaV3's `dq8_aligned_4bits(ptr, lane * 8)`: a = ptr[(lane + 31) & 31], b = ptr[lane];
//      the K = 3 decode (GLM-5.3 TR3: 192 of 256 experts per layer) is our `dq8_regs_3bits(a, b, s2)` with the lane's
//      two source words from `k3_lane_words` = ExLlamaV3's `dq8<3, cb, 4>(ptr, lane * 8)` (exl3_dq.cuh): a = ptr[i0 % 24],
//      b = ptr[i2 % 24], the same funnel shifts, the same decode; a K = 3 tile is 24 words staged byte for byte;
//   2. the final STORE: ExLlamaV3 writes lane l's four values to row (kb*128 + R), columns nb*128 + 4l .. 4l+3 of a
//      [k, n] matrix. vLLM's fused MoE wants [E, out, in] = the transpose, with gate and up stacked on the out dimension,
//      so the same four values go to rows row_offset + nb*128 + 4l + i, column kb*128 + R of a [rows_total, k] matrix.
//   3. (STACKED variant) the trellis is read from the grouped kernel's resident pack [E, k/16, n/64, 32 lanes, 4 tiles]
//      (hopper_moe.stack_repacked: a pure permutation of the stream words, word l of tile (i, 4g + j) at [i, g, l, j];
//      K = 3: [E, k/16, n/64, 4 tiles, 24 words], the tiles exactly as stored) instead of per-expert int16 tensors behind
//      pointer tables, and suh / svh from the stacked [E, ...] tensors. Only the shared-memory load and the two word reads
//      of the decode change their INDICES. So the transient tier needs no second copy of the experts: its only extra
//      memory is the arena. An optional `out_ids` table (int32, local expert -> output expert) lets one K class of a
//      mixed-K layer (GLM-5.3 TR3) write its experts into their GLOBAL rows of a full arena, so vLLM's fused MoE runs
//      once over all experts with the unmapped router ids.
// No value is computed differently, so the result equals `exllamav3_ext.reconstruct_had_batch(...).transpose(1, 2)` BIT FOR
// BIT (parity/original_basis_parity.py), and the 5.4 ms per layer transposing copy of the first implementation disappears:
// the tier can be rebuilt per prefill chunk into a fixed arena instead of keeping 1.5 GiB per layer resident.

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <optional>

#include "exl3_decode.cuh"
#include "third_party/exllamav3/hadamard_inner.cuh"

namespace aikido_orig {

template <typename T, int n>
struct Vec {
  T elems[n];
  __device__ T& operator[](int i) { return elems[i]; }
};
using FragB = Vec<half2, 2>;

#define RH_THREADS 256

// STACKED = false: packed_ptrs / suh_ptrs / svh_ptrs are per-expert pointer tables (ExLlamaV3's calling convention).
// STACKED = true: packed_ptrs is really `const int32_t*` = the pack of ALL experts, expert e at e * pack_stride ints, this
// matrix's column groups start at group_offset (gate | up are concatenated along n/64); suh / svh are base pointers of
// stacked tensors, expert e at e * s?h_stride + s?h_off halfs.
template <int cb, bool STACKED, int K>
__global__ __launch_bounds__(RH_THREADS)
void aikido_orig_batch_kernel(half* __restrict__ g_out, const void* __restrict__ packed_ptrs,
                              const void* __restrict__ suh_ptrs, const void* __restrict__ svh_ptrs,
                              int packed_blocks_n, size_t out_stride, int row_offset, int k_len,
                              size_t pack_stride, int groups_total, int group_offset, size_t suh_stride, size_t suh_off,
                              size_t svh_stride, size_t svh_off, const int32_t* __restrict__ out_ids) {
  static_assert(K == 3 || K == 4, "K = 3 and K = 4 only");
  constexpr int packed_size = 16 * K;                 // uint16 per tile: 48 (K = 3) / 64 (K = 4)
  constexpr int words_per_group = 32 * K;             // int32 per 64-column group of the STACKED pack: 96 / 128
  constexpr float r_scale = 0.08838834764831845f;

  const int bz = blockIdx.z;
  const uint16_t* g_packed = nullptr;
  const int32_t* g_pack = nullptr;
  const half* suh;
  const half* svh;
  if constexpr (STACKED) {
    g_pack = (const int32_t*)packed_ptrs + (size_t)bz * pack_stride;
    suh = (const half*)suh_ptrs + (size_t)bz * suh_stride + suh_off;
    svh = (const half*)svh_ptrs + (size_t)bz * svh_stride + svh_off;
  } else {
    g_packed = ((const uint16_t* const*)packed_ptrs)[bz];
    suh = ((const half* const*)suh_ptrs)[bz];
    svh = ((const half* const*)svh_ptrs)[bz];
  }
  // AIKIDO (3): this expert's rows of the output (out_ids: local -> global expert of a full mixed-K arena)
  half* g_unpacked = g_out + (size_t)(out_ids ? out_ids[bz] : bz) * out_stride;

  int t = threadIdx.x;
  int lane_id = t % 32;
  int warp_id = t / 32;
  int kb = blockIdx.y;
  int nb = blockIdx.x;
  int n = nb * 8;

  __shared__ uint32_t s_packed[8][8][packed_size / 2];
  __shared__ half2 stile[128 * 64];

  auto tix = [&](int R, int q, int p) { return R * 64 + (q ^ ((R >> 2) & 31)) * 2 + p; };

  constexpr int j_int4 = packed_size / 8;
  for (int u = t; u < 8 * 8 * j_int4; u += RH_THREADS) {
    int j = u / (8 * j_int4);
    int r = u % (8 * j_int4);
    if constexpr (STACKED) {
      // AIKIDO (3): k-tile row (kb*8 + j) of the pack: 2 column groups of this 128-column block x 32 lanes x 4 tiles.
      // int4 r = (group r / 32, lane r % 32) holds that lane's stream word for the 4 tiles of the group; it lands at
      // s_packed[j] int index 4r, i.e. word of (group g2, lane l, tile jj) at s_packed[j] flat index (g2*32 + l)*4 + jj.
      // K = 3: the group is 4 tiles x 24 words as stored, so the two groups of this block land as s_packed[j][wn][24],
      // exactly ExLlamaV3's staging of 8 consecutive tiles.
      const int32_t* gp = g_pack + ((size_t)(kb * 8 + j) * groups_total + group_offset + nb * 2) * words_per_group;
      ((int4*)s_packed[j])[r] = ((const int4*)gp)[r];
    } else {
      const uint16_t* gp = g_packed + ((size_t)((kb * 8 + j) * packed_blocks_n + n)) * packed_size;
      ((int4*)s_packed[j])[r] = ((const int4*)gp)[r];
    }
  }
  __syncthreads();

  for (int jj = 0; jj < 8 * 8 / (RH_THREADS / 32); ++jj) {
    int j = (warp_id / 8) * (8 / (RH_THREADS / 256)) + jj;
    int wn = warp_id % 8;
    register FragB frag[2];
    // AIKIDO (1): same decode, our entry points. ExLlamaV3: dq_dispatch<K, cb>(s_packed[j][wn], lane_id * 8, ...)
    if constexpr (K == 3) {
      int src_a, src_b, s2;
      aikido_exl3::k3_lane_words(lane_id, src_a, src_b, s2);
      const uint32_t* ptr = s_packed[j][wn];
      aikido_exl3::dq8_regs_3bits<FragB, cb>(ptr[src_a], ptr[src_b], s2, frag[0], frag[1]);
    } else if constexpr (STACKED) {
      const uint32_t* flat = (const uint32_t*)s_packed[j];     // [(g2 * 32 + lane) * 4 + jj], tile wn = g2 * 4 + jj
      const int g2 = wn / 4, tj = wn % 4;
      aikido_exl3::dq8_regs_4bits<FragB, cb>(flat[(g2 * 32 + ((lane_id + 31) & 31)) * 4 + tj], flat[(g2 * 32 + lane_id) * 4 + tj],
                                             frag[0], frag[1]);
    } else {
      const uint32_t* ptr = s_packed[j][wn];
      aikido_exl3::dq8_regs_4bits<FragB, cb>(ptr[(lane_id + 31) & 31], ptr[lane_id], frag[0], frag[1]);
    }

    half2 n0 = __shfl_down_sync(0xFFFFFFFF, frag[0][0], 4, 32);
    half2 n1 = __shfl_down_sync(0xFFFFFFFF, frag[0][1], 4, 32);
    half2 n2 = __shfl_down_sync(0xFFFFFFFF, frag[1][0], 4, 32);
    half2 n3 = __shfl_down_sync(0xFFFFFFFF, frag[1][1], 4, 32);

    if (!(lane_id & 4)) {
      half2 m0 = __halves2half2(__low2half(frag[0][0]), __low2half(n0));
      half2 m1 = __halves2half2(__high2half(frag[0][0]), __high2half(n0));
      half2 m2 = __halves2half2(__low2half(frag[0][1]), __low2half(n1));
      half2 m3 = __halves2half2(__high2half(frag[0][1]), __high2half(n1));
      half2 m4 = __halves2half2(__low2half(frag[1][0]), __low2half(n2));
      half2 m5 = __halves2half2(__high2half(frag[1][0]), __high2half(n2));
      half2 m6 = __halves2half2(__low2half(frag[1][1]), __low2half(n3));
      half2 m7 = __halves2half2(__high2half(frag[1][1]), __high2half(n3));
      int r0 = j * 16 + (lane_id % 4) * 2;
      int r1 = r0 + 1;
      int r2 = r0 + 8;
      int r3 = r0 + 9;
      int c0 = lane_id / 8;
      int q0 = (wn * 8 + c0) >> 1, p0 = c0 & 1;
      int q1 = (wn * 8 + c0 + 4) >> 1, p1 = c0 & 1;
      stile[tix(r0, q0, p0)] = m0;
      stile[tix(r1, q0, p0)] = m1;
      stile[tix(r2, q0, p0)] = m2;
      stile[tix(r3, q0, p0)] = m3;
      stile[tix(r0, q1, p1)] = m4;
      stile[tix(r1, q1, p1)] = m5;
      stile[tix(r2, q1, p1)] = m6;
      stile[tix(r3, q1, p1)] = m7;
    }
  }
  __syncthreads();

  const half2 rs2 = __float2half2_rn(r_scale);
  constexpr int CHUNKS_PW = 32 / (RH_THREADS / 32);
  #pragma unroll
  for (int qq = 0; qq < CHUNKS_PW; ++qq) {
    int q = warp_id * CHUNKS_PW + qq;
    int qs = q ^ lane_id;
    half2 a[4], b[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      half4 v = *((const half4*)(stile + (lane_id * 4 + i) * 64 + qs * 2));
      a[i] = v.x;
      b[i] = v.y;
    }
    #pragma unroll
    for (int x = 0; x < 2; ++x) {
      half2* v = x == 0 ? a : b;
      half2 s0 = __hadd2(v[0], v[1]), d0 = __hsub2(v[0], v[1]);
      half2 s1 = __hadd2(v[2], v[3]), d1 = __hsub2(v[2], v[3]);
      v[0] = __hmul2(__hadd2(s0, s1), rs2);
      v[1] = __hmul2(__hadd2(d0, d1), rs2);
      v[2] = __hmul2(__hsub2(s0, s1), rs2);
      v[3] = __hmul2(__hsub2(d0, d1), rs2);
      #pragma unroll
      for (int i = 0; i < 4; ++i) v[i] = shuffle_had_h2x32(v[i], lane_id);
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      half4 v;
      v.x = a[i];
      v.y = b[i];
      *((half4*)(stile + (lane_id * 4 + i) * 64 + qs * 2)) = v;
    }
  }
  __syncthreads();

  constexpr int ROWS_PW = 128 / (RH_THREADS / 32);
  #pragma unroll
  for (int rr = 0; rr < ROWS_PW; ++rr) {
    int R = warp_id * ROWS_PW + rr;
    int base = R * 64 + (lane_id ^ ((R >> 2) & 31)) * 2;
    half2 v01 = stile[base];
    half2 v23 = stile[base + 1];
    float v0 = __low2float(v01), v1 = __high2float(v01);
    float v2 = __low2float(v23), v3 = __high2float(v23);
    float s0 = v0 + v1, d0 = v0 - v1;
    float s1 = v2 + v3, d1 = v2 - v3;
    half2 h01 = __hmul2(__floats2half2_rn(s0 + s1, d0 + d1), rs2);
    half2 h23 = __hmul2(__floats2half2_rn(s0 - s1, d0 - d1), rs2);
    h01 = shuffle_had_h2x32(h01, lane_id);
    h23 = shuffle_had_h2x32(h23, lane_id);
    // AIKIDO (2): ExLlamaV3 scales here (o = (h * suh[kb*128 + R]) * svh[nb*128 + 4*lane + j]) and stores one half4 at
    // [kb*128 + R][nb*128 + 4*lane .. +3] of a [k, n] matrix. We keep the transformed tile in shared memory (the slots
    // this lane just read) and scale + store it transposed below, 16 bytes per store, instead of four 2-byte stores per
    // lane 4 rows apart (k_len * 2 bytes each): same two fp16 multiplies per element in the same order.
    stile[base] = h01;
    stile[base + 1] = h23;
  }
  __syncthreads();

  // Transposed, vectorised store: thread t owns output row (row_offset + nb*128 + c), c = t % 128, and the 64 columns
  // kb*128 + k0 .. + 63, k0 = (t / 128) * 64 (a warp reads 32 consecutive c of one row R: 16 distinct words, no bank
  // conflicts); element (R, c) of the tile sits at stile[tix(R, c / 4, (c / 2) & 1)] half (c & 1). Pairs of rows
  // (R, R + 1) form one half2 so the scaling is the same HMUL2 arithmetic elementwise: (h * su[R]) * sv[c].
  {
    const int c = t & 127, k0 = (t >> 7) * 64;
    const int q = c >> 2, p = (c >> 1) & 1, e = c & 1;
    const half2 sv2 = __half2half2(svh[nb * 128 + c]);
    half* dst = g_unpacked + (size_t)(row_offset + nb * 128 + c) * k_len + (kb * 128 + k0);
    #pragma unroll 2
    for (int r8 = 0; r8 < 64; r8 += 8) {
      half2 o[4];
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int R = k0 + r8 + 2 * i;
        const half2 a = stile[tix(R, q, p)], b = stile[tix(R + 1, q, p)];
        const half2 h2 = e ? __halves2half2(__high2half(a), __high2half(b)) : __halves2half2(__low2half(a), __low2half(b));
        const half2 su2 = __halves2half2(suh[kb * 128 + R], suh[kb * 128 + R + 1]);
        o[i] = __hmul2(__hmul2(h2, su2), sv2);
      }
      *((int4*)(dst + r8)) = *((const int4*)o);
    }
  }
}

}  // namespace aikido_orig

// out: fp16 [E, rows_total, k] contiguous (vLLM [E, out, in]); this call fills rows row_offset .. row_offset + n - 1 of every
// expert with the n x k transpose of W. Pointer tables: int64 device tensors (trellis int16 [k/16, n/16, 64], suh fp16 [k],
// svh fp16 [n] per expert), as exllamav3_ext.reconstruct_had_batch takes them. bits = K (3 or 4); cb 1 = MCG, 2 = MUL1.
void moe_build_orig_vllm(at::Tensor& out, const at::Tensor& trellis_ptrs, const at::Tensor& suh_ptrs,
                         const at::Tensor& svh_ptrs, int64_t n, int64_t row_offset, int64_t cb, int64_t bits) {
  const at::cuda::OptionalCUDAGuard device_guard(out.device());
  TORCH_CHECK(out.dim() == 3 && out.dtype() == at::kHalf && out.is_contiguous(), "out must be contiguous fp16 [E, rows, k]");
  const int64_t e = out.size(0), rows_total = out.size(1), k = out.size(2);
  TORCH_CHECK(k % 128 == 0 && n % 128 == 0 && row_offset % 128 == 0 && row_offset >= 0 && row_offset + n <= rows_total,
              "k, n, row_offset must be multiples of 128 and fit the output");
  for (const at::Tensor* p : {&trellis_ptrs, &suh_ptrs, &svh_ptrs})
    TORCH_CHECK(p->dtype() == at::kLong && p->is_contiguous() && p->device() == out.device() && p->numel() >= e,
                "pointer tables must be contiguous int64 tensors on the output device with one entry per expert");
  if (e == 0) return;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  dim3 grid((unsigned)(n / 128), (unsigned)(k / 128), (unsigned)e);
  auto launch = [&](auto kernel) {
    kernel<<<grid, RH_THREADS, 0, stream>>>((half*)out.data_ptr(), (const void*)trellis_ptrs.data_ptr(),
                                            (const void*)suh_ptrs.data_ptr(), (const void*)svh_ptrs.data_ptr(),
                                            (int)(n / 16), (size_t)rows_total * k, (int)row_offset, (int)k,
                                            (size_t)0, 0, 0, (size_t)0, (size_t)0, (size_t)0, (size_t)0, (const int32_t*)nullptr);
  };
  if (cb == 1 && bits == 4) launch(aikido_orig::aikido_orig_batch_kernel<1, false, 4>);
  else if (cb == 2 && bits == 4) launch(aikido_orig::aikido_orig_batch_kernel<2, false, 4>);
  else if (cb == 1 && bits == 3) launch(aikido_orig::aikido_orig_batch_kernel<1, false, 3>);
  else if (cb == 2 && bits == 3) launch(aikido_orig::aikido_orig_batch_kernel<2, false, 3>);
  else TORCH_CHECK(false, "original-basis builder: codebook ", cb, " / K ", bits, " not built (cb 1 = MCG, 2 = MUL1; K 3, 4)");
}

// STACKED variant: reads the grouped kernel's resident pack. pack: int32 [E, k/16, groups_total, 32, 4] for K = 4 or
// [E, k/16, groups_total, 4, 24] for K = 3 (n_total = 64 * groups_total columns; this matrix = columns col_offset ..
// col_offset + n - 1 of it); suh: fp16 [E, ..., k] with this matrix's vector at flat offset suh_off inside an expert's
// slab; svh: fp16 [E, n_total'] with offset svh_off. out_ids (optional int32 [E_pack]): output expert of each pack expert;
// without it expert e of the pack writes expert e of out, and out must hold exactly the pack's experts.
void moe_build_orig_vllm_stacked(at::Tensor& out, const at::Tensor& pack, const at::Tensor& suh, const at::Tensor& svh,
                                 int64_t n, int64_t col_offset, int64_t suh_off, int64_t svh_off, int64_t row_offset, int64_t cb,
                                 const std::optional<at::Tensor>& out_ids) {
  const at::cuda::OptionalCUDAGuard device_guard(out.device());
  TORCH_CHECK(out.dim() == 3 && out.dtype() == at::kHalf && out.is_contiguous(), "out must be contiguous fp16 [E, rows, k]");
  const int64_t rows_total = out.size(1), k = out.size(2);
  TORCH_CHECK(pack.dim() == 5 && pack.dtype() == at::kInt && pack.is_contiguous() && pack.size(1) * 16 == k &&
              ((pack.size(3) == 32 && pack.size(4) == 4) || (pack.size(3) == 4 && pack.size(4) == 24)),
              "pack must be contiguous int32 [E, k/16, n_total/64, 32, 4] (K = 4) or [E, k/16, n_total/64, 4, 24] (K = 3)");
  const int64_t e = pack.size(0), bits = pack.size(4) == 24 ? 3 : 4;
  const int32_t* ids = nullptr;
  if (out_ids.has_value() && out_ids->defined()) {
    TORCH_CHECK(out_ids->dtype() == at::kInt && out_ids->is_contiguous() && out_ids->device() == out.device() && out_ids->numel() == e,
                "out_ids must be a contiguous int32 tensor on the output device with one entry per pack expert");
    ids = out_ids->data_ptr<int32_t>();
  } else {
    TORCH_CHECK(out.size(0) == e, "out must hold exactly the pack's experts when no out_ids table is given");
  }
  const int64_t groups_total = pack.size(2);
  TORCH_CHECK(k % 128 == 0 && n % 128 == 0 && col_offset % 128 == 0 && col_offset + n <= groups_total * 64 && row_offset % 128 == 0 &&
              row_offset >= 0 && row_offset + n <= rows_total, "k, n, offsets must be multiples of 128 and fit");
  TORCH_CHECK(suh.dtype() == at::kHalf && svh.dtype() == at::kHalf && suh.is_contiguous() && svh.is_contiguous() &&
              suh.size(0) == e && svh.size(0) == e, "suh / svh must be contiguous fp16 stacked per expert");
  const int64_t suh_stride = suh.numel() / e, svh_stride = svh.numel() / e;
  TORCH_CHECK(suh_off >= 0 && suh_off + k <= suh_stride && svh_off >= 0 && svh_off + n <= svh_stride, "suh / svh offsets out of range");
  if (e == 0) return;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  dim3 grid((unsigned)(n / 128), (unsigned)(k / 128), (unsigned)e);
  auto launch = [&](auto kernel) {
    kernel<<<grid, RH_THREADS, 0, stream>>>((half*)out.data_ptr(), (const void*)pack.data_ptr(), (const void*)suh.data_ptr(),
                                            (const void*)svh.data_ptr(), 0, (size_t)rows_total * k, (int)row_offset, (int)k,
                                            (size_t)(pack.numel() / e), (int)groups_total, (int)(col_offset / 64),
                                            (size_t)suh_stride, (size_t)suh_off, (size_t)svh_stride, (size_t)svh_off, ids);
  };
  if (cb == 1 && bits == 4) launch(aikido_orig::aikido_orig_batch_kernel<1, true, 4>);
  else if (cb == 2 && bits == 4) launch(aikido_orig::aikido_orig_batch_kernel<2, true, 4>);
  else if (cb == 1 && bits == 3) launch(aikido_orig::aikido_orig_batch_kernel<1, true, 3>);
  else if (cb == 2 && bits == 3) launch(aikido_orig::aikido_orig_batch_kernel<2, true, 3>);
  else TORCH_CHECK(false, "original-basis builder: codebook ", cb, " / K ", bits, " not built (cb 1 = MCG, 2 = MUL1; K 3, 4)");
}
