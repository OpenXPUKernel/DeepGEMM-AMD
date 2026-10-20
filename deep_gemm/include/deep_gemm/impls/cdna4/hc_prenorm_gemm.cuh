// CDNA4 hyper-connection prenorm GEMM: D = A @ B^T for BF16 A [M, K] and FP32 B [N, K], and
// sqr_sum = the sum of squares of A's rows, over K splits.
//
// N is small (the 24 mix outputs), so streaming A bounds the time. CDNA4 has no TF32 MFMA: B is
// split into BF16 hi = bf16(b) and lo = bf16(b - hi) in registers, and both run BF16 MFMAs
// against the exact BF16 A (16 mantissa bits of B, more than TF32's 10). A workgroup takes BM
// rows and every column; its four waves take quarters of the K range of its split and add their
// tiles in LDS in wave order. MFMA q of a step takes K = 32 q .. + 31 (lane group g: 32 q + 8 g
// .. + 7), so one load instruction reads 64 adjacent bytes of a row of A.
#pragma once
#include <deep_gemm/impls/cdna4/device_common.h>

using deep_gemm::cdna4::HcPrenormGemmArgs;

// FM fragments of 16 rows and NF fragments of 16 columns per workgroup
template <int FM_, int NF_> struct HcPrenormGemmConfig {
  static constexpr int FM = FM_, NF = NF_, WAVES = 4, NT = 64 * WAVES, BM = 16 * FM, BN = 16 * NF;
  static constexpr int BK = 128; // K values per step: four 16x16x32 MFMAs
  // LDS: per wave its D tile and its rows' sums of squares
  static constexpr int TILE_FLOATS = BM * BN + BM;
  static constexpr int LDS_BYTES = WAVES * TILE_FLOATS * 4;
};

// keep the first n (may be <= 0 or >= 8) BF16 values of 8
__device__ __forceinline__ v4i keep_first_bf16(v4i v, int n) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int keep = min(max(n - 2 * i, 0), 2);
    v[i] &= keep == 2 ? -1 : keep == 1 ? 0xffff : 0;
  }
  return v;
}

template <class Cfg>
__device__ __forceinline__ void hc_prenorm_gemm_impl(const HcPrenormGemmArgs &p) {
#if defined(__HIP_DEVICE_COMPILE__)
  constexpr int FM = Cfg::FM, NF = Cfg::NF, BM = Cfg::BM, BN = Cfg::BN, BK = Cfg::BK;
  extern __shared__ __attribute__((aligned(16))) float smem[];
  const int tid = threadIdx.x, lane = tid % 64, g = lane / 16, r16 = lane % 16;
  const int wave = __builtin_amdgcn_readfirstlane(tid / 64);
  const int m0 = blockIdx.x * BM, n0 = blockIdx.z * BN, split = blockIdx.y;

  // K steps of this split, then of this wave
  const int k_steps = (p.k + BK - 1) / BK;
  const int split_steps = (k_steps + p.splits - 1) / p.splits;
  const int s_begin = min(k_steps, split * split_steps), s_end = min(k_steps, s_begin + split_steps);
  const int wave_steps = (s_end - s_begin + Cfg::WAVES - 1) / Cfg::WAVES;
  const int w_begin = min(s_end, s_begin + wave * wave_steps), w_end = min(s_end, w_begin + wave_steps);

  // rows past M / N read zeros (the descriptors end with the last row)
  const int rows = min(BM, p.m - m0), cols = min(BN, p.n - n0);
  const __amdgpu_buffer_rsrc_t a_rsrc = make_rsrc(p.a + (size_t)m0 * p.lda, (uint32_t)(rows * p.lda * 2));
  const __amdgpu_buffer_rsrc_t b_rsrc = make_rsrc(p.b + (size_t)n0 * p.ldb, (uint32_t)(cols * p.ldb * 4));

  // K offset (within a step) of the first value of A part q / B part c of lane group g
  auto a_k = [&](int q) { return 32 * q + 8 * g; };
  auto b_k = [&](int c) { return 32 * (c / 2) + 8 * g + 4 * (c % 2); };
  struct Step {
    v4i a[FM][4];  // [fragment][MFMA q]: 8 BF16 of row 16 i + r16 from K a_k(q)
    v4f b[NF][8];  // [fragment][part c]: 4 floats of row 16 j + r16 from K b_k(c)
  };
  auto load_step = [&](int s) {
    Step t;
    const bool in_range = s < w_end;
    const int k0 = s * BK;
#pragma unroll
    for (int i = 0; i < FM; ++i)
#pragma unroll
      for (int q = 0; q < 4; ++q)
        t.a[i][q] = in_range ? (v4i)__builtin_amdgcn_raw_buffer_load_b128(
                                   a_rsrc, (uint32_t)(((16 * i + r16) * p.lda + k0 + a_k(q)) * 2), 0, 0)
                             : v4i{};
#pragma unroll
    for (int j = 0; j < NF; ++j)
#pragma unroll
      for (int c = 0; c < 8; ++c)
        t.b[j][c] = in_range ? (v4f)__builtin_amdgcn_raw_buffer_load_b128(
                                   b_rsrc, (uint32_t)(((16 * j + r16) * p.ldb + k0 + b_k(c)) * 4), 0, 0)
                             : v4f{};
    return t;
  };

  v4f acc[FM][NF];
  float sq[FM];
#pragma unroll
  for (int i = 0; i < FM; ++i) {
    sq[i] = 0.f;
#pragma unroll
    for (int j = 0; j < NF; ++j)
      acc[i][j] = v4f{};
  }
  Step cur = load_step(w_begin);
  for (int s = w_begin; s < w_end; ++s) {
    const Step next = load_step(s + 1);
    __builtin_amdgcn_sched_barrier(0);
    // the last step of a K that is not a multiple of BK: values past K (the next row) become 0
    const int valid = p.k - s * BK;
    if (valid < BK) {
#pragma unroll
      for (int i = 0; i < FM; ++i)
#pragma unroll
        for (int q = 0; q < 4; ++q)
          cur.a[i][q] = keep_first_bf16(cur.a[i][q], valid - a_k(q));
#pragma unroll
      for (int j = 0; j < NF; ++j)
#pragma unroll
        for (int c = 0; c < 8; ++c)
#pragma unroll
          for (int e = 0; e < 4; ++e)
            cur.b[j][c][e] = b_k(c) + e < valid ? cur.b[j][c][e] : 0.f;
    }
    // B as BF16 hi and lo: the 8 values of MFMA q are floats 8 q .. + 7 (b[j][2 q], b[j][2 q + 1])
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      v4i hi[NF], lo[NF];
#pragma unroll
      for (int j = 0; j < NF; ++j) {
        float v[8];
#pragma unroll
        for (int e = 0; e < 4; ++e)
          v[e] = cur.b[j][2 * q][e], v[4 + e] = cur.b[j][2 * q + 1][e];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const uint32_t h = pack_bf16x2(v[2 * e], v[2 * e + 1]);
          float h0, h1;
          unpack_bf16x2(h, h0, h1);
          hi[j][e] = (int)h;
          lo[j][e] = (int)pack_bf16x2(v[2 * e] - h0, v[2 * e + 1] - h1);
        }
      }
#pragma unroll
      for (int i = 0; i < FM; ++i)
#pragma unroll
        for (int j = 0; j < NF; ++j) {
          acc[i][j] = mfma_bf16_16x16x32(hi[j], cur.a[i][q], acc[i][j]);
          acc[i][j] = mfma_bf16_16x16x32(lo[j], cur.a[i][q], acc[i][j]);
        }
      // squares of the row's 8 values
#pragma unroll
      for (int i = 0; i < FM; ++i)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          float x0, x1;
          unpack_bf16x2((uint32_t)cur.a[i][q][e], x0, x1);
          sq[i] = __builtin_fmaf(x0, x0, sq[i]);
          sq[i] = __builtin_fmaf(x1, x1, sq[i]);
        }
    }
    __builtin_amdgcn_sched_barrier(0);
    cur = next;
  }

  // ---- the wave's tile into LDS: acc[i][j][e] of lane (g, r16) is D[16 i + r16][16 j + 4 g + e];
  // a row's sum of squares adds its four lane groups' K slices
  float *const tile = smem + wave * Cfg::TILE_FLOATS;
#pragma unroll
  for (int i = 0; i < FM; ++i) {
#pragma unroll
    for (int j = 0; j < NF; ++j)
      *(v4f *)(tile + (16 * i + r16) * BN + 16 * j + 4 * g) = acc[i][j];
    float s = sq[i];
    s += __shfl_xor(s, 16);
    s += __shfl_xor(s, 32);
    if (g == 0)
      tile[BM * BN + 16 * i + r16] = s;
  }
  __syncthreads();
  // the wave tiles added in wave order
  float *const d = p.d + (size_t)split * p.m * p.n;
  for (int e = tid; e < BM * BN; e += Cfg::NT) {
    const int row = e / BN, col = e % BN;
    if (row >= rows || col >= cols)
      continue;
    float v = smem[e];
#pragma unroll
    for (int w = 1; w < Cfg::WAVES; ++w)
      v += smem[w * Cfg::TILE_FLOATS + e];
    d[(size_t)(m0 + row) * p.n + n0 + col] = v;
  }
  if (blockIdx.z == 0)
    for (int row = tid; row < rows; row += Cfg::NT) {
      float v = smem[BM * BN + row];
#pragma unroll
      for (int w = 1; w < Cfg::WAVES; ++w)
        v += smem[w * Cfg::TILE_FLOATS + BM * BN + row];
      p.sqr_sum[(size_t)split * p.m + m0 + row] = v;
    }
#endif
}

// final_d = sum_s d[s], final_sqr_sum = sum_s sqr_sum[s], in split order
__device__ __forceinline__ void hc_prenorm_reduce_impl(const HcPrenormGemmArgs &p) {
  const int64_t mn = (int64_t)p.m * p.n;
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < mn + p.m; i += (int64_t)gridDim.x * blockDim.x) {
    const bool is_d = i < mn;
    const float *src = is_d ? p.d + i : p.sqr_sum + (i - mn);
    const int64_t stride = is_d ? mn : p.m;
    float v = src[0];
    for (int s = 1; s < p.splits; ++s)
      v += src[s * stride];
    if (is_d)
      p.final_d[i] = v;
    else
      p.final_sqr_sum[i - mn] = v;
  }
}
