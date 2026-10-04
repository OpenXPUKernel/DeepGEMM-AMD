#pragma once
#include "device_common.h"
#include "tile_scheduler.cuh"

// ====================================== MFMA ========================================
// v_mfma_scale_f32_32x32x64_f8f6f4:
//   lane l, h = l / 32: A row / B col = l % 32; bytes 0-15 must come from K block 0 (k 0..31) and
//   bytes 16-31 from K block 1 (k 32..63) of this 64-wide step. We use k = 16h + j for bytes j < 16
//   and k = 32 + 16h + (j - 16) for bytes j >= 16, i.e. two 16-byte chunks: chunk h and chunk 2 + h.
//   Scale: lane l supplies the code of (row l % 32, block h) in byte `opsel` of its scale register.
//   Output: acc[i] of lane l is D[8 * (i / 4) + 4 * h + i % 4][l % 32].
template <int OPSEL>
__device__ __forceinline__ v16f mfma_mx_sel(const v8i &a, const v8i &b, const v16f &c, uint32_t sa,
                                            uint32_t sb) {
  return __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(a, b, c, kFmtFp8, kFmtFp8, OPSEL, sa, OPSEL, sb);
}

// Store one 32x32 accumulator tile (rows r0.., cols c0..) with bounds checks.
template <class OutT>
__device__ __forceinline__ void store_tile32(const GemmArgs &p, const v16f &acc, int r0, int c0, int lane) {
  const int col = c0 + lane % 32, h = lane / 32;
  if (col >= p.N)
    return;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const int row = r0 + 8 * (i / 4) + 4 * h + i % 4;
    if (row < p.M) {
      float v = p.alpha * acc[i];
      if (p.C)
        v += load_out((const OutT *)p.C + (size_t)row * p.ldd + col);
      store_out((OutT *)p.D + (size_t)row * p.ldd + col, v);
    }
  }
}

// =============================== Tile configuration ================================
// Block tile BM x BN, BK = 128 (one packed scale word per row), WM x WN waves, each wave computes
// (BM/WM) x (BN/WN) = TM x TN MFMA tiles of 32 x 32. LDS holds STAGES copies of the A and B tiles,
// row-major, 128 B per row; the 16 B chunk c of row r is stored at chunk c ^ (r & 7).
template <int BM_, int BN_, int WM_, int WN_, int STAGES_, int GROUP_M_ = 8> struct TileCfg {
  static constexpr int BM = BM_, BN = BN_, BK = 128, WM = WM_, WN = WN_, NW = WM_ * WN_, NT = 64 * NW;
  static constexpr int TM = BM / WM / 32, TN = BN / WN / 32;
  static constexpr int STAGES = STAGES_, GROUP_M = GROUP_M_;
  static constexpr int A_BYTES = BM * BK, B_BYTES = BN * BK, STAGE_BYTES = A_BYTES + B_BYTES;
  static constexpr int LDS_BYTES = STAGES * STAGE_BYTES;
  static_assert(STAGES == 1 || STAGES == 2, "tiled GEMM supports one or two LDS stages");
  static_assert(LDS_BYTES <= 160 * 1024, "CDNA4 LDS capacity");
  static_assert(BM % (32 * WM) == 0 && BN % (32 * WN) == 0, "wave tile must be a multiple of 32");
  static_assert((BM / 8) % NW == 0 && (BN / 8) % NW == 0, "each wave copies whole 8-row groups");
};

__device__ __forceinline__ int lds_chunk(int row, int c) { return c ^ (row & 7); }

template <class Cfg, class OutT> __device__ __forceinline__ void gemm_tiled_impl(GemmArgs p) {
  constexpr int BM = Cfg::BM, BN = Cfg::BN, BK = Cfg::BK, TM = Cfg::TM, TN = Cfg::TN;
  constexpr int NT = Cfg::NT, STAGES = Cfg::STAGES;
  extern __shared__ __attribute__((aligned(1024))) uint8_t smem[];

  const int tid = threadIdx.x, lane = tid % 64, wave = tid / 64, h = lane / 32;
  const int wm = wave / Cfg::WN, wn = wave % Cfg::WN;
  const int tiles_m = (p.M + BM - 1) / BM, tiles_n = (p.N + BN - 1) / BN;
  int tile_m, tile_n;
  tile_coords<Cfg::GROUP_M>(blockIdx.x, tiles_m, tiles_n, tile_m, tile_n);
  const int m0 = tile_m * BM, n0 = tile_n * BN;
  const int KT = p.K / BK;

  // ---------------- global -> LDS copy: which 16 B chunks this thread moves ----------------
  // Thread t moves chunks t, t + NT, ... through registers: row = q / 8, logical chunk q % 8.
  constexpr int A_PER = BM * 8 / NT, B_PER = BN * 8 / NT; // 16 B chunks per thread per stage
  uint32_t a_goff[A_PER], b_goff[B_PER];                  // global byte offsets (without k0)
  uint32_t a_loff[A_PER], b_loff[B_PER];                  // LDS byte offsets inside a stage
#pragma unroll
  for (int i = 0; i < A_PER; ++i) {
    const int q = tid + i * NT, row = q / 8, c = q % 8;
    const int lc = lds_chunk(row, c);
    a_goff[i] = (uint32_t)min(m0 + row, p.M - 1) * p.lda + c * 16;
    a_loff[i] = (uint32_t)(row * 128 + lc * 16);
  }
#pragma unroll
  for (int i = 0; i < B_PER; ++i) {
    const int q = tid + i * NT, row = q / 8, c = q % 8;
    const int lc = lds_chunk(row, c);
    b_goff[i] = (uint32_t)min(n0 + row, p.N - 1) * p.ldb + c * 16;
    b_loff[i] = Cfg::A_BYTES + (uint32_t)(row * 128 + lc * 16);
  }

  // ---------------- MFMA operand reads from LDS ----------------
  // A fragment of MFMA tile tm, step s: row = wm*BM/WM + 32*tm + lane%32, chunks 4s + h and 4s + 2 + h.
  // (row & 7) == (lane & 7) because the row base is a multiple of 8, so the swizzled chunk offsets
  // depend on the lane only.
  const uint32_t a_row = (uint32_t)(wm * (BM / Cfg::WM) + lane % 32) * 128;
  const uint32_t b_row = Cfg::A_BYTES + (uint32_t)(wn * (BN / Cfg::WN) + lane % 32) * 128;
  uint32_t coff[2][2];
#pragma unroll
  for (int s = 0; s < 2; ++s) {
    coff[s][0] = lds_chunk(lane, 4 * s + h) * 16;
    coff[s][1] = lds_chunk(lane, 4 * s + 2 + h) * 16;
  }

  // ---------------- scales: one packed word per row per K tile, straight from global ----------------
  int sa_row[TM], sb_row[TN];
#pragma unroll
  for (int t = 0; t < TM; ++t)
    sa_row[t] = min(m0 + wm * (BM / Cfg::WM) + 32 * t + lane % 32, p.M - 1);
#pragma unroll
  for (int t = 0; t < TN; ++t)
    sb_row[t] = min(n0 + wn * (BN / Cfg::WN) + 32 * t + lane % 32, p.N - 1);
  uint32_t sa[TM], sb[TN];
  auto load_scales = [&](int kt, uint32_t *xa, uint32_t *xb) {
#pragma unroll
    for (int t = 0; t < TM; ++t)
      xa[t] = (uint32_t)p.sfa[(size_t)kt * p.lda_sf + sa_row[t]];
#pragma unroll
    for (int t = 0; t < TN; ++t)
      xb[t] = (uint32_t)p.sfb[(size_t)kt * p.ldb_sf + sb_row[t]];
  };

  v4i regs_a[A_PER], regs_b[B_PER];
  auto issue_global = [&](int kt) {
    const uint8_t *ga = p.A + (size_t)kt * BK;
    const uint8_t *gb = p.B + (size_t)kt * BK;
#pragma unroll
    for (int i = 0; i < A_PER; ++i)
      regs_a[i] = *(const v4i *)(ga + a_goff[i]);
#pragma unroll
    for (int i = 0; i < B_PER; ++i)
      regs_b[i] = *(const v4i *)(gb + b_goff[i]);
  };
  auto write_lds = [&](int stage) {
    uint8_t *sbase = smem + stage * Cfg::STAGE_BYTES;
#pragma unroll
    for (int i = 0; i < A_PER; ++i)
      *(v4i *)(sbase + a_loff[i]) = regs_a[i];
#pragma unroll
    for (int i = 0; i < B_PER; ++i)
      *(v4i *)(sbase + b_loff[i]) = regs_b[i];
  };

  v16f acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j)
      acc[i][j] = v16f{};

  auto compute = [&](int stage, const uint32_t *xa, const uint32_t *xb) {
    const uint8_t *sbase = smem + stage * Cfg::STAGE_BYTES;
    uint32_t ssa[TM], ssb[TN]; // put block (2s + h) of this lane into byte 2s
#pragma unroll
    for (int t = 0; t < TM; ++t)
      ssa[t] = xa[t] >> (8 * h);
#pragma unroll
    for (int t = 0; t < TN; ++t)
      ssb[t] = xb[t] >> (8 * h);
#pragma unroll
    for (int s = 0; s < 2; ++s) {
      v8i fa[TM], fb[TN];
#pragma unroll
      for (int t = 0; t < TM; ++t)
        fa[t] = cat8(*(const v4i *)(sbase + a_row + t * 32 * 128 + coff[s][0]),
                     *(const v4i *)(sbase + a_row + t * 32 * 128 + coff[s][1]));
#pragma unroll
      for (int t = 0; t < TN; ++t)
        fb[t] = cat8(*(const v4i *)(sbase + b_row + t * 32 * 128 + coff[s][0]),
                     *(const v4i *)(sbase + b_row + t * 32 * 128 + coff[s][1]));
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
          acc[i][j] = s == 0 ? mfma_mx_sel<0>(fa[i], fb[j], acc[i][j], ssa[i], ssb[j])
                             : mfma_mx_sel<2>(fa[i], fb[j], acc[i][j], ssa[i], ssb[j]);
    }
  };

  if constexpr (STAGES == 1) {
    for (int kt = 0; kt < KT; ++kt) {
      issue_global(kt);
      load_scales(kt, sa, sb);
      write_lds(0);
      __syncthreads();
      compute(0, sa, sb);
      __syncthreads();
    }
  } else {
    uint32_t sa_n[TM], sb_n[TN];
    issue_global(0);
    load_scales(0, sa_n, sb_n);
    write_lds(0);
    for (int kt = 0; kt < KT; ++kt) {
#pragma unroll
      for (int t = 0; t < TM; ++t)
        sa[t] = sa_n[t];
#pragma unroll
      for (int t = 0; t < TN; ++t)
        sb[t] = sb_n[t];
      __syncthreads(); // tile kt visible to all waves; everyone is done reading the other stage
      const bool more = kt + 1 < KT;
      if (more) {
        issue_global(kt + 1);
        load_scales(kt + 1, sa_n, sb_n);
      }
      compute(kt % 2, sa, sb);
      if (more)
        write_lds((kt + 1) % 2);
    }
  }

  // ---------------- epilogue ----------------
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j)
      store_tile32<OutT>(p, acc[i][j], m0 + wm * (BM / Cfg::WM) + 32 * i, n0 + wn * (BN / Cfg::WN) + 32 * j,
                         lane);
}

// JIT candidate families.  The generated wrapper supplies launch bounds and
// dynamic LDS size; these aliases only describe legal tile geometry.
using Tiled128x128S2 = TileCfg<128, 128, 2, 2, 2>;
// A full 256x256 stage consumes 128 KiB of LDS, so only the one-stage
// variant fits the device LDS budget.
using Tiled256x256S1 = TileCfg<256, 256, 4, 4, 1>;
namespace deep_gemm::cdna4 {
using ::Tiled128x128S2;
using ::Tiled256x256S1;
} // namespace deep_gemm::cdna4
