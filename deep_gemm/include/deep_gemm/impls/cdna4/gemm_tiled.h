#pragma once
#include <type_traits>
#include "device_common.h"

// ====================================== MFMA ========================================
// v_mfma_scale_f32_32x32x64_f8f6f4:
//   lane l, h = l / 32: A row / B col = l % 32; bytes 0-15 must come from K block 0 (k 0..31) and
//   bytes 16-31 from K block 1 (k 32..63) of this 64-wide step. We use k = 16h + j for bytes j < 16
//   and k = 32 + 16h + (j - 16) for bytes j >= 16, i.e. two 16-byte chunks: chunk h and chunk 2 + h.
//   Scale: lane l supplies the code of (row l % 32, block h) in byte `opsel` of its scale register.
//   Output: acc[i] of lane l is D[8 * (i / 4) + 4 * h + i % 4][l % 32].
template <int OPSEL>
__device__ __forceinline__ v16f mfma_mx_sel(const v8i& a, const v8i& b, const v16f& c, uint32_t sa, uint32_t sb) {
  return __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(a, b, c, kFmtFp8, kFmtFp8, OPSEL, sa, OPSEL, sb);
}

__device__ __forceinline__ v8i cat8(const v4i& lo, const v4i& hi) {
  return __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7);
}

// Store one 32x32 accumulator tile (rows r0.., cols c0..) with bounds checks.
template <class OutT>
__device__ __forceinline__ void store_tile32(const GemmArgs& p, const v16f& acc, int r0, int c0, int lane) {
  const int col = c0 + lane % 32, h = lane / 32;
  if (col >= p.N) return;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const int row = r0 + 8 * (i / 4) + 4 * h + i % 4;
    if (row < p.M) {
      float v = p.alpha * acc[i];
      if (p.C) v += load_out((const OutT*)p.C + (size_t)row * p.ldd + col);
      store_out((OutT*)p.D + (size_t)row * p.ldd + col, v);
    }
  }
}

// =================================== v2 .. v5 =======================================
// Block tile BM x BN, BK = 128 (one packed scale word per row), WM x WN waves, each wave computes
// (BM/WM) x (BN/WN) = TM x TN MFMA tiles of 32 x 32. LDS holds STAGES copies of the A and B tiles,
// row-major, 128 B per row; with SWIZZLE the 16 B chunk c of row r is stored at chunk c ^ (r & 7).
template <int BM_, int BN_, int WM_, int WN_, int STAGES_, bool SWIZZLE_, bool DIRECT_, int SCHED_ = 0,
          int GROUP_M_ = 8, int DEBUG_ = 0>
struct TileCfg {
  static constexpr int BM = BM_, BN = BN_, BK = 128, WM = WM_, WN = WN_, NW = WM_ * WN_, NT = 64 * NW;
  static constexpr int TM = BM / WM / 32, TN = BN / WN / 32;
  static constexpr int STAGES = STAGES_, SCHED = SCHED_, GROUP_M = GROUP_M_;
  static constexpr int DEBUG = DEBUG_;  // 1: load only the first K tile (compute-only upper bound)
  static constexpr bool SWIZZLE = SWIZZLE_, DIRECT = DIRECT_;
  static constexpr int A_BYTES = BM * BK, B_BYTES = BN * BK, STAGE_BYTES = A_BYTES + B_BYTES;
  static constexpr int LDS_BYTES = STAGES * STAGE_BYTES;
  static_assert(LDS_BYTES <= 160 * 1024, "CDNA4 LDS capacity");
  static_assert(BM % (32 * WM) == 0 && BN % (32 * WN) == 0, "wave tile must be a multiple of 32");
  static_assert((BM / 8) % NW == 0 && (BN / 8) % NW == 0, "each wave copies whole 8-row groups");
};

__device__ __forceinline__ void g2l_16(const void* g, uint8_t* lds_wave_base) {
  __builtin_amdgcn_global_load_lds((const __attribute__((address_space(1))) void*)g,
                                   (__attribute__((address_space(3))) void*)lds_wave_base, 16, 0, 0);
}

template <bool SWIZZLE>
__device__ __forceinline__ int lds_chunk(int row, int c) { return SWIZZLE ? (c ^ (row & 7)) : c; }

// Map a linear block id to (tile_m, tile_n): groups of GROUP_M m-tiles walk down N together so that
// the WGs running at the same time share A rows and B columns in L2 / MALL.
template <int GROUP_M>
__device__ __forceinline__ void tile_coords(int bid, int tiles_m, int tiles_n, int& tm, int& tn) {
  const int per_group = GROUP_M * tiles_n;
  const int g = bid / per_group, first_m = g * GROUP_M;
  const int gm = min(tiles_m - first_m, GROUP_M);
  const int r = bid % per_group;
  tm = first_m + r % gm;
  tn = r / gm;
}

// Workgroups are dispatched round-robin over the 8 XCDs (XCD = bid % 8), and each XCD has its own
// L2. Renumber so that every XCD works on a contiguous range of tile ids; tile_coords then turns
// the ~32 tiles an XCD runs at the same time into a compact block that shares A rows / B columns.
__device__ __forceinline__ int xcd_remap(int bid, int tiles) {
  constexpr int NXCD = 8;
  const int per = (tiles + NXCD - 1) / NXCD;          // tiles per XCD (the first `tall` get one more)
  const int tall = tiles % NXCD == 0 ? NXCD : tiles % NXCD;
  const int xcd = bid % NXCD, local = bid / NXCD;
  return xcd < tall ? xcd * per + local : tall * per + (xcd - tall) * (per - 1) + local;
}

template <class Cfg, class OutT>
__device__ __forceinline__ void gemm_tiled_impl(GemmArgs p) {
  constexpr int BM = Cfg::BM, BN = Cfg::BN, BK = Cfg::BK, TM = Cfg::TM, TN = Cfg::TN, NW = Cfg::NW;
  constexpr int NT = Cfg::NT, STAGES = Cfg::STAGES;
  constexpr bool SW = Cfg::SWIZZLE;
  extern __shared__ __attribute__((aligned(1024))) uint8_t smem[];

  const int tid = threadIdx.x, lane = tid % 64, wave = tid / 64, h = lane / 32;
  const int wm = wave / Cfg::WN, wn = wave % Cfg::WN;
  const int tiles_m = (p.M + BM - 1) / BM, tiles_n = (p.N + BN - 1) / BN;
  int tile_m, tile_n;
  tile_coords<Cfg::GROUP_M>(blockIdx.x, tiles_m, tiles_n, tile_m, tile_n);
  const int m0 = tile_m * BM, n0 = tile_n * BN;
  const int KT = p.K / BK;

  // ---------------- global -> LDS copy: which 16 B chunks this thread moves ----------------
  // DIRECT: one global_load_lds per wave fills 1 KiB = 8 rows of the LDS tile; lane l writes LDS
  //   chunk l of that KiB (row 8g + l/8, physical chunk l%8) and therefore fetches logical chunk
  //   (l%8) ^ (row & 7) of that row from global memory.
  // register path: thread t moves chunks t, t + NT, ...: row = q / 8, logical chunk q % 8.
  constexpr int A_PER = BM * 8 / NT, B_PER = BN * 8 / NT;  // 16 B chunks per thread per stage
  uint32_t a_goff[A_PER], b_goff[B_PER];                   // global byte offsets (without k0)
  uint32_t a_loff[A_PER], b_loff[B_PER];                   // LDS byte offsets inside a stage
#pragma unroll
  for (int i = 0; i < A_PER; ++i) {
    int row, c, lrow, lc;
    if (Cfg::DIRECT) { row = 8 * (i * NW + wave) + lane / 8; lc = lane % 8; c = lds_chunk<SW>(row, lc); }
    else { const int q = tid + i * NT; row = q / 8; c = q % 8; lc = lds_chunk<SW>(row, c); }
    lrow = row;
    a_goff[i] = (uint32_t)min(m0 + row, p.M - 1) * p.lda + c * 16;
    a_loff[i] = Cfg::DIRECT ? (uint32_t)(8 * (i * NW + wave)) * 128 : (uint32_t)(lrow * 128 + lc * 16);
  }
#pragma unroll
  for (int i = 0; i < B_PER; ++i) {
    int row, c, lrow, lc;
    if (Cfg::DIRECT) { row = 8 * (i * NW + wave) + lane / 8; lc = lane % 8; c = lds_chunk<SW>(row, lc); }
    else { const int q = tid + i * NT; row = q / 8; c = q % 8; lc = lds_chunk<SW>(row, c); }
    lrow = row;
    b_goff[i] = (uint32_t)min(n0 + row, p.N - 1) * p.ldb + c * 16;
    b_loff[i] = Cfg::A_BYTES + (Cfg::DIRECT ? (uint32_t)(8 * (i * NW + wave)) * 128 : (uint32_t)(lrow * 128 + lc * 16));
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
    coff[s][0] = lds_chunk<SW>(lane, 4 * s + h) * 16;
    coff[s][1] = lds_chunk<SW>(lane, 4 * s + 2 + h) * 16;
  }

  // ---------------- scales: one packed word per row per K tile, straight from global ----------------
  int sa_row[TM], sb_row[TN];
#pragma unroll
  for (int t = 0; t < TM; ++t) sa_row[t] = min(m0 + wm * (BM / Cfg::WM) + 32 * t + lane % 32, p.M - 1);
#pragma unroll
  for (int t = 0; t < TN; ++t) sb_row[t] = min(n0 + wn * (BN / Cfg::WN) + 32 * t + lane % 32, p.N - 1);
  uint32_t sa[TM], sb[TN];
  auto load_scales = [&](int kt, uint32_t* xa, uint32_t* xb) {
#pragma unroll
    for (int t = 0; t < TM; ++t) xa[t] = (uint32_t)p.sfa[(size_t)kt * p.lda_sf + sa_row[t]];
#pragma unroll
    for (int t = 0; t < TN; ++t) xb[t] = (uint32_t)p.sfb[(size_t)kt * p.ldb_sf + sb_row[t]];
  };

  v4i regs_a[Cfg::DIRECT ? 1 : A_PER], regs_b[Cfg::DIRECT ? 1 : B_PER];
  auto issue_global = [&](int kt, int stage) {  // DIRECT: straight into LDS; else into registers
    const uint8_t* ga = p.A + (size_t)kt * BK;
    const uint8_t* gb = p.B + (size_t)kt * BK;
    uint8_t* sbase = smem + stage * Cfg::STAGE_BYTES;
#pragma unroll
    for (int i = 0; i < A_PER; ++i) {
      if constexpr (Cfg::DIRECT) g2l_16(ga + a_goff[i], sbase + a_loff[i]);
      else regs_a[i] = *(const v4i*)(ga + a_goff[i]);
    }
#pragma unroll
    for (int i = 0; i < B_PER; ++i) {
      if constexpr (Cfg::DIRECT) g2l_16(gb + b_goff[i], sbase + b_loff[i]);
      else regs_b[i] = *(const v4i*)(gb + b_goff[i]);
    }
  };
  auto write_lds = [&](int stage) {  // register path only
    if constexpr (!Cfg::DIRECT) {
      uint8_t* sbase = smem + stage * Cfg::STAGE_BYTES;
#pragma unroll
      for (int i = 0; i < A_PER; ++i) *(v4i*)(sbase + a_loff[i]) = regs_a[i];
#pragma unroll
      for (int i = 0; i < B_PER; ++i) *(v4i*)(sbase + b_loff[i]) = regs_b[i];
    }
  };

  v16f acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = v16f{};

  auto compute = [&](int stage, const uint32_t* xa, const uint32_t* xb) {
    const uint8_t* sbase = smem + stage * Cfg::STAGE_BYTES;
    uint32_t ssa[TM], ssb[TN];  // put block (2s + h) of this lane into byte 2s
#pragma unroll
    for (int t = 0; t < TM; ++t) ssa[t] = xa[t] >> (8 * h);
#pragma unroll
    for (int t = 0; t < TN; ++t) ssb[t] = xb[t] >> (8 * h);
#pragma unroll
    for (int s = 0; s < 2; ++s) {
      v8i fa[TM], fb[TN];
#pragma unroll
      for (int t = 0; t < TM; ++t)
        fa[t] = cat8(*(const v4i*)(sbase + a_row + t * 32 * 128 + coff[s][0]),
                     *(const v4i*)(sbase + a_row + t * 32 * 128 + coff[s][1]));
#pragma unroll
      for (int t = 0; t < TN; ++t)
        fb[t] = cat8(*(const v4i*)(sbase + b_row + t * 32 * 128 + coff[s][0]),
                     *(const v4i*)(sbase + b_row + t * 32 * 128 + coff[s][1]));
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
          acc[i][j] = s == 0 ? mfma_mx_sel<0>(fa[i], fb[j], acc[i][j], ssa[i], ssb[j])
                             : mfma_mx_sel<2>(fa[i], fb[j], acc[i][j], ssa[i], ssb[j]);
      if constexpr (Cfg::SCHED == 1) {
        // interleave: 1 MFMA, then a few LDS reads of the next step, ...
#pragma unroll
        for (int i = 0; i < TM * TN; ++i) {
          __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);  // MFMA
          __builtin_amdgcn_sched_group_barrier(0x100, 1, 0);  // DS read
        }
      }
    }
  };

  if constexpr (STAGES == 1) {
    for (int kt = 0; kt < KT; ++kt) {
      issue_global(kt, 0);
      load_scales(kt, sa, sb);
      if constexpr (Cfg::DIRECT) __builtin_amdgcn_s_waitcnt(0x0f70);  // vmcnt(0)
      else write_lds(0);
      __syncthreads();
      compute(0, sa, sb);
      __syncthreads();
    }
  } else {
    uint32_t sa_n[TM], sb_n[TN];
    issue_global(0, 0);
    load_scales(0, sa_n, sb_n);
    if constexpr (!Cfg::DIRECT) write_lds(0);
    for (int kt = 0; kt < KT; ++kt) {
#pragma unroll
      for (int t = 0; t < TM; ++t) sa[t] = sa_n[t];
#pragma unroll
      for (int t = 0; t < TN; ++t) sb[t] = sb_n[t];
      if constexpr (Cfg::DIRECT) __builtin_amdgcn_s_waitcnt(0x0f70);  // vmcnt(0): tile kt is in LDS
      __syncthreads();  // tile kt visible to all waves; everyone is done reading the other stage
      const bool more = kt + 1 < KT && (Cfg::DEBUG != 1 || kt < 1);
      if (more) {
        issue_global(kt + 1, (kt + 1) % 2);
        load_scales(kt + 1, sa_n, sb_n);
      }
      compute(kt % 2, sa, sb);
      if constexpr (!Cfg::DIRECT) if (more) write_lds((kt + 1) % 2);
    }
  }

  // ---------------- epilogue ----------------
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j)
      store_tile32<OutT>(p, acc[i][j], m0 + wm * (BM / Cfg::WM) + 32 * i, n0 + wn * (BN / Cfg::WN) + 32 * j, lane);
}


// JIT candidate families.  The generated wrapper supplies launch bounds and
// dynamic LDS size; these aliases only describe legal tile geometry.
using Tiled64x64S1 = TileCfg<64, 64, 1, 1, 1, true, false>;
using Tiled64x64S2 = TileCfg<64, 64, 1, 1, 2, true, false>;
using Tiled128x128S1 = TileCfg<128, 128, 2, 2, 1, true, false>;
using Tiled128x128S2 = TileCfg<128, 128, 2, 2, 2, true, false>;
// A full 256x256 stage consumes 128 KiB of LDS, so only the one-stage
// variant fits the device LDS budget.
using Tiled256x256S1 = TileCfg<256, 256, 4, 4, 1, true, false>;
namespace deep_gemm::cdna4 {
using ::Tiled64x64S1;
using ::Tiled64x64S2;
using ::Tiled128x128S1;
using ::Tiled128x128S2;
using ::Tiled256x256S1;
} // namespace deep_gemm::cdna4
