#pragma once
#include <type_traits>
#include "device_common.h"
/**
 * Shared HIP helpers for the FP8 pipeline.
 *
 * Shared LDS addressing, conversion, and reduction helpers for generated HIP
 * kernels. The entry bodies are device functions; launch wrappers are emitted
 * by the runtime JIT.
 */
#pragma once

// -----------------------------------------------------------------------------
// Configuration and synchronization helpers
// -----------------------------------------------------------------------------
#include <type_traits>

#include "gemm_tiled.h"

template <int BM_ = 256, int BN_ = 256, int GROUP_M_ = 8, bool XCD_REMAP_ = true, bool SPLITK_ = false,
          int DEBUG_ = 0, int SWZ_ = 1>
struct PipeCfg {
  static constexpr int BM = BM_, BN = BN_, BK = 128, WM = 2, WN = 2, NW = 4, NT = 256;
  static constexpr int TM = BM / WM / 32, TN = BN / WN / 32;
  static constexpr int GROUP_M = GROUP_M_;
  static constexpr bool XCD_REMAP = XCD_REMAP_, SPLITK = SPLITK_;
  // 1: no DMA in the main loop, 2: no epilogue stores, 3: the main loop always loads K tile 0
  static constexpr int DEBUG = DEBUG_;
  // LDS swizzle: 16 B chunk c of tile row r is stored at chunk c ^ key(r).
  //   0: key = r & 7. 2-way bank conflicts on CDNA4, whose 64 banks serve 16 lanes of a ds_read_b128
  //      per cycle: those lanes read rows r .. r + 15, and rows r and r + 8 get the same key.
  //   1: key = (r >> 1) & 7. Even rows (bank half 0) and odd rows (bank half 1) each get 8 different keys.
  static constexpr int SWZ = SWZ_;
  __host__ __device__ static constexpr int key(int r) { return SWZ ? (r >> 1) & 7 : r & 7; }
  // one stage: A tile | B tile | A scales (one int32 per row) | B scales
  static constexpr int A_OFF = 0, B_OFF = BM * BK, SA_OFF = B_OFF + BN * BK, SB_OFF = SA_OFF + BM * 4;
  static constexpr int STAGE_BYTES = SB_OFF + BN * 4, LDS_BYTES = 2 * STAGE_BYTES;
  static constexpr int A_LOADS = BM / 8 / NW, B_LOADS = BN / 8 / NW;  // per wave, 1 KiB each
  static constexpr int DMA_PER_TILE = A_LOADS + B_LOADS + 2;           // + one scale load for A and B
  static constexpr int FRAG_READS = 2 * (TM + TN);                     // ds_read_b128 per K step
  static constexpr int LDS_READS = FRAG_READS + TM + TN;               // + scale words
  // epilogue: per wave a 32 x 128 fp32 strip, rows padded by 16 B against bank conflicts
  static constexpr int EPI_LD = 128 + 4, EPI_WAVE_BYTES = 32 * EPI_LD * 4;
  static_assert(NW * EPI_WAVE_BYTES <= LDS_BYTES, "epilogue staging must fit in the operand stages");
  static_assert(BM == 256 && BN == 256, "scale DMA assumes one 64-row scale load per wave");
  static_assert(TM * TN == 16 && FRAG_READS == 16, "the hand-written schedule assumes 16 MFMAs per step");
};

// s_waitcnt immediate (CDNA4 encoding) that waits for vmcnt <= n and nothing else
constexpr int waitcnt_vm(int n) { return (n & 15) | ((n >> 4) << 14) | (7 << 4) | (15 << 8); }

template <int N> using int_c = std::integral_constant<int, N>;

typedef __attribute__((address_space(3))) void* lds_ptr_t;

typedef __bf16 v2bf __attribute__((ext_vector_type(2)));

__device__ __forceinline__ uint32_t pack_bf16x2(float a, float b) {  // v_cvt_pk_bf16_f32, RNE
  return __builtin_bit_cast(uint32_t, __builtin_convertvector((v2f){a, b}, v2bf));
}
__device__ __forceinline__ void unpack_bf16x2(uint32_t x, float& a, float& b) {
  a = __uint_as_float(x << 16);
  b = __uint_as_float(x & 0xffff0000u);
}

// -----------------------------------------------------------------------------
// Output conversion and vector stores
// -----------------------------------------------------------------------------
// Out[row][col .. col + 7] = v (+ C[row][col .. col + 7]) for the columns < N; v is already scaled
// by alpha. Out and C share the row stride ldo (elements).
template <class OutT>
__device__ __forceinline__ void store8(void* out, const void* C, int ldo, int M, int N, int row, int col, v4f lo,
                                       v4f hi, bool vec_ok) {
  if (row >= M || col >= N) return;
  const size_t off = (size_t)row * ldo + col;
  if (vec_ok && col + 8 <= N) {
    if constexpr (std::is_same<OutT, uint16_t>::value) {
      if (C) {
        const v4i c = *(const v4i*)((const uint16_t*)C + off);
        float c0, c1;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          unpack_bf16x2(c[i], c0, c1);
          lo[2 * i] += c0;
          lo[2 * i + 1] += c1;
          unpack_bf16x2(c[2 + i], c0, c1);
          hi[2 * i] += c0;
          hi[2 * i + 1] += c1;
        }
      }
      const v4i o = {(int)pack_bf16x2(lo[0], lo[1]), (int)pack_bf16x2(lo[2], lo[3]),
                     (int)pack_bf16x2(hi[0], hi[1]), (int)pack_bf16x2(hi[2], hi[3])};
      *(v4i*)((uint16_t*)out + off) = o;
    } else {
      if (C) {
        lo += *(const v4f*)((const float*)C + off);
        hi += *(const v4f*)((const float*)C + off + 4);
      }
      *(v4f*)((float*)out + off) = lo;
      *(v4f*)((float*)out + off + 4) = hi;
    }
  } else {
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      if (col + e >= N) break;
      float v = e < 4 ? lo[e] : hi[e - 4];
      if (C) v += load_out((const OutT*)C + off + e);
      store_out((OutT*)out + off + e, v);
    }
  }
}

// -----------------------------------------------------------------------------
// Buffer descriptors and legacy tiled kernel
// -----------------------------------------------------------------------------
// Kernel bodies that use AMDGPU-only types (buffer descriptors) are compiled for the device only;
// the host pass just needs the kernel symbol for the launch stub.
#if defined(__HIP_DEVICE_COMPILE__)
__device__ __forceinline__ __amdgpu_buffer_rsrc_t make_rsrc(const void* base, uint32_t bytes) {
  return __builtin_amdgcn_make_buffer_rsrc(const_cast<void*>(base), (short)0, (int)bytes, 0x00020000);
}
#endif

template <class Cfg, class OutT>
__device__ __forceinline__ void gemm_pipe_impl(GemmArgs p) {
#if defined(__HIP_DEVICE_COMPILE__)
  constexpr int BM = Cfg::BM, BN = Cfg::BN, BK = Cfg::BK, TM = Cfg::TM, TN = Cfg::TN, NW = Cfg::NW;
  extern __shared__ __attribute__((aligned(1024))) uint8_t smem[];

  const int tid = threadIdx.x, lane = tid % 64, h = lane / 32;
  const int wave = __builtin_amdgcn_readfirstlane(tid / 64);
  const int wm = wave / 2, wn = wave % 2;
  const int tiles_m = (p.M + BM - 1) / BM, tiles_n = (p.N + BN - 1) / BN, tiles = tiles_m * tiles_n;
  const int splits = Cfg::SPLITK ? p.splits : 1;
  const int uid = Cfg::XCD_REMAP ? xcd_remap(blockIdx.x, tiles * splits) : blockIdx.x;
  const int split = uid / tiles;
  int tile_m, tile_n;
  tile_coords<Cfg::GROUP_M>(uid - split * tiles, tiles_m, tiles_n, tile_m, tile_n);
  const int m0 = tile_m * BM, n0 = tile_n * BN;

  // K tiles of this workgroup: kb .. kb + nkt - 1 (the last K tile of the matrix may be partial).
  const int KT = (p.K + BK - 1) / BK, kps = (KT + splits - 1) / splits;
  const int kb = split * kps, nkt = min(KT - kb, kps);
  const bool tail = p.K % BK != 0 && kb + nkt == KT;
  // logical tile t -> K tile; the partial tile goes first so that it is fixed up in the prologue
  auto ktile = [&](int t) { return tail ? (t == 0 ? KT - 1 : kb + t - 1) : kb + t; };

  // ---- global -> LDS. Load i of this wave fills rows 8 * (i * NW + wave) .. + 7 of the tile;
  // lane l lands at LDS chunk l of that KiB (row r = 8 * (i * NW + wave) + l / 8, physical chunk
  // l % 8) and therefore fetches logical chunk (l % 8) ^ key(r). NW is even, so key(r) does not
  // depend on i and the per-lane offset is the same for every load.
  // descriptors end with the last row's K extent (rounded to the 16-byte load size)
  const uint32_t k16 = (uint32_t)(p.K + 15) & ~15u;
  const __amdgpu_buffer_rsrc_t ra = make_rsrc(p.A, (uint32_t)(p.M - 1) * p.lda + k16);
  const __amdgpu_buffer_rsrc_t rb = make_rsrc(p.B, (uint32_t)(p.N - 1) * p.ldb + k16);
  // scale descriptors end at the last word that belongs to the matrix (row M - 1 of the last K tile)
  const __amdgpu_buffer_rsrc_t rsa = make_rsrc(p.sfa, (uint32_t)((KT - 1) * p.lda_sf + p.M) * 4);
  const __amdgpu_buffer_rsrc_t rsb = make_rsrc(p.sfb, (uint32_t)((KT - 1) * p.ldb_sf + p.N) * 4);
  // per-lane part of the offset in a VGPR, everything uniform (tile row, load index, K) in soffset
  static_assert(NW % 2 == 0, "key(row of load i) must not depend on i");
  const uint32_t dma_chunk = (lane % 8) ^ Cfg::key(8 * wave + lane / 8);
  const uint32_t a_voff = (uint32_t)(lane / 8) * p.lda + dma_chunk * 16;
  const uint32_t b_voff = (uint32_t)(lane / 8) * p.ldb + dma_chunk * 16;
  const uint32_t a_soff = (uint32_t)(m0 + 8 * wave) * p.lda, b_soff = (uint32_t)(n0 + 8 * wave) * p.ldb;
  const uint32_t a_step = 8 * NW * p.lda, b_step = 8 * NW * p.ldb;  // between loads i and i + 1
  // scales: wave w copies the words of rows 64w .. 64w + 63 (4 B per lane)
  const uint32_t s_voff = (uint32_t)lane * 4;
  const uint32_t sa_soff = (uint32_t)(m0 + 64 * wave) * 4, sb_soff = (uint32_t)(n0 + 64 * wave) * 4;

  // DMA number j (0 .. DMA_PER_TILE - 1) of K tile kp into the stage at byte offset stage_off
  auto dma = [&](int j, uint32_t stage_off, int kp) {
    uint8_t* st = smem + stage_off;
    if (j < Cfg::A_LOADS)
      __builtin_amdgcn_raw_ptr_buffer_load_lds(ra, (lds_ptr_t)(st + Cfg::A_OFF + (j * NW + wave) * 1024), 16,
                                               a_voff, a_soff + j * a_step + kp * BK, 0, 0);
    else if (j < Cfg::A_LOADS + Cfg::B_LOADS)
      __builtin_amdgcn_raw_ptr_buffer_load_lds(rb, (lds_ptr_t)(st + Cfg::B_OFF + ((j - Cfg::A_LOADS) * NW + wave) * 1024),
                                               16, b_voff, b_soff + (j - Cfg::A_LOADS) * b_step + kp * BK, 0, 0);
    else if (j == Cfg::A_LOADS + Cfg::B_LOADS)
      __builtin_amdgcn_raw_ptr_buffer_load_lds(rsa, (lds_ptr_t)(st + Cfg::SA_OFF + wave * 256), 4, s_voff,
                                               sa_soff + kp * p.lda_sf * 4, 0, 0);
    else
      __builtin_amdgcn_raw_ptr_buffer_load_lds(rsb, (lds_ptr_t)(st + Cfg::SB_OFF + wave * 256), 4, s_voff,
                                               sb_soff + kp * p.ldb_sf * 4, 0, 0);
  };

  // ---- LDS -> registers. Operands are kept as 16-byte halves: f[r], r < 8: A tile r / 2,
  // r >= 8: B tile (r - 8) / 2; half r % 2. A fragment of tile t, step s: row wm*128 + 32t +
  // lane%32, chunk 4s + h (half 0) and 4s + 2 + h (half 1); physical chunk = logical ^ key(lane % 32).
  const uint32_t a_row = (uint32_t)(wm * (BM / 2) + lane % 32) * 128;
  const uint32_t b_row = (uint32_t)(wn * (BN / 2) + lane % 32) * 128;
  const uint32_t rkey = Cfg::key(lane % 32);
  uint32_t coff[2][2];
#pragma unroll
  for (int s = 0; s < 2; ++s) {
    coff[s][0] = ((4 * s + h) ^ rkey) * 16;
    coff[s][1] = ((4 * s + 2 + h) ^ rkey) * 16;
  }
  const uint32_t sa_lds = (uint32_t)(wm * (BM / 2) + lane % 32) * 4;
  const uint32_t sb_lds = (uint32_t)(wn * (BN / 2) + lane % 32) * 4;
  const uint32_t sh = 8 * h;  // lane half h uses K block 2s + h = byte 2s of (word >> 8h)

  v4i f0[16], f1[16];        // operands of step 0 / step 1
  uint32_t sca[TM], scb[TN];  // raw scale words of the next tile

  // LDS read number r (0 .. LDS_READS - 1) of step S from the stage at stage_off: operand halves
  // first, then the scale words (only read together with step 0)
  auto lds_read = [&](int r, int S, uint32_t stage_off, v4i* f) {
    const uint8_t* st = smem + stage_off;
    if (r < 8)
      f[r] = *(const v4i*)(st + Cfg::A_OFF + a_row + (r / 2) * 4096 + coff[S][r % 2]);
    else if (r < 16)
      f[r] = *(const v4i*)(st + Cfg::B_OFF + b_row + ((r - 8) / 2) * 4096 + coff[S][r % 2]);
    else if (r < 16 + TM)
      sca[r - 16] = *(const uint32_t*)(st + Cfg::SA_OFF + sa_lds + (r - 16) * 128);
    else
      scb[r - 16 - TM] = *(const uint32_t*)(st + Cfg::SB_OFF + sb_lds + (r - 16 - TM) * 128);
  };

  v16f acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = v16f{};

  // MFMA number i (0..15) of step S: output tile (i / TN, i % TN). B goes into the MFMA's A slot,
  // so the MFMA computes the transposed tile: acc[ti][tj][e] of lane l is
  //   D[wm*128 + 32 ti + l % 32][wn*128 + 32 tj + 8 (e / 4) + 4 h + e % 4],
  // i.e. every lane holds runs of 4 consecutive columns of one row (see the epilogue).
  auto mfma = [&](int i, auto step_c, const v4i* f, const uint32_t* xa, const uint32_t* xb) {
    constexpr int S = decltype(step_c)::value;
    const int ti = i / TN, tj = i % TN;
    acc[ti][tj] = mfma_mx_sel<2 * S>(cat8(f[8 + 2 * tj], f[8 + 2 * tj + 1]), cat8(f[2 * ti], f[2 * ti + 1]),
                                     acc[ti][tj], xb[tj], xa[ti]);
  };

  // One K tile; logical tile kt sits in stage kt & 1. Tiles past the end are clamped to the last
  // one, which only re-loads data nobody reads, so every iteration runs the same code.
  auto iter = [&](int kt) {
    const uint32_t cur = (kt & 1) * Cfg::STAGE_BYTES, nxt = Cfg::STAGE_BYTES - cur;
    uint32_t xa[TM], xb[TN];
#pragma unroll
    for (int t = 0; t < TM; ++t) xa[t] = sca[t] >> sh;
#pragma unroll
    for (int t = 0; t < TN; ++t) xb[t] = scb[t] >> sh;

    // first half: step-0 MFMAs, each followed by one ds_read of the step-1 operands
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      mfma(i, int_c<0>{}, f0, xa, xb);
      lds_read(i, 1, cur, f1);
      __builtin_amdgcn_sched_barrier(0);
    }
    // everyone is done with this stage, and tile kt + 1 has fully landed in the other one
    asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)\n\ts_barrier" ::: "memory");
    __builtin_amdgcn_sched_barrier(0);

    // second half: step-1 MFMAs; refill this stage with tile kt + 2 (2 DMAs after each of the
    // first 9 MFMAs) and read the step-0 operands + scales of tile kt + 1 (2 reads after each of
    // the first 12 MFMAs)
    const int kl = ktile(Cfg::DEBUG == 3 ? 0 : min(kt + 2, nkt - 1));
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      mfma(i, int_c<1>{}, f1, xa, xb);
#pragma unroll
      for (int j = 2 * i; j < 2 * i + 2; ++j)
        if (j < Cfg::DMA_PER_TILE && Cfg::DEBUG != 1) dma(j, cur, kl);
#pragma unroll
      for (int r = 2 * i; r < 2 * i + 2; ++r)
        if (r < Cfg::LDS_READS) lds_read(r, 0, nxt, f0);
      __builtin_amdgcn_sched_barrier(0);
    }
  };

  // ---- prologue: tiles 0 and 1 in flight, wait for tile 0, read its step-0 operands
#pragma unroll
  for (int j = 0; j < Cfg::DMA_PER_TILE; ++j) dma(j, 0, ktile(0));
#pragma unroll
  for (int j = 0; j < Cfg::DMA_PER_TILE; ++j) dma(j, Cfg::STAGE_BYTES, ktile(min(1, nkt - 1)));
  __builtin_amdgcn_s_waitcnt(waitcnt_vm(Cfg::DMA_PER_TILE));
  asm volatile("s_barrier" ::: "memory");
  if (tail) {
    // Tile 0 is the partial last K tile: only kv bytes of each row are data. Zero the rest (it holds
    // the next row, padding, or 0 past the end of the buffer), and set the scale codes of K blocks
    // without data to 127 (= 1.0) so that padding codes cannot turn 0 * scale into NaN.
    const int kv = p.K - (KT - 1) * BK, nb = (kv + 31) / 32;
    for (int r = tid; r < BM + BN; r += Cfg::NT) {
      uint8_t* row = smem + (r < BM ? Cfg::A_OFF + r * BK : Cfg::B_OFF + (r - BM) * BK);
      for (int c = kv / 16; c < 8; ++c) {  // logical chunk c is stored at chunk c ^ key(row)
        v4i* q = (v4i*)(row + (c ^ Cfg::key(r < BM ? r : r - BM)) * 16);
        v4i v = *q;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int keep = kv - c * 16 - 4 * i;  // bytes of dword i that are data
          v[i] = keep >= 4 ? v[i] : keep <= 0 ? 0 : (int)((uint32_t)v[i] & ((1u << (8 * keep)) - 1));
        }
        *q = v;
      }
      if (nb < 4) {
        uint32_t* w = (uint32_t*)(smem + (r < BM ? Cfg::SA_OFF + r * 4 : Cfg::SB_OFF + (r - BM) * 4));
        const uint32_t keep = (1u << (8 * nb)) - 1;
        *w = (*w & keep) | (0x7f7f7f7fu & ~keep);
      }
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
  }
#pragma unroll
  for (int r = 0; r < Cfg::LDS_READS; ++r) lds_read(r, 0, 0, f0);

  for (int kt = 0; kt < nkt; ++kt) iter(kt);
  // all DMAs have landed and every wave is done with the operand stages: LDS is free
  asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)\n\ts_barrier" ::: "memory");

  if constexpr (Cfg::DEBUG == 2) {
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
      for (int j = 0; j < TN; ++j) s += acc[i][j][0] + acc[i][j][15];
    if (s == 1234.5f) ((float*)p.D)[tid] = s;
    return;
  }

  // ---- epilogue: each wave stages its 128 x 128 result through its own LDS strip buffer, 32 rows
  // at a time, then every lane writes 8 consecutive outputs (16 B of bf16): one store instruction
  // covers 4 rows x 128 columns. Split-K writes the raw fp32 partial into the workspace instead.
  using OutE = typename std::conditional<Cfg::SPLITK, float, OutT>::type;
  void* const out = Cfg::SPLITK ? (void*)(p.ws + (size_t)split * p.M * p.ldw) : p.D;
  const void* const cin = Cfg::SPLITK ? nullptr : p.C;
  const int ldo = Cfg::SPLITK ? p.ldw : p.ldd;
  const float alpha = Cfg::SPLITK ? 1.f : p.alpha;
  float* stg = (float*)(smem + wave * Cfg::EPI_WAVE_BYTES);
  const int er = lane / 16, ec = (lane % 16) * 8;
  const bool vec_ok = (ldo * (int)sizeof(OutE)) % 16 == 0 && ((uintptr_t)out & 15) == 0 &&
                      (!cin || ((uintptr_t)cin & 15) == 0);
#pragma unroll
  for (int ti = 0; ti < TM; ++ti) {
#pragma unroll
    for (int tj = 0; tj < TN; ++tj)
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const v4f v = {acc[ti][tj][4 * j], acc[ti][tj][4 * j + 1], acc[ti][tj][4 * j + 2], acc[ti][tj][4 * j + 3]};
        *(v4f*)(stg + (lane % 32) * Cfg::EPI_LD + 32 * tj + 8 * j + 4 * h) = v * alpha;
      }
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const int r = er + 4 * k;
      const v4f lo = *(const v4f*)(stg + r * Cfg::EPI_LD + ec), hi = *(const v4f*)(stg + r * Cfg::EPI_LD + ec + 4);
      store8<OutE>(out, cin, ldo, p.M, p.N, m0 + wm * (BM / 2) + 32 * ti + r, n0 + wn * (BN / 2) + ec, lo, hi,
                   vec_ok);
    }
  }
#endif
}

// D = alpha * sum_s ws[s] (+ C), partials added in split order so the result does not depend on
// which workgroup finished first. One thread per 4 columns; blockIdx.y walks the rows.
template <class OutT>
__device__ __forceinline__ void splitk_reduce_impl(GemmArgs p) {
  const int col = (blockIdx.x * 256 + threadIdx.x) * 4;
  if (col >= p.N) return;
  constexpr int VB = 4 * sizeof(OutT);  // bytes of 4 outputs
  const bool vec_ok = (p.ldd * (int)sizeof(OutT)) % VB == 0 && (uintptr_t)p.D % VB == 0 &&
                      (!p.C || (uintptr_t)p.C % VB == 0) && col + 4 <= p.N;
  const size_t split_stride = (size_t)p.M * p.ldw;
  for (int row = blockIdx.y; row < p.M; row += gridDim.y) {
    const float* w = p.ws + (size_t)row * p.ldw + col;
    v4f s = *(const v4f*)w;
    for (int k = 1; k < p.splits; ++k) s += *(const v4f*)(w + k * split_stride);
    s *= p.alpha;
    const size_t off = (size_t)row * p.ldd + col;
    if (vec_ok) {
      if constexpr (std::is_same<OutT, uint16_t>::value) {
        if (p.C) {
          const uint2 c = *(const uint2*)((const uint16_t*)p.C + off);
          float c0, c1, c2, c3;
          unpack_bf16x2(c.x, c0, c1);
          unpack_bf16x2(c.y, c2, c3);
          s += v4f{c0, c1, c2, c3};
        }
        *(uint2*)((uint16_t*)p.D + off) = uint2{pack_bf16x2(s[0], s[1]), pack_bf16x2(s[2], s[3])};
      } else {
        if (p.C) s += *(const v4f*)((const float*)p.C + off);
        *(v4f*)((float*)p.D + off) = s;
      }
    } else {
      for (int e = 0; e < 4 && col + e < p.N; ++e) {
        float v = s[e];
        if (p.C) v += load_out((const OutT*)p.C + off + e);
        store_out((OutT*)p.D + off + e, v);
      }
    }
  }
}


namespace deep_gemm::cdna4 {
using ::PipeCfg;
} // namespace deep_gemm::cdna4
