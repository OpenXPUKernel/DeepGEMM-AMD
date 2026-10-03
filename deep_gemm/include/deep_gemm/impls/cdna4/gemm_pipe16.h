#pragma once
#include <type_traits>
#include "gemm_pipe.h"
// -----------------------------------------------------------------------------
// Compile-time unrolling
// -----------------------------------------------------------------------------
// f(integral_constant<int, I>) for I = B .. E - 1: a compile-time unrolled loop. The 64 MFMA slots of
// an iteration are too big for `#pragma unroll` to be reliable, and a rolled loop would put the
// register arrays in scratch.
template <int B, int E, class F>
__device__ __forceinline__ void static_for(F&& f) {
  if constexpr (B < E) {
    f(std::integral_constant<int, B>{});
    static_for<B + 1, E>(f);
  }
}

// -----------------------------------------------------------------------------
// Pipeline configuration and LDS layout
// -----------------------------------------------------------------------------
template <int GROUP_M_ = 8, bool XCD_REMAP_ = true, bool SPLITK_ = false, int DEBUG_ = 0, int BAR1_ = 16,
          int SPAN_ = 48, bool PERSIST_ = false, bool EPI_LDS_ = false, bool ONE_BAR_ = false, bool DYN_ = false,
          int L2PF_ = 0, bool TAIL_ = false, int MN_ = 0>
struct Pipe16Cfg {
  static constexpr int BM = 256, BN = 256, BK = 128, NW = 4, NT = 256;
  static constexpr int GROUP_M = GROUP_M_;
  static constexpr bool XCD_REMAP = XCD_REMAP_, SPLITK = SPLITK_;
  static constexpr bool PERSIST = PERSIST_;
  // MN-major operands (bit 0: A, bit 1: B): the operand is stored as [K][MN] with row stride ld
  // (lda / ldb). Its tile sits in LDS as 128 K rows x 256 B (16 B chunk c of row r at c ^ keyMN(r))
  // and is read with ds_read_b64_tr_b8, which transposes 8 x 8 bytes in hardware.
  static constexpr bool A_MN = MN_ & 1, B_MN = MN_ & 2;
  // PERSIST with a K-major B: B fragments 2p and 2p + 1 give a lane 8 consecutive output columns
  // (one 16 B store). A transposing read needs 16 consecutive rows per fragment, so an MN-major B
  // keeps the plain layout and the epilogue writes 4 columns per fragment.
  static constexpr bool BPAIR = PERSIST && !B_MN;
  // PERSIST, bf16: the epilogue goes through LDS (after the two stages) so that a store instruction
  // writes 2 rows x 512 B instead of 16 rows x 64 B
  static constexpr bool EPI_LDS = EPI_LDS_ && PERSIST;
  static_assert(!(PERSIST && SPLITK), "the persistent kernel has no split-K");
  // v14 (TAIL, PERSIST + XCD_REMAP): the partial last round of tiles goes to blocks 0 .. rest - 1,
  // i.e. spread across the available workgroups instead of concentrating them at the front.
  static constexpr bool TAIL_SPREAD = TAIL_ && PERSIST && XCD_REMAP;
  // experiments (optional scheduling modes):
  //   ONE_BAR: one barrier per K step, at BAR1. Before it every wave waits until step s + 1 has
  //            landed; after it stage cur is free and stage nxt is readable, so barrier 2 goes away.
  //   DYN    : PERSIST; every tile after the first is claimed at run time from per-XCD queues
  //            (p.tile_ctr, 9 zeroed ints; a workgroup whose XCD has run dry steals from the others).
  //   L2PF   : touch K tile s + L2PF (one 4 B DMA per 128 B line, into a dummy LDS area) so that its
  //            DMA later hits in L2.
  static constexpr bool ONE_BAR = ONE_BAR_, DYN = DYN_ && PERSIST_;
  static constexpr int L2PF = L2PF_;
  // timing experiments (wrong results), bit mask: 1 no DMA in the main loop, 2 no epilogue stores,
  // 4 the main loop always loads K tile 0, 8 no scale reads, 16 no barriers, 32 no operand reads,
  // 64 all output rows of a tile go to its first row. 128 (PERSIST, correct results): every wave
  // writes s_memrealtime stamps to ((uint64_t*)ws)[(4 blockIdx.x + wave) * 64 + e]: e = 0 start,
  // 1 prologue done, 2 + 4 t + {0 first K step, 1 last K step, 2 epilogue, 3 epilogue done} of tile t;
  // with 1024 as well, e <= 31 and the shader clock counter (s_memtime) goes to slot 32 + e.
  // 256 (DYN, correct results): the DYN loop with the static tile sequence and no atomics.
  static constexpr int DEBUG = DEBUG_;
  static constexpr bool NO_DMA = DEBUG & 1, NO_STORE = DEBUG & 2, K0_ONLY = DEBUG & 4, NO_SCALE = DEBUG & 8,
                        NO_BAR = DEBUG & 16, NO_RD = DEBUG & 32, ONE_ROW = DEBUG & 64, TRACE = DEBUG & 128,
                        DYN_STATIC = DEBUG & 256, CLK_TRACE = DEBUG & 1024;
  static constexpr int A_OFF = 0, B_OFF = BM * BK, SA_OFF = B_OFF + BN * BK, SB_OFF = SA_OFF + BM * 4;
  static constexpr int STAGE_BYTES = SB_OFF + BN * 4;
  // EPI_LDS buffer: 32 rows (16 of each wave row) x 256 bf16, rows padded to 528 B against bank conflicts
  static constexpr int EPI_OFF = 2 * STAGE_BYTES, EPI_PITCH = 2 * BN + 16;
  // DYN: the next tile id, written by wave 0 and read by all; L2PF: destination of the touch DMAs
  static constexpr int DYN_OFF = EPI_OFF + (EPI_LDS_ && PERSIST_ ? 32 * EPI_PITCH : 0);
  static constexpr int PF_OFF = DYN_OFF + (DYN ? 16 : 0);
  static constexpr int LDS_BYTES = PF_OFF + (L2PF ? NW * 256 : 0);
  static_assert(LDS_BYTES <= 160 * 1024, "CDNA4 LDS");
  static constexpr int A_LOADS = BM / 8 / NW, B_LOADS = BN / 8 / NW;  // per wave, 1 KiB each
  static constexpr int DMA_PER_TILE = A_LOADS + B_LOADS + 2;           // + one scale load for A and B
  static constexpr int EPI_LD = 128 + 4, EPI_WAVE_BYTES = 32 * EPI_LD * 4;
  static_assert(NW * EPI_WAVE_BYTES <= LDS_BYTES, "epilogue staging must fit in the operand stages");
  // slots (MFMA numbers 0..63 of an iteration) of the two barriers; the DMAs of tile kt + 2 are
  // spread evenly over the SPAN slots that follow barrier 1
  static constexpr int BAR1 = BAR1_, BAR2 = 48, SPAN = SPAN_;
  // LDS reads per fragment: K-major 2 halves (ds_read_b128) + 1 scale code; MN-major 4 transposing
  // reads of 8 K bytes each + 1 scale code. A set is 4 A + 4 B fragments, RPS reads per slot.
  static constexpr int FR_A = A_MN ? 5 : 3, FR_B = B_MN ? 5 : 3;
  static constexpr int READS = 4 * (FR_A + FR_B), RPS = READS > 32 ? 3 : 2, RD_SLOTS = (READS + RPS - 1) / RPS;
  static_assert(BAR1 >= RD_SLOTS && BAR1 + SPAN <= 64, "DMA window: after the reads of stage cur, within the iteration");
  static_assert(BAR2 + RD_SLOTS <= 64, "the reads of the next step fit after barrier 2");
  static_assert(!(L2PF && MN_), "L2PF touches K-major rows");
  // read r of set 0 (fragments A 4-7, B 4-7) or set 1 (B 0, A 0-3, B 1-3: the order in which the
  // first quadrant of the next step needs them): fragment f of A or B, part `sub` of it
  struct RdItem {
    bool isB;
    int f, sub;
  };
  __host__ __device__ static constexpr RdItem rd_item(int set, int r) {
    for (int item = 0; item < 8; ++item) {
      const bool isB = set == 0 ? item >= 4 : item == 0 || item >= 5;
      const int f = set == 0 ? 4 + item % 4 : item == 0 ? 0 : item <= 4 ? item - 1 : item - 4;
      const int n = isB ? FR_B : FR_A;
      if (r < n) return RdItem{isB, f, r};
      r -= n;
    }
    return RdItem{false, 0, -1};
  }
  static_assert(SPAN >= DMA_PER_TILE, "at most one DMA per slot");
  __host__ __device__ static constexpr int dma_slot(int j) { return BAR1 + j * SPAN / DMA_PER_TILE; }
  // the DMA issued in slot t, or -1 (closed form, so that the unrolled loop folds it)
  __host__ __device__ static constexpr int dma_at(int t) {
    const int u = t - BAR1, j = (u * DMA_PER_TILE + SPAN - 1) / SPAN;
    return u >= 0 && j < DMA_PER_TILE && j * SPAN / DMA_PER_TILE == u ? j : -1;
  }
  // DMAs of tile kt + 2 issued before barrier 2 (where tile kt + 1 must have landed)
  static constexpr int dma_before_bar2() {
    int n = 0;
    for (int j = 0; j < DMA_PER_TILE; ++j) n += dma_slot(j) < BAR2;
    return n;
  }
  // B row (within the wave's 128) read by lane row r16 of B fragment j: bfrag(j) + brow(r16)
  __host__ __device__ static constexpr int bfrag(int j) { return BPAIR ? 32 * (j / 2) + 4 * (j % 2) : 16 * j; }
  __host__ __device__ static constexpr int brow(int r16) { return BPAIR ? 8 * (r16 / 4) + r16 % 4 : r16; }
  // MN-major tile: the 16 lanes of a transposing read point at 8 consecutive K rows, two 8 B halves
  // of one chunk each, and the 4 lane groups at rows 16 apart: 32 rows must cover the 16 chunk
  // positions twice, so the key takes row bits 0-2 and 4.
  __host__ __device__ static constexpr int keyMN(int r) { return (r & 7) | (((r >> 4) & 1) << 3); }
  // MN-major DMA j of wave w (4 K rows x 256 B per DMA): rows mn_row(w) + mn_jrow(j) + 0..3. The
  // wave and the lane fix row bits 0-2 and 4 (so the lane's key does not depend on j), j sets bits
  // 3, 5 and 6.
  __host__ __device__ static constexpr int mn_row(int w) { return 4 * (w & 1) + 16 * (w >> 1); }
  __host__ __device__ static constexpr int mn_jrow(int j) { return ((j & 1) << 3) | ((j >> 1) << 5); }
  // swizzle keys (physical chunk = logical chunk ^ key(row)). The 16 lanes that one ds_read_b128
  // cycle serves read rows r16 = 0..15 of a fragment and must hit 16 different 16 B bank groups,
  // i.e. different (row % 2, key) pairs. A: rows 0..15 -> (r >> 1) & 7. B with PERSIST: rows
  // {8a + b : a, b < 4} -> bit 1 and bits 3-4 of the row.
  __host__ __device__ static constexpr int key(int r) { return (r >> 1) & 7; }
  __host__ __device__ static constexpr int keyB(int r) {
    return BPAIR ? ((r >> 1) & 1) | (((r >> 3) & 3) << 1) : key(r);
  }
};

// -----------------------------------------------------------------------------
// Device load/store primitives
// -----------------------------------------------------------------------------
#if defined(__HIP_DEVICE_COMPILE__)
// A 16 B buffer store reads the upper 8 bytes of its data a cycle after issue, so the next
// instruction must not overwrite them (ISA "manually inserted wait states"). LLVM only inserts that
// wait state for stores whose soffset is not an SGPR, and these stores always pass one.
__device__ __forceinline__ void buffer_store16(v4u v, __amdgpu_buffer_rsrc_t r, uint32_t voff, int soff) {
  __builtin_amdgcn_raw_buffer_store_b128(v, r, voff, soff, 0);
  __builtin_amdgcn_sched_barrier(0);
  asm volatile("s_nop 0");
  __builtin_amdgcn_sched_barrier(0);
}
#endif

#if defined(__HIP_DEVICE_COMPILE__)
// ds_read_b64_tr_b8. LLVM makes an LDS read wait for every LDS DMA in flight (s_waitcnt vmcnt) unless
// the read carries alias metadata; ds_read_b128 has TBAA, the intrinsic call has none. Inlining a
// function with a restrict parameter attaches noalias scopes to the call, which is enough.
__device__ __forceinline__ v2i ds_read_tr8(const __attribute__((address_space(3))) v2i* __restrict__ p) {
  return __builtin_amdgcn_ds_read_tr8_b64_v2i32(const_cast<__attribute__((address_space(3))) v2i*>(p));
}
#endif

// -----------------------------------------------------------------------------
// Epilogue helpers
// -----------------------------------------------------------------------------
// store8 for 4 consecutive outputs of a row (persistent kernel with an MN-major B)
template <class OutT>
__device__ __forceinline__ void store4(void* out, const void* C, int ldo, int M, int N, int row, int col, v4f v,
                                       bool vec_ok) {
  if (row >= M || col >= N) return;
  const size_t off = (size_t)row * ldo + col;
  if (vec_ok && col + 4 <= N) {
    if constexpr (std::is_same<OutT, uint16_t>::value) {
      if (C) {
        const v2i c = *(const v2i*)((const uint16_t*)C + off);
        float c0, c1;
        unpack_bf16x2(c[0], c0, c1);
        v[0] += c0;
        v[1] += c1;
        unpack_bf16x2(c[1], c0, c1);
        v[2] += c0;
        v[3] += c1;
      }
      *(v2u*)((uint16_t*)out + off) = v2u{pack_bf16x2(v[0], v[1]), pack_bf16x2(v[2], v[3])};
    } else {
      if (C) v += *(const v4f*)((const float*)C + off);
      *(v4f*)((float*)out + off) = v;
    }
  } else {
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      if (col + e >= N) break;
      float x = v[e];
      if (C) x += load_out((const OutT*)C + off + e);
      store_out((OutT*)out + off + e, x);
    }
  }
}

// -----------------------------------------------------------------------------
// Main pipelined GEMM kernel
// -----------------------------------------------------------------------------
template <class Cfg, class OutT>
__device__ __forceinline__ void gemm_pipe16_impl(GemmArgs p) {
#if defined(__HIP_DEVICE_COMPILE__)
  constexpr int BM = Cfg::BM, BN = Cfg::BN, BK = Cfg::BK, NW = Cfg::NW;
  extern __shared__ __attribute__((aligned(1024))) uint8_t smem[];

  const int tid = threadIdx.x, lane = tid % 64, g = lane / 16, r16 = lane % 16;
  const int wave = __builtin_amdgcn_readfirstlane(tid / 64);
  const int wm = wave / 2, wn = wave % 2;
  const int tiles_m = (p.M + BM - 1) / BM, tiles_n = (p.N + BN - 1) / BN, tiles = tiles_m * tiles_n;
  const int KT = (p.K + BK - 1) / BK;
  auto stamp = [&](int e) {
    if constexpr (Cfg::TRACE && Cfg::CLK_TRACE) {  // slots 32 + e: shader clock counter
      ((uint64_t*)p.ws)[(4 * blockIdx.x + wave) * 64 + min(e, 31)] = __builtin_amdgcn_s_memrealtime();
      ((uint64_t*)p.ws)[(4 * blockIdx.x + wave) * 64 + 32 + min(e, 31)] = __builtin_amdgcn_s_memtime();
    } else if constexpr (Cfg::TRACE)
      ((uint64_t*)p.ws)[(4 * blockIdx.x + wave) * 64 + min(e, 63)] = __builtin_amdgcn_s_memrealtime();
  };
  stamp(0);

  // ---- work of this workgroup: ntiles output tiles with logical ids lid0, lid0 + gridDim.x, ...
  // (PERSIST; otherwise one tile and one K split), each over K tiles kb .. kb + nkt - 1
  int lid0, ntiles = 1, split = 0, kb = 0, nkt = KT, full = 0, rest = 0;
  if constexpr (Cfg::PERSIST) {
    lid0 = Cfg::XCD_REMAP ? xcd_remap(blockIdx.x, gridDim.x) : blockIdx.x;
    ntiles = (tiles - lid0 + gridDim.x - 1) / gridDim.x;
    if constexpr (Cfg::TAIL_SPREAD) {
      full = tiles / gridDim.x;
      rest = tiles - full * gridDim.x;
      ntiles = full + (blockIdx.x < rest);
    }
  } else {
    const int splits = Cfg::SPLITK ? p.splits : 1;
    const int uid = Cfg::XCD_REMAP ? xcd_remap(blockIdx.x, tiles * splits) : blockIdx.x;
    const int kps = (KT + splits - 1) / splits;
    split = uid / tiles;
    lid0 = uid - split * tiles;
    kb = split * kps;
    nkt = min(KT - kb, kps);
  }
  auto tile_origin_id = [&](int lid, int& m0, int& n0) {  // logical tile id -> origin
    int tm, tn;
    tile_coords<Cfg::GROUP_M>(lid, tiles_m, tiles_n, tm, tn);
    m0 = tm * BM;
    n0 = tn * BN;
  };
  // TAIL_SPREAD: the partial last round goes to blocks 0 .. rest - 1, rest / 8 per XCD
  auto lid_of = [&](int t) {
    return Cfg::TAIL_SPREAD && t == full ? full * (int)gridDim.x + xcd_remap(blockIdx.x, rest) : lid0 + t * (int)gridDim.x;
  };
  auto tile_origin = [&](int t, int& m0, int& n0) { tile_origin_id(lid_of(t), m0, n0); };
  // a partial last K tile is processed first (not with PERSIST)
  const bool tail = !Cfg::PERSIST && p.K % BK != 0 && kb + nkt == KT;
  auto ktile = [&](int t) { return tail ? (t == 0 ? KT - 1 : kb + t - 1) : kb + t; };

  // ---- global -> LDS, as in gemm_pipe (load i of this wave: rows 8 (i NW + wave) .. + 7).
  // MN-major operand: load j of this wave covers K rows mn_row(wave) + mn_jrow(j) + lane / 16, lane
  // chunk lane % 16. Its buffer ends after the last K row (rows past K read as 0, so a partial K
  // tile needs no clearing); the bounds check works per dword, hence align(MN, 4).
  const uint32_t k16 = (uint32_t)(p.K + 15) & ~15u;
  const __amdgpu_buffer_rsrc_t ra = make_rsrc(p.A, Cfg::A_MN ? (uint32_t)(p.K - 1) * p.lda + ((p.M + 3) & ~3)
                                                             : (uint32_t)(p.M - 1) * p.lda + k16);
  const __amdgpu_buffer_rsrc_t rb = make_rsrc(p.B, Cfg::B_MN ? (uint32_t)(p.K - 1) * p.ldb + ((p.N + 3) & ~3)
                                                             : (uint32_t)(p.N - 1) * p.ldb + k16);
  const __amdgpu_buffer_rsrc_t rsa = make_rsrc(p.sfa, (uint32_t)((KT - 1) * p.lda_sf + p.M) * 4);
  const __amdgpu_buffer_rsrc_t rsb = make_rsrc(p.sfb, (uint32_t)((KT - 1) * p.ldb_sf + p.N) * 4);
  static_assert(NW == 4, "the keys of the rows of load i must not depend on i");
  const uint32_t mn_voff_row = (uint32_t)(lane / 16), mn_voff_chunk =
      (uint32_t)((lane % 16) ^ Cfg::keyMN(Cfg::mn_row(wave) + lane / 16)) * 16;
  const uint32_t a_voff = Cfg::A_MN ? mn_voff_row * p.lda + mn_voff_chunk
                                    : (uint32_t)(lane / 8) * p.lda + ((lane % 8) ^ Cfg::key(8 * wave + lane / 8)) * 16;
  const uint32_t b_voff = Cfg::B_MN ? mn_voff_row * p.ldb + mn_voff_chunk
                                    : (uint32_t)(lane / 8) * p.ldb + ((lane % 8) ^ Cfg::keyB(8 * wave + lane / 8)) * 16;
  const uint32_t a_step = 8 * NW * p.lda, b_step = 8 * NW * p.ldb;
  const uint32_t s_voff = (uint32_t)(64 * wave + lane) * 4;
  // L2PF: lane l of wave w touches row 64 w + l of the tile (Src already holds row 8 w)
  const uint32_t pf_a_voff = (uint32_t)(56 * wave + lane) * p.lda, pf_b_voff = (uint32_t)(56 * wave + lane) * p.ldb;
  auto touch = [&](__amdgpu_buffer_rsrc_t r, uint32_t voff, uint32_t soff) {
    __builtin_amdgcn_raw_ptr_buffer_load_lds(r, (lds_ptr_t)(smem + Cfg::PF_OFF + wave * 256), 4, voff, soff, 0, 0);
  };
  // the tile's part of the DMA source offsets
  struct Src {
    uint32_t a, b, sa, sb;
  };
  auto src_of_id = [&](int lid) {
    int m0, n0;
    tile_origin_id(lid, m0, n0);
    return Src{Cfg::A_MN ? (uint32_t)Cfg::mn_row(wave) * p.lda + m0 : (uint32_t)(m0 + 8 * wave) * p.lda,
               Cfg::B_MN ? (uint32_t)Cfg::mn_row(wave) * p.ldb + n0 : (uint32_t)(n0 + 8 * wave) * p.ldb,
               (uint32_t)m0 * 4, (uint32_t)n0 * 4};
  };
  auto src_of = [&](int t) { return src_of_id(lid_of(t)); };
  // DMA j of K tile kt (local index) of the tile with offsets `ld` into the stage at stage_off
  auto dma = [&](int j, uint32_t stage_off, const Src& ld, int kt) {
    uint8_t* st = smem + stage_off;
    const int kp = ktile(Cfg::K0_ONLY ? 0 : kt);
    if (j < Cfg::A_LOADS) {
      if constexpr (Cfg::A_MN)
        __builtin_amdgcn_raw_ptr_buffer_load_lds(
            ra, (lds_ptr_t)(st + Cfg::A_OFF + (Cfg::mn_row(wave) + Cfg::mn_jrow(j)) * 256), 16, a_voff,
            ld.a + (uint32_t)(kp * BK + Cfg::mn_jrow(j)) * p.lda, 0, 0);
      else
        __builtin_amdgcn_raw_ptr_buffer_load_lds(ra, (lds_ptr_t)(st + Cfg::A_OFF + (j * NW + wave) * 1024), 16,
                                                 a_voff, ld.a + j * a_step + kp * BK, 0, 0);
    } else if (j < Cfg::A_LOADS + Cfg::B_LOADS) {
      const int jb = j - Cfg::A_LOADS;
      if constexpr (Cfg::B_MN)
        __builtin_amdgcn_raw_ptr_buffer_load_lds(
            rb, (lds_ptr_t)(st + Cfg::B_OFF + (Cfg::mn_row(wave) + Cfg::mn_jrow(jb)) * 256), 16, b_voff,
            ld.b + (uint32_t)(kp * BK + Cfg::mn_jrow(jb)) * p.ldb, 0, 0);
      else
        __builtin_amdgcn_raw_ptr_buffer_load_lds(rb, (lds_ptr_t)(st + Cfg::B_OFF + (jb * NW + wave) * 1024), 16,
                                                 b_voff, ld.b + jb * b_step + kp * BK, 0, 0);
    } else if (j == Cfg::A_LOADS + Cfg::B_LOADS)
      __builtin_amdgcn_raw_ptr_buffer_load_lds(rsa, (lds_ptr_t)(st + Cfg::SA_OFF + wave * 256), 4, s_voff,
                                               ld.sa + kp * p.lda_sf * 4, 0, 0);
    else
      __builtin_amdgcn_raw_ptr_buffer_load_lds(rsb, (lds_ptr_t)(st + Cfg::SB_OFF + wave * 256), 4, s_voff,
                                               ld.sb + kp * p.ldb_sf * 4, 0, 0);
  };

  // ---- LDS -> registers. Fragment f of A: rows wm*128 + 16 f + r16; of B: rows wn*128 + bfrag(f) +
  // brow(r16). A lane reads logical chunks g and g + 4 of its row (physical = logical ^ key(row);
  // the key only depends on r16) and byte g of the row's scale word.
  const uint32_t a_rd = Cfg::A_OFF + (uint32_t)(wm * 128 + r16) * 128;
  const uint32_t b_rd = Cfg::B_OFF + (uint32_t)(wn * 128 + Cfg::brow(r16)) * 128;
  const uint32_t ca0 = (uint32_t)(g ^ Cfg::key(r16)) * 16, ca1 = (uint32_t)((g + 4) ^ Cfg::key(r16)) * 16;
  const uint32_t cb0 = (uint32_t)(g ^ Cfg::keyB(Cfg::brow(r16))) * 16;
  const uint32_t cb1 = (uint32_t)((g + 4) ^ Cfg::keyB(Cfg::brow(r16))) * 16;
  const uint32_t sa_rd = Cfg::SA_OFF + (uint32_t)(wm * 128 + r16) * 4 + g;
  const uint32_t sb_rd = Cfg::SB_OFF + (uint32_t)(wn * 128 + Cfg::brow(r16)) * 4 + g;
  // MN-major: fragment f covers the 16 MN rows of chunk 8 wm + f (A) / 8 wn + f (B). Transposing
  // read (h, s) gives lane (g, r16) the 8 bytes K = 64 h + 16 g + 8 s .. + 7 of its MN row r16;
  // lane r16 points at K row 64 h + 16 g + 8 s + r16 / 2, half r16 % 2 of the chunk. The key of
  // that row is (r16 / 2) | (g & 1) << 3 for every h, s, so fragment f is base ^ (f << 4) and
  // (h, s) is an immediate offset.
  const uint32_t mn_key = (uint32_t)((r16 >> 1) | ((g & 1) << 3));
  const uint32_t mn_rd = (uint32_t)(16 * g + (r16 >> 1)) * 256 + 8 * (r16 & 1);
  const uint32_t a_tr = Cfg::A_OFF + mn_rd + ((8u * wm) ^ mn_key) * 16;
  const uint32_t b_tr = Cfg::B_OFF + mn_rd + ((8u * wn) ^ mn_key) * 16;

  v4i fa[8][2], fb[8][2];  // operand halves
  uint32_t sa[8], sb[8];   // scale codes (in byte 0)
  if constexpr (Cfg::NO_SCALE)
    for (int f = 0; f < 8; ++f) sa[f] = sb[f] = 127;

  // read `sub` of fragment f of A (isB = false) or B: K-major 0, 1 operand halves, 2 scale code;
  // MN-major 0-3 transposing reads (h = sub / 2, s = sub % 2), 4 scale code
  auto rd_frag = [&](bool isB, int f, int sub, uint32_t stage_off) {
    const uint8_t* st = smem + stage_off;
    const bool mn = isB ? Cfg::B_MN : Cfg::A_MN;
    const int scale_sub = mn ? 4 : 2;
    if (sub == scale_sub) {
      if (Cfg::NO_SCALE) return;
      if (!isB) sa[f] = *(const uint8_t*)(st + sa_rd + f * 64);
      else sb[f] = *(const uint8_t*)(st + sb_rd + Cfg::bfrag(f) * 4);
      return;
    }
    v4i& dst = isB ? fb[f][mn ? sub / 2 : sub] : fa[f][mn ? sub / 2 : sub];
    if (mn) {
      const uint32_t addr = ((isB ? b_tr : a_tr) ^ (uint32_t)(f << 4)) + (64 * (sub / 2) + 8 * (sub % 2)) * 256;
      const v2i v = ds_read_tr8((const __attribute__((address_space(3))) v2i*)(st + addr));
      dst[2 * (sub % 2)] = v[0];
      dst[2 * (sub % 2) + 1] = v[1];
    } else if (!isB) {
      dst = *(const v4i*)(st + a_rd + f * 2048 + (sub ? ca1 : ca0));
    } else {
      dst = *(const v4i*)(st + b_rd + Cfg::bfrag(f) * 128 + (sub ? cb1 : cb0));
    }
  };
  auto rd = [&](int set, int r, uint32_t stage_off) {
    const typename Cfg::RdItem it = Cfg::rd_item(set, r);
    rd_frag(it.isB, it.f, it.sub, stage_off);
  };
  constexpr int READS = Cfg::READS, RPS = Cfg::RPS;

  v4f acc[8][8];
  if constexpr (!Cfg::PERSIST) {
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
      for (int j = 0; j < 8; ++j) acc[i][j] = v4f{};
  }

  // MFMA of slot t. B goes into the MFMA's A slot, so acc[i][j][e] of lane l is
  //   D[wm*128 + 16 i + l % 16][wn*128 + bfrag(j) + brow(4 (l / 16) + e)]  (one row per lane).
  // (PERSIST: the first step of a tile accumulates onto 0 instead of zeroing 256 registers)
  auto mfma = [&](int t, bool zero) {
    const int q = t / 16, j = (q >> 1) * 4 + (t % 16) / 4, i = (q & 1) * 4 + t % 4;
    acc[i][j] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(cat8(fb[j][0], fb[j][1]), cat8(fa[i][0], fa[i][1]),
                                                                 zero ? v4f{} : acc[i][j], kFmtFp8, kFmtFp8, 0, sb[j],
                                                                 0, sa[i]);
  };

  // One K step: K tile k of the tile with DMA offsets `src` (the next tile: `src_nx`); s counts the
  // steps of the workgroup. Step s sits in stage s & 1, and its iteration loads step s + 2 into the
  // same stage: K tile k + 2, or K tile k + 2 - nkt of the next tile (after the last tile: of the
  // same tile again, into a stage nobody reads). Selects, not branches: a branch in the K loop
  // costs the register allocator its coalescing (64 VGPR copies per iteration).
  // The last step of a tile does not read the next step's fragments, which would otherwise be
  // live across the epilogue.
  auto iter = [&](int s, int k, const Src& src, const Src& src_nx, auto last, auto first) {
    constexpr bool LAST = decltype(last)::value, FIRST = decltype(first)::value;
    const uint32_t cur = (s & 1) * Cfg::STAGE_BYTES, nxt = Cfg::STAGE_BYTES - cur;
    const bool same = k + 2 < nkt;
    const Src ld{same ? src.a : src_nx.a, same ? src.b : src_nx.b, same ? src.sa : src_nx.sa,
                 same ? src.sb : src_nx.sb};
    const int kl = same ? k + 2 : k + 2 - nkt;
    // L2PF: the K tile L2PF steps ahead (of this tile, or of the next one)
    const bool pf_same = k + Cfg::L2PF < nkt;
    const uint32_t pf_a = pf_same ? src.a : src_nx.a, pf_b = pf_same ? src.b : src_nx.b;
    const int pf_k = pf_same ? k + Cfg::L2PF : min(k + Cfg::L2PF - nkt, nkt - 1);
    static_for<0, 64>([&](auto T) {
      constexpr int t = decltype(T)::value;
      mfma(t, FIRST);
      if constexpr (!Cfg::NO_RD)  // A / B 4-7 of this step
        static_for<0, RPS>([&](auto U) {
          constexpr int r = RPS * t + decltype(U)::value;
          if constexpr (r < READS) rd(0, r, cur);
        });
      if constexpr (t == Cfg::BAR1 && !Cfg::NO_BAR) {
        if constexpr (Cfg::ONE_BAR)  // step s + 1 has landed (nothing of step s + 2 is issued yet) and stage cur is read
          asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)\n\ts_barrier" ::: "memory");
        else  // every wave has read all of stage cur
          asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
      }
      if constexpr (Cfg::dma_at(t) >= 0 && !Cfg::NO_DMA) dma(Cfg::dma_at(t), cur, ld, kl);
      if constexpr (t == Cfg::BAR2 && !Cfg::NO_BAR && !Cfg::ONE_BAR) {  // step s + 1 has landed (only the DMAs of step s + 2 may be in flight)
        __builtin_amdgcn_s_waitcnt(waitcnt_vm(Cfg::NO_DMA ? 0 : Cfg::dma_before_bar2()));
        asm volatile("s_barrier" ::: "memory");
      }
      // before the DMA window (the load path is idle there); barrier 2 of this step waits for them
      if constexpr (Cfg::L2PF && t == 2) touch(ra, pf_a_voff, pf_a + pf_k * BK);
      if constexpr (Cfg::L2PF && t == 6) touch(rb, pf_b_voff, pf_b + pf_k * BK);
      if constexpr (!LAST && !Cfg::NO_RD && t >= Cfg::BAR2)  // A / B 0-3 of the next step
        static_for<0, RPS>([&](auto U) {
          constexpr int r = RPS * (t - Cfg::BAR2) + decltype(U)::value;
          if constexpr (r < READS) rd(1, r, nxt);
        });
      __builtin_amdgcn_sched_barrier(0);
    });
  };

  // PERSIST epilogue of tile t, straight from the accumulators: lane (g, r16) owns columns
  // 32 p + 8 g .. + 7 of row 16 i + r16 (fragments 2p and 2p + 1).
  // Full tiles without C take buffer stores without any branch: tile origin + 16 i rows in soffset,
  // the column 32 p in the immediate offset, the lane's part in one VGPR.
  const bool vec_ok = (p.ldd * (int)sizeof(OutT)) % 16 == 0 && ((uintptr_t)p.D & 15) == 0 &&
                      ((uintptr_t)p.C & 15) == 0;
  const uint64_t out_bytes = (uint64_t)p.M * p.ldd * sizeof(OutT);
  const bool fast_ok = !Cfg::ONE_ROW && !p.C && vec_ok && out_bytes < (1ull << 32);
  const __amdgpu_buffer_rsrc_t rout = make_rsrc(p.D, fast_ok ? (uint32_t)out_bytes : 0u);
  const uint32_t out_lane = ((uint32_t)(wm * 128 + r16) * p.ldd + wn * 128 + 8 * g) * sizeof(OutT);
  auto store_tile_fast = [&](int m0, int n0, auto unit_alpha) {
    static_for<0, 8>([&](auto I) {
      constexpr int i = decltype(I)::value;
      const int soff = (int)(((uint32_t)(m0 + 16 * i) * p.ldd + n0) * sizeof(OutT));
      static_for<0, 4>([&](auto P) {
        constexpr int pp = decltype(P)::value;
        v4f lo = acc[i][2 * pp], hi = acc[i][2 * pp + 1];
        if constexpr (!decltype(unit_alpha)::value) {
          lo *= p.alpha;
          hi *= p.alpha;
        }
        constexpr uint32_t col = 32 * pp * sizeof(OutT);
        if constexpr (std::is_same<OutT, uint16_t>::value) {
          const v4u o = {pack_bf16x2(lo[0], lo[1]), pack_bf16x2(lo[2], lo[3]), pack_bf16x2(hi[0], hi[1]),
                         pack_bf16x2(hi[2], hi[3])};
          buffer_store16(o, rout, out_lane + col, soff);
        } else {
          buffer_store16(__builtin_bit_cast(v4u, lo), rout, out_lane + col, soff);
          buffer_store16(__builtin_bit_cast(v4u, hi), rout, out_lane + col + 16, soff);
        }
        __builtin_amdgcn_sched_barrier(0);
      });
    });
  };
  // EPI_LDS: round i stages rows 16 i .. + 15 of both wave rows (32 rows x 256 columns); wave w
  // then stores staging rows 8 w .. 8 w + 7, two per instruction (lane: row + lane / 32, 16 B
  // chunk lane % 32)
  // (without BPAIR, lane (g, r16) owns columns 16 j + 4 g .. + 3 of fragment j: 8 B per fragment)
  const uint32_t epi_wr = Cfg::EPI_OFF + (uint32_t)(wm * 16 + r16) * Cfg::EPI_PITCH +
                          (uint32_t)(wn * 128 + (Cfg::BPAIR ? 8 : 4) * g) * 2;
  const uint32_t epi_rd = Cfg::EPI_OFF + (uint32_t)(8 * wave + lane / 32) * Cfg::EPI_PITCH + (lane % 32) * 16;
  const uint32_t epi_out = ((uint32_t)((wave / 2) * 128 + 8 * (wave % 2) + lane / 32) * p.ldd + (lane % 32) * 8) * 2;
  auto store_tile_lds = [&](int m0, int n0, auto unit_alpha) {
    static_for<0, 8>([&](auto I) {
      constexpr int i = decltype(I)::value;
      if constexpr (Cfg::BPAIR) {
        static_for<0, 4>([&](auto P) {
          constexpr int pp = decltype(P)::value;
          v4f lo = acc[i][2 * pp], hi = acc[i][2 * pp + 1];
          if constexpr (!decltype(unit_alpha)::value) {
            lo *= p.alpha;
            hi *= p.alpha;
          }
          *(v4u*)(smem + epi_wr + 64 * pp) = v4u{pack_bf16x2(lo[0], lo[1]), pack_bf16x2(lo[2], lo[3]),
                                                 pack_bf16x2(hi[0], hi[1]), pack_bf16x2(hi[2], hi[3])};
        });
      } else {
        static_for<0, 8>([&](auto J) {
          constexpr int j = decltype(J)::value;
          v4f v = acc[i][j];
          if constexpr (!decltype(unit_alpha)::value) v *= p.alpha;
          *(v2u*)(smem + epi_wr + 32 * j) = v2u{pack_bf16x2(v[0], v[1]), pack_bf16x2(v[2], v[3])};
        });
      }
      asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
      static_for<0, 4>([&](auto S) {
        constexpr int s = decltype(S)::value;
        const v4u v = *(const v4u*)(smem + epi_rd + 2 * s * Cfg::EPI_PITCH);
        const int soff = (int)(((uint32_t)(m0 + 16 * i + 2 * s) * p.ldd + n0) * 2);
        buffer_store16(v, rout, epi_out, soff);
      });
      asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
    });
  };
  auto epilogue_direct = [&](int lid) {  // logical tile id
    int m0, n0;
    tile_origin_id(lid, m0, n0);
    if (!Cfg::NO_STORE && fast_ok && m0 + BM <= p.M && n0 + BN <= p.N) {
      if constexpr (Cfg::EPI_LDS && std::is_same<OutT, uint16_t>::value) {
        if (p.alpha == 1.f)
          store_tile_lds(m0, n0, std::true_type{});
        else
          store_tile_lds(m0, n0, std::false_type{});
        return;
      }
      if constexpr (Cfg::BPAIR) {
        if (p.alpha == 1.f)
          store_tile_fast(m0, n0, std::true_type{});
        else
          store_tile_fast(m0, n0, std::false_type{});
        return;
      }
    }
    if constexpr (Cfg::NO_STORE) {
      float s = 0.f;
#pragma unroll
      for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) s += acc[i][j][0] + acc[i][j][3];
      if (s == 1234.5f) ((float*)p.D)[tid] = s;
      return;
    }
    if constexpr (Cfg::BPAIR) {
#pragma unroll
      for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int pp = 0; pp < 4; ++pp) {
          store8<OutT>(p.D, p.C, p.ldd, p.M, p.N, m0 + (Cfg::ONE_ROW ? 0 : wm * 128 + 16 * i + r16),
                       n0 + wn * 128 + 32 * pp + 8 * g, acc[i][2 * pp] * p.alpha, acc[i][2 * pp + 1] * p.alpha, vec_ok);
          // keeps the scheduler from hoisting all 256 accumulator reads (the registers are full)
          __builtin_amdgcn_sched_barrier(0);
        }
    } else {
#pragma unroll
      for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          store4<OutT>(p.D, p.C, p.ldd, p.M, p.N, m0 + (Cfg::ONE_ROW ? 0 : wm * 128 + 16 * i + r16),
                       n0 + wn * 128 + 16 * j + 4 * g, acc[i][j] * p.alpha, vec_ok);
          __builtin_amdgcn_sched_barrier(0);
        }
    }
  };

  // ---- DYN: per-XCD tile queues. In every round of gridDim.x tiles, XCD x (= blockIdx.x % 8) owns
  // the block of logical ids xcd_remap gives it, and its queue walks that block round by round. The
  // first tile of a workgroup is its static one (round 0); a workgroup whose queue is empty steals
  // from the following XCDs. claim_issue sends the atomic, claim_publish (later, so that the wait for
  // it is free) turns it into a tile id (-1: none left) in LDS, with scalar arithmetic: in the
  // lane-0 branch the arithmetic is VALU, and its constants, live across the main loop, spill.
  int* const dyn_slot = (int*)(smem + Cfg::DYN_OFF);
  int dyn_c = 0;  // wave 0 lane 0: the result of the last claim
  // On a uniform address the atomic optimizer makes a wave reduction of the atomic, which waits for
  // the result on the spot (s_waitcnt vmcnt(0): also for the DMAs and stores in flight). An address
  // in a VGPR keeps the wait at the first use of the result.
  auto claim_atomic = [&](int x) {
    int zero;
    asm volatile("v_mov_b32 %0, 0" : "=v"(zero));
    return atomicAdd(p.tile_ctr + x + zero, 1);
  };
  auto claim_issue = [&]() {
    if constexpr (Cfg::DYN && Cfg::DYN_STATIC)
      ++dyn_c;
    else if constexpr (Cfg::DYN)
      if (wave == 0 && lane == 0) dyn_c = claim_atomic(blockIdx.x % 8);
  };
  auto claim_publish = [&]() {
    if constexpr (Cfg::DYN)
      if (wave == 0) {
        int nid;
        if constexpr (Cfg::DYN_STATIC) {
          nid = lid0 + dyn_c * (int)gridDim.x;
        } else {
          const int G = gridDim.x, X = blockIdx.x % 8, per = (G + 7) / 8, tall = G % 8 ? G % 8 : 8;
          auto qcnt = [&](int x) { return x < tall ? per : per - 1; };
          auto qtile = [&](int x, int c) {
            const int n = qcnt(x), r = c / n;
            return r * G + (x < tall ? x * per : tall * per + (x - tall) * (per - 1)) + c - r * n;
          };
          nid = qtile(X, qcnt(X) + __builtin_amdgcn_readfirstlane(dyn_c));
#pragma nounroll
          for (int i = 1; i < 8 && nid >= tiles; ++i) {
            const int x = (X + i) & 7;
            if (!qcnt(x)) continue;
            int c = 0;
            if (lane == 0) c = claim_atomic(x);
            nid = qtile(x, qcnt(x) + __builtin_amdgcn_readfirstlane(c));
          }
        }
        if (lane == 0) *dyn_slot = nid < tiles ? nid : -1;
      }
  };

  // ---- prologue: steps 0 and 1 in flight, wait for step 0, read its A / B 0-3
  claim_issue();
  Src src = src_of(0), src_nx = src_of(min(1, ntiles - 1));
#pragma unroll
  for (int j = 0; j < Cfg::DMA_PER_TILE; ++j) dma(j, 0, src, 0);
#pragma unroll
  for (int j = 0; j < Cfg::DMA_PER_TILE; ++j) dma(j, Cfg::STAGE_BYTES, nkt > 1 ? src : src_nx, nkt > 1 ? 1 : 0);
  __builtin_amdgcn_s_waitcnt(waitcnt_vm(Cfg::DMA_PER_TILE));
  claim_publish();
  asm volatile("s_barrier" ::: "memory");
  if (tail) {  // see gemm_pipe: zero the K overhang of the partial tile, padding scale codes = 1.0
    const int kv = p.K - (KT - 1) * BK, nb = (kv + 31) / 32;
    for (int r = tid; r < BM + BN; r += Cfg::NT) {
      uint8_t* row = smem + (r < BM ? Cfg::A_OFF + r * BK : Cfg::B_OFF + (r - BM) * BK);
      for (int c = kv / 16; c < 8 && !(r < BM ? Cfg::A_MN : Cfg::B_MN); ++c) {  // MN-major: K rows past K read as 0
        v4i* q = (v4i*)(row + (c ^ (r < BM ? Cfg::key(r) : Cfg::keyB(r - BM))) * 16);
        v4i v = *q;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int keep = kv - c * 16 - 4 * i;
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
  auto read_first = [&](uint32_t stage_off) {  // A / B 0-3 of the first step of a tile
    static_for<0, READS>([&](auto R) { rd(1, decltype(R)::value, stage_off); });
  };
  read_first(0);
  stamp(1);
  constexpr std::false_type not_last{}, not_first{};
  constexpr std::true_type last{}, first{};

  if constexpr (Cfg::DYN) {  // nkt >= 3 (launch_pipe16)
    // The id of tile t + 1 sits in LDS (*dyn_slot) from the epilogue of tile t - 1 (the prologue
    // for t = 0) and is read after K step 0 of tile t, whose barriers publish it. Wave 0 never
    // holds an atomic result across a K step: the main loop has no free VGPR for it (404 B of
    // spills). Held across the epilogue it still spills, and the spill store waits for it.
    int lid = lid0, s = 0;
    for (int t = 0;; ++t) {
      stamp(2 + 4 * t);
      iter(s++, 0, src, src, not_last, first);
      const int nid = __builtin_amdgcn_readfirstlane(*(volatile int*)dyn_slot);
      src_nx = nid >= 0 ? src_of_id(nid) : src;
      for (int k = 1; k < nkt - 1; ++k, ++s) iter(s, k, src, src_nx, not_last, not_first);
      stamp(3 + 4 * t);
      iter(s++, nkt - 1, src, src_nx, last, not_first);
      stamp(4 + 4 * t);
      if (nid >= 0) claim_issue();
      epilogue_direct(lid);
      if (nid >= 0) claim_publish();
      stamp(5 + 4 * t);
      if (nid < 0) break;
      lid = nid;
      src = src_nx;
      read_first((s & 1) * Cfg::STAGE_BYTES);
    }
    // the last workgroup to finish resets the counters for the next launch
    if (wave == 0 && lane == 0 && atomicAdd(&p.tile_ctr[8], 1) == (int)gridDim.x - 1)
      for (int i = 0; i < 9; ++i) atomicExch(&p.tile_ctr[i], 0);
    return;
  }

  if constexpr (Cfg::PERSIST) {  // nkt >= 2: steps s + 2 are at most one tile ahead
    int s = 0;
    for (int t = 0; t < ntiles; ++t) {
      if (t > 0) {
        src = src_nx;
        src_nx = src_of(min(t + 1, ntiles - 1));
      }
      stamp(2 + 4 * t);
      iter(s++, 0, src, src_nx, not_last, first);
      for (int k = 1; k < nkt - 1; ++k, ++s) iter(s, k, src, src_nx, not_last, not_first);
      stamp(3 + 4 * t);
      iter(s++, nkt - 1, src, src_nx, last, not_first);
      stamp(4 + 4 * t);
      epilogue_direct(lid_of(t));
      stamp(5 + 4 * t);
      if (t + 1 < ntiles) read_first((s & 1) * Cfg::STAGE_BYTES);
    }
    return;
  }

  for (int s = 0; s < nkt - 1; ++s) iter(s, s, src, src, not_last, not_first);
  iter(nkt - 1, nkt - 1, src, src, last, not_first);
  asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)\n\ts_barrier" ::: "memory");

  if constexpr (Cfg::NO_STORE) {
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
      for (int j = 0; j < 8; ++j) s += acc[i][j][0] + acc[i][j][3];
    if (s == 1234.5f) ((float*)p.D)[tid] = s;
    return;
  }

  // ---- epilogue: as in gemm_pipe, 32 rows (fragments 2q, 2q + 1) at a time through the wave's
  // LDS strip, then 16 B per lane and 4 rows x 128 columns per store instruction
  int m0, n0;
  tile_origin(0, m0, n0);
  using OutE = typename std::conditional<Cfg::SPLITK, float, OutT>::type;
  void* const out = Cfg::SPLITK ? (void*)(p.ws + (size_t)split * p.M * p.ldw) : p.D;
  const void* const cin = Cfg::SPLITK ? nullptr : p.C;
  const int ldo = Cfg::SPLITK ? p.ldw : p.ldd;
  const float alpha = Cfg::SPLITK ? 1.f : p.alpha;
  float* stg = (float*)(smem + wave * Cfg::EPI_WAVE_BYTES);
  const int er = lane / 16, ec = (lane % 16) * 8;
  const bool vec_ok_e = (ldo * (int)sizeof(OutE)) % 16 == 0 && ((uintptr_t)out & 15) == 0 &&
                        (!cin || ((uintptr_t)cin & 15) == 0);
#pragma unroll
  for (int q = 0; q < 4; ++q) {
#pragma unroll
    for (int ii = 0; ii < 2; ++ii)
#pragma unroll
      for (int j = 0; j < 8; ++j)
        *(v4f*)(stg + (16 * ii + r16) * Cfg::EPI_LD + 16 * j + 4 * g) = acc[2 * q + ii][j] * alpha;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const int r = er + 4 * k;
      const v4f lo = *(const v4f*)(stg + r * Cfg::EPI_LD + ec), hi = *(const v4f*)(stg + r * Cfg::EPI_LD + ec + 4);
      store8<OutE>(out, cin, ldo, p.M, p.N, m0 + wm * 128 + 32 * q + r, n0 + wn * 128 + ec, lo, hi, vec_ok_e);
    }
  }
#endif
}


namespace deep_gemm::cdna4 {
using ::Pipe16Cfg;
} // namespace deep_gemm::cdna4
