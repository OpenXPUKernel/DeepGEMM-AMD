// CDNA4 FP8 GEMM using 16x16x128 scaled MFMA instructions.
// The host normalizes operands to K-major and JIT-selects one execution mode.
#pragma once
#include <type_traits>
#include "epilogue.cuh"
#include "tile_scheduler.cuh"
// -----------------------------------------------------------------------------
// Compile-time unrolling
// -----------------------------------------------------------------------------
// f(integral_constant<int, I>) for I = B .. E - 1: a compile-time unrolled loop. The 64 MFMA slots of
// an iteration are too big for `#pragma unroll` to be reliable, and a rolled loop would put the
// register arrays in scratch.
template <int B, int E, class F> __device__ __forceinline__ void static_for(F &&f) {
  if constexpr (B < E) {
    f(std::integral_constant<int, B>{});
    static_for<B + 1, E>(f);
  }
}

// -----------------------------------------------------------------------------
// Pipeline configuration and LDS layout
// -----------------------------------------------------------------------------
// Modes emitted by the JIT. Persistent execution uses an LDS epilogue and
// spreads the final round of tiles; split-K writes FP32 partials for reduction.
enum class Fp8GemmMode { DataParallel, Persistent, SplitK };

template <Fp8GemmMode Mode> struct Fp8GemmConfig {
  static constexpr int BM = 256, BN = 256, BK = 128, NW = 4, NT = 256;
  static constexpr int GROUP_M = 8;
  static constexpr bool PERSIST = Mode == Fp8GemmMode::Persistent;
  static constexpr bool SPLITK = Mode == Fp8GemmMode::SplitK;
  // Pair B fragments in persistent mode so each lane owns contiguous columns.
  static constexpr bool BPAIR = PERSIST;
  static constexpr int A_OFF = 0, B_OFF = BM * BK;
  static constexpr int SA_OFF = B_OFF + BN * BK, SB_OFF = SA_OFF + BM * 4;
  static constexpr int STAGE_BYTES = SB_OFF + BN * 4;
  static constexpr int EPI_OFF = 2 * STAGE_BYTES, EPI_PITCH = 2 * BN + 16;
  static constexpr int LDS_BYTES = EPI_OFF + (PERSIST ? 32 * EPI_PITCH : 0);
  static constexpr int A_LOADS = BM / 8 / NW, B_LOADS = BN / 8 / NW;
  static constexpr int DMA_PER_TILE = A_LOADS + B_LOADS + 2;
  static constexpr int EPI_LD = 128 + 4, EPI_WAVE_BYTES = 32 * EPI_LD * 4;
  static_assert(NW * EPI_WAVE_BYTES <= LDS_BYTES, "epilogue staging must fit in the operand stages");

  // Two operand halves and one scale per fragment, spread across MFMA slots.
  static constexpr int FR_A = 3, FR_B = 3;
  static constexpr int READS = 4 * (FR_A + FR_B), RPS = 2, RD_SLOTS = READS / RPS;
  static constexpr int BAR1 = 16, BAR2 = 48, SPAN = 48;
  static_assert(BAR1 >= RD_SLOTS && BAR1 + SPAN <= 64, "invalid DMA schedule");
  static_assert(BAR2 + RD_SLOTS <= 64, "operand reads must fit after barrier 2");
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
      if (r < n)
        return RdItem{isB, f, r};
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
    for (int j = 0; j < DMA_PER_TILE; ++j)
      n += dma_slot(j) < BAR2;
    return n;
  }
  // B row (within the wave's 128) read by lane row r16 of B fragment j: bfrag(j) + brow(r16)
  __host__ __device__ static constexpr int bfrag(int j) {
    return BPAIR ? 32 * (j / 2) + 4 * (j % 2) : 16 * j;
  }
  __host__ __device__ static constexpr int brow(int r16) { return BPAIR ? 8 * (r16 / 4) + r16 % 4 : r16; }
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

// -----------------------------------------------------------------------------
// Main pipelined GEMM kernel
// -----------------------------------------------------------------------------
template <class Cfg, class OutT> __device__ __forceinline__ void fp8_gemm_impl(GemmArgs p) {
#if defined(__HIP_DEVICE_COMPILE__)
  constexpr int BM = Cfg::BM, BN = Cfg::BN, BK = Cfg::BK, NW = Cfg::NW;
  extern __shared__ __attribute__((aligned(1024))) uint8_t smem[];

  const int tid = threadIdx.x, lane = tid % 64, g = lane / 16, r16 = lane % 16;
  const int wave = __builtin_amdgcn_readfirstlane(tid / 64);
  const int wm = wave / 2, wn = wave % 2;
  const int tiles_m = (p.M + BM - 1) / BM, tiles_n = (p.N + BN - 1) / BN, tiles = tiles_m * tiles_n;
  const int KT = (p.K + BK - 1) / BK;
  // ---- work of this workgroup: ntiles output tiles with logical ids lid0, lid0 + gridDim.x, ...
  // (PERSIST; otherwise one tile and one K split), each over K tiles kb .. kb + nkt - 1
  int lid0, ntiles = 1, split = 0, kb = 0, nkt = KT, full = 0, rest = 0;
  if constexpr (Cfg::PERSIST) {
    lid0 = xcd_remap(blockIdx.x, gridDim.x);
    full = tiles / gridDim.x;
    rest = tiles - full * gridDim.x;
    ntiles = full + (blockIdx.x < rest);
  } else {
    const int splits = Cfg::SPLITK ? p.splits : 1;
    const int uid = xcd_remap(blockIdx.x, tiles * splits);
    const int kps = (KT + splits - 1) / splits;
    split = uid / tiles;
    lid0 = uid - split * tiles;
    kb = split * kps;
    nkt = min(KT - kb, kps);
  }
  auto tile_origin_id = [&](int lid, int &m0, int &n0) { // logical tile id -> origin
    int tm, tn;
    tile_coords<Cfg::GROUP_M>(lid, tiles_m, tiles_n, tm, tn);
    m0 = tm * BM;
    n0 = tn * BN;
  };
  // Spread the partial last round across the participating workgroups.
  auto lid_of = [&](int t) {
    return Cfg::PERSIST && t == full ? full * (int)gridDim.x + xcd_remap(blockIdx.x, rest)
                                     : lid0 + t * (int)gridDim.x;
  };
  auto tile_origin = [&](int t, int &m0, int &n0) { tile_origin_id(lid_of(t), m0, n0); };
  // a partial last K tile is processed first (not with PERSIST)
  const bool tail = !Cfg::PERSIST && p.K % BK != 0 && kb + nkt == KT;
  auto ktile = [&](int t) { return tail ? (t == 0 ? KT - 1 : kb + t - 1) : kb + t; };

  // ---- Global -> LDS. Inputs have been normalized to K-major by the host API.
  // Load i of this wave fills rows 8 * (i * NW + wave) .. + 7 of the tile.
  const uint32_t k16 = (uint32_t)(p.K + 15) & ~15u;
  const __amdgpu_buffer_rsrc_t ra = make_rsrc(p.A, (uint32_t)(p.M - 1) * p.lda + k16);
  const __amdgpu_buffer_rsrc_t rb = make_rsrc(p.B, (uint32_t)(p.N - 1) * p.ldb + k16);
  const __amdgpu_buffer_rsrc_t rsa = make_rsrc(p.sfa, (uint32_t)((KT - 1) * p.lda_sf + p.M) * 4);
  const __amdgpu_buffer_rsrc_t rsb = make_rsrc(p.sfb, (uint32_t)((KT - 1) * p.ldb_sf + p.N) * 4);
  static_assert(NW == 4, "the keys of the rows of load i must not depend on i");
  const uint32_t a_voff = (uint32_t)(lane / 8) * p.lda + ((lane % 8) ^ Cfg::key(8 * wave + lane / 8)) * 16;
  const uint32_t b_voff = (uint32_t)(lane / 8) * p.ldb + ((lane % 8) ^ Cfg::keyB(8 * wave + lane / 8)) * 16;
  const uint32_t a_step = 8 * NW * p.lda, b_step = 8 * NW * p.ldb;
  const uint32_t s_voff = (uint32_t)(64 * wave + lane) * 4;
  // the tile's part of the DMA source offsets
  struct Src {
    uint32_t a, b, sa, sb;
  };
  auto src_of_id = [&](int lid) {
    int m0, n0;
    tile_origin_id(lid, m0, n0);
    return Src{(uint32_t)(m0 + 8 * wave) * p.lda, (uint32_t)(n0 + 8 * wave) * p.ldb, (uint32_t)m0 * 4,
               (uint32_t)n0 * 4};
  };
  auto src_of = [&](int t) { return src_of_id(lid_of(t)); };
  // DMA j of K tile kt (local index) of the tile with offsets `ld` into the stage at stage_off
  auto dma = [&](int j, uint32_t stage_off, const Src &ld, int kt) {
    uint8_t *st = smem + stage_off;
    const int kp = ktile(kt);
    if (j < Cfg::A_LOADS) {
      __builtin_amdgcn_raw_ptr_buffer_load_lds(ra, (lds_ptr_t)(st + Cfg::A_OFF + (j * NW + wave) * 1024), 16,
                                               a_voff, ld.a + j * a_step + kp * BK, 0, 0);
    } else if (j < Cfg::A_LOADS + Cfg::B_LOADS) {
      const int jb = j - Cfg::A_LOADS;
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
  v4i fa[8][2], fb[8][2]; // operand halves
  uint32_t sa[8], sb[8];  // scale codes (in byte 0)

  // Read sub 0/1 of a fragment for its operand halves, or sub 2 for its scale.
  auto rd_frag = [&](bool isB, int f, int sub, uint32_t stage_off) {
    const uint8_t *st = smem + stage_off;
    if (sub == 2) {
      if (!isB)
        sa[f] = *(const uint8_t *)(st + sa_rd + f * 64);
      else
        sb[f] = *(const uint8_t *)(st + sb_rd + Cfg::bfrag(f) * 4);
      return;
    }
    v4i &dst = isB ? fb[f][sub] : fa[f][sub];
    if (!isB) {
      dst = *(const v4i *)(st + a_rd + f * 2048 + (sub ? ca1 : ca0));
    } else {
      dst = *(const v4i *)(st + b_rd + Cfg::bfrag(f) * 128 + (sub ? cb1 : cb0));
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
      for (int j = 0; j < 8; ++j)
        acc[i][j] = v4f{};
  }

  // MFMA of slot t. B goes into the MFMA's A slot, so acc[i][j][e] of lane l is
  //   D[wm*128 + 16 i + l % 16][wn*128 + bfrag(j) + brow(4 (l / 16) + e)]  (one row per lane).
  // (PERSIST: the first step of a tile accumulates onto 0 instead of zeroing 256 registers)
  auto mfma = [&](int t, bool zero) {
    const int q = t / 16, j = (q >> 1) * 4 + (t % 16) / 4, i = (q & 1) * 4 + t % 4;
    acc[i][j] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
        cat8(fb[j][0], fb[j][1]), cat8(fa[i][0], fa[i][1]), zero ? v4f{} : acc[i][j], kFmtFp8, kFmtFp8, 0,
        sb[j], 0, sa[i]);
  };

  // One K step: K tile k of the tile with DMA offsets `src` (the next tile: `src_nx`); s counts the
  // steps of the workgroup. Step s sits in stage s & 1, and its iteration loads step s + 2 into the
  // same stage: K tile k + 2, or K tile k + 2 - nkt of the next tile (after the last tile: of the
  // same tile again, into a stage nobody reads). Selects, not branches: a branch in the K loop
  // costs the register allocator its coalescing (64 VGPR copies per iteration).
  // The last step of a tile does not read the next step's fragments, which would otherwise be
  // live across the epilogue.
  auto iter = [&](int s, int k, const Src &src, const Src &src_nx, auto last, auto first) {
    constexpr bool LAST = decltype(last)::value, FIRST = decltype(first)::value;
    const uint32_t cur = (s & 1) * Cfg::STAGE_BYTES, nxt = Cfg::STAGE_BYTES - cur;
    const bool same = k + 2 < nkt;
    const Src ld{same ? src.a : src_nx.a, same ? src.b : src_nx.b, same ? src.sa : src_nx.sa,
                 same ? src.sb : src_nx.sb};
    const int kl = same ? k + 2 : k + 2 - nkt;
    static_for<0, 64>([&](auto T) {
      constexpr int t = decltype(T)::value;
      mfma(t, FIRST);
      // A / B fragments 4-7 of this step.
      static_for<0, RPS>([&](auto U) {
        constexpr int r = RPS * t + decltype(U)::value;
        if constexpr (r < READS)
          rd(0, r, cur);
      });
      if constexpr (t == Cfg::BAR1) {
        // Every wave has consumed the current stage; its storage can be refilled.
        asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
      }
      if constexpr (Cfg::dma_at(t) >= 0)
        dma(Cfg::dma_at(t), cur, ld, kl);
      if constexpr (t == Cfg::BAR2) { // step s + 1 has landed (only the DMAs of step s + 2 may be in flight)
        __builtin_amdgcn_s_waitcnt(waitcnt_vm(Cfg::dma_before_bar2()));
        asm volatile("s_barrier" ::: "memory");
      }
      if constexpr (!LAST && t >= Cfg::BAR2) // A / B 0-3 of the next step
        static_for<0, RPS>([&](auto U) {
          constexpr int r = RPS * (t - Cfg::BAR2) + decltype(U)::value;
          if constexpr (r < READS)
            rd(1, r, nxt);
        });
      __builtin_amdgcn_sched_barrier(0);
    });
  };

  // PERSIST epilogue of tile t, straight from the accumulators: lane (g, r16) owns columns
  // 32 p + 8 g .. + 7 of row 16 i + r16 (fragments 2p and 2p + 1).
  // Full tiles without C take buffer stores without any branch: tile origin + 16 i rows in soffset,
  // the column 32 p in the immediate offset, the lane's part in one VGPR.
  const bool vec_ok =
      (p.ldd * (int)sizeof(OutT)) % 16 == 0 && ((uintptr_t)p.D & 15) == 0 && ((uintptr_t)p.C & 15) == 0;
  const uint64_t out_bytes = (uint64_t)p.M * p.ldd * sizeof(OutT);
  const bool fast_ok = !p.C && vec_ok && out_bytes < (1ull << 32);
  const __amdgpu_buffer_rsrc_t rout = make_rsrc(p.D, fast_ok ? (uint32_t)out_bytes : 0u);
  // EPI_LDS: round i stages rows 16 i .. + 15 of both wave rows (32 rows x 256 columns); wave w
  // then stores staging rows 8 w .. 8 w + 7, two per instruction (lane: row + lane / 32, 16 B
  // chunk lane % 32)
  const uint32_t epi_wr =
      Cfg::EPI_OFF + (uint32_t)(wm * 16 + r16) * Cfg::EPI_PITCH + (uint32_t)(wn * 128 + 8 * g) * 2;
  const uint32_t epi_rd = Cfg::EPI_OFF + (uint32_t)(8 * wave + lane / 32) * Cfg::EPI_PITCH + (lane % 32) * 16;
  const uint32_t epi_out =
      ((uint32_t)((wave / 2) * 128 + 8 * (wave % 2) + lane / 32) * p.ldd + (lane % 32) * 8) * 2;
  auto store_tile_lds = [&](int m0, int n0, auto unit_alpha) {
    static_for<0, 8>([&](auto I) {
      constexpr int i = decltype(I)::value;
      static_for<0, 4>([&](auto P) {
        constexpr int pp = decltype(P)::value;
        v4f lo = acc[i][2 * pp], hi = acc[i][2 * pp + 1];
        if constexpr (!decltype(unit_alpha)::value) {
          lo *= p.alpha;
          hi *= p.alpha;
        }
        *(v4u *)(smem + epi_wr + 64 * pp) = v4u{pack_bf16x2(lo[0], lo[1]), pack_bf16x2(lo[2], lo[3]),
                                                pack_bf16x2(hi[0], hi[1]), pack_bf16x2(hi[2], hi[3])};
      });
      asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
      static_for<0, 4>([&](auto S) {
        constexpr int s = decltype(S)::value;
        const v4u v = *(const v4u *)(smem + epi_rd + 2 * s * Cfg::EPI_PITCH);
        const int soff = (int)(((uint32_t)(m0 + 16 * i + 2 * s) * p.ldd + n0) * 2);
        buffer_store16(v, rout, epi_out, soff);
      });
      asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
    });
  };
  auto epilogue_direct = [&](int lid) { // logical tile id
    int m0, n0;
    tile_origin_id(lid, m0, n0);
    if (fast_ok && m0 + BM <= p.M && n0 + BN <= p.N) {
      if constexpr (Cfg::PERSIST && std::is_same<OutT, uint16_t>::value) {
        if (p.alpha == 1.f)
          store_tile_lds(m0, n0, std::true_type{});
        else
          store_tile_lds(m0, n0, std::false_type{});
        return;
      }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
      for (int pp = 0; pp < 4; ++pp) {
        store8<OutT>(p.D, p.C, p.ldd, p.M, p.N, m0 + (wm * 128 + 16 * i + r16),
                     n0 + wn * 128 + 32 * pp + 8 * g, acc[i][2 * pp] * p.alpha, acc[i][2 * pp + 1] * p.alpha,
                     vec_ok);
        // keeps the scheduler from hoisting all 256 accumulator reads (the registers are full)
        __builtin_amdgcn_sched_barrier(0);
      }

  };

  // ---- prologue: steps 0 and 1 in flight, wait for step 0, read its A / B 0-3
  Src src = src_of(0), src_nx = src_of(min(1, ntiles - 1));
#pragma unroll
  for (int j = 0; j < Cfg::DMA_PER_TILE; ++j)
    dma(j, 0, src, 0);
#pragma unroll
  for (int j = 0; j < Cfg::DMA_PER_TILE; ++j)
    dma(j, Cfg::STAGE_BYTES, nkt > 1 ? src : src_nx, nkt > 1 ? 1 : 0);
  __builtin_amdgcn_s_waitcnt(waitcnt_vm(Cfg::DMA_PER_TILE));
  asm volatile("s_barrier" ::: "memory");
  if (tail) { // Zero the K overhang; padding scale codes represent 1.0.
    const int kv = p.K - (KT - 1) * BK, nb = (kv + 31) / 32;
    for (int r = tid; r < BM + BN; r += Cfg::NT) {
      uint8_t *row = smem + (r < BM ? Cfg::A_OFF + r * BK : Cfg::B_OFF + (r - BM) * BK);
      for (int c = kv / 16; c < 8; ++c) {
        v4i *q = (v4i *)(row + (c ^ (r < BM ? Cfg::key(r) : Cfg::keyB(r - BM))) * 16);
        v4i v = *q;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int keep = kv - c * 16 - 4 * i;
          v[i] = keep >= 4 ? v[i] : keep <= 0 ? 0 : (int)((uint32_t)v[i] & ((1u << (8 * keep)) - 1));
        }
        *q = v;
      }
      if (nb < 4) {
        uint32_t *w = (uint32_t *)(smem + (r < BM ? Cfg::SA_OFF + r * 4 : Cfg::SB_OFF + (r - BM) * 4));
        const uint32_t keep = (1u << (8 * nb)) - 1;
        *w = (*w & keep) | (0x7f7f7f7fu & ~keep);
      }
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
  }
  auto read_first = [&](uint32_t stage_off) { // A / B 0-3 of the first step of a tile
    static_for<0, READS>([&](auto R) { rd(1, decltype(R)::value, stage_off); });
  };
  read_first(0);
  constexpr std::false_type not_last{}, not_first{};
  constexpr std::true_type last{}, first{};

  if constexpr (Cfg::PERSIST) { // nkt >= 2: steps s + 2 are at most one tile ahead
    int s = 0;
    for (int t = 0; t < ntiles; ++t) {
      if (t > 0) {
        src = src_nx;
        src_nx = src_of(min(t + 1, ntiles - 1));
      }
      iter(s++, 0, src, src_nx, not_last, first);
      for (int k = 1; k < nkt - 1; ++k, ++s)
        iter(s, k, src, src_nx, not_last, not_first);
      iter(s++, nkt - 1, src, src_nx, last, not_first);
      epilogue_direct(lid_of(t));
      if (t + 1 < ntiles)
        read_first((s & 1) * Cfg::STAGE_BYTES);
    }
    return;
  }

  for (int s = 0; s < nkt - 1; ++s)
    iter(s, s, src, src, not_last, not_first);
  iter(nkt - 1, nkt - 1, src, src, last, not_first);
  asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)\n\ts_barrier" ::: "memory");

  // ---- epilogue: stage 32 rows (fragments 2q, 2q + 1) at a time through the wave's
  // LDS strip, then 16 B per lane and 4 rows x 128 columns per store instruction
  int m0, n0;
  tile_origin(0, m0, n0);
  using OutE = typename std::conditional<Cfg::SPLITK, float, OutT>::type;
  void *const out = Cfg::SPLITK ? (void *)(p.ws + (size_t)split * p.M * p.ldw) : p.D;
  const void *const cin = Cfg::SPLITK ? nullptr : p.C;
  const int ldo = Cfg::SPLITK ? p.ldw : p.ldd;
  const float alpha = Cfg::SPLITK ? 1.f : p.alpha;
  float *stg = (float *)(smem + wave * Cfg::EPI_WAVE_BYTES);
  const int er = lane / 16, ec = (lane % 16) * 8;
  const bool vec_ok_e = (ldo * (int)sizeof(OutE)) % 16 == 0 && ((uintptr_t)out & 15) == 0 &&
                        (!cin || ((uintptr_t)cin & 15) == 0);
#pragma unroll
  for (int q = 0; q < 4; ++q) {
#pragma unroll
    for (int ii = 0; ii < 2; ++ii)
#pragma unroll
      for (int j = 0; j < 8; ++j)
        *(v4f *)(stg + (16 * ii + r16) * Cfg::EPI_LD + 16 * j + 4 * g) = acc[2 * q + ii][j] * alpha;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const int r = er + 4 * k;
      const v4f lo = *(const v4f *)(stg + r * Cfg::EPI_LD + ec),
                hi = *(const v4f *)(stg + r * Cfg::EPI_LD + ec + 4);
      store8<OutE>(out, cin, ldo, p.M, p.N, m0 + wm * 128 + 32 * q + r, n0 + wn * 128 + ec, lo, hi, vec_ok_e);
    }
  }
#endif
}
