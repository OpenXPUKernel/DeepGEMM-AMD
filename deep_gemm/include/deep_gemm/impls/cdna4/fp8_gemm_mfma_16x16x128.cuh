// CDNA4 FP8 GEMM using 16x16x128 scaled MFMA instructions.
// Direct global-to-LDS loads feed two software-pipelined stages. MN-major
// operands stay in their original layout and use transposing LDS reads.
#pragma once
#include "epilogue.cuh"
#include "tile_scheduler.cuh"
#include <type_traits>

// f(integral_constant<int, I>) for I = B .. E - 1: a compile-time unrolled loop. The 64 MFMA slots
// of an iteration are too big for `#pragma unroll` to be reliable, and a rolled loop would put the
// register arrays in scratch.
template <int B, int E, class F> __device__ __forceinline__ void static_for(F &&f) {
  if constexpr (B < E) {
    f(std::integral_constant<int, B>{});
    static_for<B + 1, E>(f);
  }
}

enum class Mfma16x16x128GemmMode { DataParallel, Persistent, SplitK };

// The masks use bit 0 for A and bit 1 for B. MN_MAJOR selects transposing
// operand reads; SF_GRAN_128 selects one packed scale code per 128 K values.
template <Mfma16x16x128GemmMode Mode, int MN_MAJOR = 0, int SF_GRAN_128 = 0>
struct Mfma16x16x128GemmConfig {
  static constexpr int BM = 256, BN = 256, BK = 128, NW = 4, NT = 256;
  static constexpr int GROUP_M = 8;
  static constexpr bool PERSISTENT = Mode == Mfma16x16x128GemmMode::Persistent;
  static constexpr bool SPLIT_K = Mode == Mfma16x16x128GemmMode::SplitK;
  static constexpr bool A_MN_MAJOR = MN_MAJOR & 1, B_MN_MAJOR = MN_MAJOR & 2;
  static constexpr int SFA_SHIFT = SF_GRAN_128 & 1 ? 2 : 0;
  static constexpr int SFB_SHIFT = SF_GRAN_128 & 2 ? 2 : 0;
  // Paired B fragments give each lane eight adjacent output columns. A
  // transposing read requires sixteen adjacent rows and uses unpaired fragments.
  static constexpr bool PAIRED_B_FRAGMENTS = PERSISTENT && !B_MN_MAJOR;
  static constexpr int A_OFF = 0, B_OFF = BM * BK, SA_OFF = B_OFF + BN * BK,
                       SB_OFF = SA_OFF + BM * 4;
  static constexpr int STAGE_BYTES = SB_OFF + BN * 4;
  // Persistent epilogue buffer: 32 rows (16 of each wave row) x 256 bf16, rows padded to 528 B
  // against bank conflicts
  static constexpr int EPI_OFF = 2 * STAGE_BYTES, EPI_PITCH = 2 * BN + 16;
  static constexpr int LDS_BYTES = EPI_OFF + (PERSISTENT ? 32 * EPI_PITCH : 0);
  static_assert(LDS_BYTES <= 160 * 1024, "CDNA4 LDS capacity");
  // Each lane moves DMA_BYTES with buffer_load_dwordx4. K_LANES lanes cover
  // one K-major row; a wave covers K_ROWS rows. An MN-major row has BM bytes.
  static constexpr int WAVE_SIZE = 64, DMA_BYTES = 16, SCALE_BYTES = sizeof(int32_t);
  static constexpr int K_LANES = BK / DMA_BYTES, K_ROWS = WAVE_SIZE / K_LANES;
  static constexpr int MN_LANES = BM / DMA_BYTES, MN_ROWS = WAVE_SIZE / MN_LANES;
  static constexpr int K_WAVE_BYTES = K_ROWS * BK, SCALE_WAVE_BYTES = WAVE_SIZE * SCALE_BYTES;
  static constexpr int lg(int n) {
    int b = 0;
    while ((1 << b) < n)
      ++b;
    return b;
  }
  // ds_read_b64_tr_b8 reads TR_ROWS K rows (row bits 0 .. LG_TR-1). A fragment is two of those,
  // TR_ROWS apart, so bit HALF_BIT selects the half and keyMN ignores it. The next bit, KEY_BIT,
  // is the lane group. DMA index j uses only the bits keyMN does not read (HALF_BIT, then
  // KEY_BIT + 1 and up), which keeps a lane's swizzle key the same for every j.
  static constexpr int TR_ROWS = 8, LG_TR = lg(TR_ROWS), LG_MN_ROWS = lg(MN_ROWS);
  static constexpr int HALF_BIT = LG_TR, KEY_BIT = HALF_BIT + 1;
  static_assert(BN == BM && WAVE_SIZE % K_LANES == 0 && WAVE_SIZE % MN_LANES == 0,
                "A and B share the MN-major lane map");
  static_assert(LG_MN_ROWS + 1 == LG_TR && (1 << (LG_TR + 1)) == MN_LANES,
                "the lane takes bits 0..LG_MN_ROWS-1; one transpose bit is left for the wave, and "
                "the key covers every chunk");
  static constexpr int A_LOADS = BM / (K_ROWS * NW),
                       B_LOADS = BN / (K_ROWS * NW); // per wave, K_WAVE_BYTES
  static_assert(A_LOADS == BK / (NW * MN_ROWS), "MN-major K rows take one DMA per remaining group");
  static constexpr int DMA_PER_TILE = A_LOADS + B_LOADS + 2; // + one scale load for A and B
  static constexpr int EPI_LD = 128 + 4, EPI_WAVE_BYTES = 32 * EPI_LD * 4;
  static_assert(NW * EPI_WAVE_BYTES <= LDS_BYTES,
                "epilogue staging must fit in the operand stages");
  // slots (MFMA numbers 0..63 of an iteration) of the two barriers; the DMAs of tile kt + 2 are
  // spread evenly over the SPAN slots that follow barrier 1
  static constexpr int BAR1 = 16, BAR2 = 48, SPAN = 48;
  // LDS reads that bring one group of fragments into registers (A and B fragments 0-3, or 4-7):
  // 8 fragments x (2 operand halves + 1 scale code). A half is one ds_read_b128 (K-major) or two
  // ds_read_b64_tr_b8 (MN-major).
  static constexpr int READS_PER_FRAGMENT_GROUP = 24;
  static_assert(BAR1 >= READS_PER_FRAGMENT_GROUP / 2 && BAR1 + SPAN <= 64,
                "DMA window: after the reads of stage cur, within the iteration");
  static_assert(SPAN >= DMA_PER_TILE, "at most one DMA per slot");
  __host__ __device__ static constexpr int dma_slot(int j) {
    return BAR1 + j * SPAN / DMA_PER_TILE;
  }
  // slot that builds DMA descriptor i (A, B, scales of A, B): the one before its first DMA
  __host__ __device__ static constexpr int rsrc_slot(int i) {
    return dma_slot(i == 0 ? 0 : i == 1 ? A_LOADS : A_LOADS + B_LOADS + i - 2) - 1;
  }
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
    return PAIRED_B_FRAGMENTS ? 32 * (j / 2) + 4 * (j % 2) : 16 * j;
  }
  __host__ __device__ static constexpr int brow(int r16) {
    return PAIRED_B_FRAGMENTS ? 8 * (r16 / 4) + r16 % 4 : r16;
  }
  // MN-major tile. keyMN(row) is bits 0..LG_TR-1, plus KEY_BIT packed above them: the bits a
  // transposing read's lane fixes (r16 / 2, and the low bit of its lane group).
  __host__ __device__ static constexpr int keyMN(int r) {
    return (r & ((1 << LG_TR) - 1)) | (((r >> KEY_BIT) & 1) << LG_TR);
  }
  // DMA j (0 .. A_LOADS-1) of wave w loads MN_ROWS K rows x BM bytes, at base row mn_row(w) +
  // mn_jrow(j). The lane adds 0 .. MN_ROWS-1. Wave bit 0 takes the remaining transpose bit and
  // wave bit 1 takes KEY_BIT; j takes HALF_BIT and the bits above KEY_BIT.
  __host__ __device__ static constexpr int mn_row(int w) {
    return ((w & 1) << LG_MN_ROWS) | ((w >> 1) << KEY_BIT);
  }
  __host__ __device__ static constexpr int mn_jrow(int j) {
    return ((j & 1) << HALF_BIT) | ((j >> 1) << (KEY_BIT + 1));
  }
  // swizzle keys (physical chunk = logical chunk ^ key(row)). The 16 lanes that one ds_read_b128
  // cycle serves read rows r16 = 0..15 of a fragment and must hit 16 different 16 B bank groups,
  // i.e. different (row % 2, key) pairs. A: rows 0..15 -> (r >> 1) & 7. B with PAIRED_B_FRAGMENTS:
  // rows {8a + b : a, b < 4} -> bit 1 and bits 3-4 of the row.
  __host__ __device__ static constexpr int key(int r) { return (r >> 1) & 7; }
  __host__ __device__ static constexpr int keyB(int r) {
    return PAIRED_B_FRAGMENTS ? ((r >> 1) & 1) | (((r >> 3) & 3) << 1) : key(r);
  }
};

// The partial last K tile sits in stage 0 (it is processed first): only kv = K - (KT - 1) BK bytes
// of each row are data. Zero the rest (it holds the next row, padding, or 0 past the end of the
// buffer), and set the scale codes of K blocks without data to 127 (= 1.0) so that padding codes
// cannot turn 0 * scale into NaN. An MN-major operand needs no clearing: its K rows past K are
// outside its buffer and read as 0. With 128-wide scale blocks (fix_sa / fix_sb false) the tile has
// one code, and the other bytes of the word belong to other K tiles.
template <class Cfg>
__device__ __forceinline__ void clear_k_overhang(uint8_t *smem, int tid, int K, int KT,
                                                 bool fix_sa = true, bool fix_sb = true) {
  constexpr int BM = Cfg::BM, BN = Cfg::BN, BK = Cfg::BK;
  const int kv = K - (KT - 1) * BK, nb = (kv + 31) / 32;
  for (int r = tid; r < BM + BN; r += Cfg::NT) {
    uint8_t *row = smem + (r < BM ? Cfg::A_OFF + r * BK : Cfg::B_OFF + (r - BM) * BK);
    for (int c = kv / 16; c < 8 && !(r < BM ? Cfg::A_MN_MAJOR : Cfg::B_MN_MAJOR); ++c) {
      v4i *q =
          (v4i *)(row + (c ^ (r < BM ? Cfg::key(r) : Cfg::keyB(r - BM))) * 16); // logical chunk c
      v4i v = *q;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int keep = kv - c * 16 - 4 * i; // bytes of dword i that are data
        v[i] = keep >= 4 ? v[i] : keep <= 0 ? 0 : (int)((uint32_t)v[i] & ((1u << (8 * keep)) - 1));
      }
      *q = v;
    }
    if (nb < 4 && (r < BM ? fix_sa : fix_sb)) {
      uint32_t *w =
          (uint32_t *)(smem + (r < BM ? Cfg::SA_OFF + r * 4 : Cfg::SB_OFF + (r - BM) * 4));
      const uint32_t keep = (1u << (8 * nb)) - 1;
      *w = (*w & keep) | (0x7f7f7f7fu & ~keep);
    }
  }
  asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
}

template <class Cfg, class OutT>
__device__ __forceinline__ void fp8_gemm_mfma_16x16x128_impl(GemmArgs p) {
#if defined(__HIP_DEVICE_COMPILE__)
  constexpr int BM = Cfg::BM, BN = Cfg::BN, BK = Cfg::BK, NW = Cfg::NW;
  extern __shared__ __attribute__((aligned(1024))) uint8_t smem[];

  const int tid = threadIdx.x, lane = tid % 64, g = lane / 16, r16 = lane % 16;
  const int wave = __builtin_amdgcn_readfirstlane(tid / 64);
  const int wm = wave / 2, wn = wave % 2;
  const int tiles_m = (p.M + BM - 1) / BM, tiles_n = (p.N + BN - 1) / BN, tiles = tiles_m * tiles_n;
  const int KT = (p.K + BK - 1) / BK;
  // ---- work of this workgroup: ntiles output tiles with logical ids lid0, lid0 + gridDim.x, ...
  // (PERSISTENT; otherwise one tile and one K split), each over K tiles kb .. kb + nkt - 1
  int lid0, ntiles = 1, split = 0, kb = 0, nkt = KT, full = 0, rest = 0;
  if constexpr (Cfg::PERSISTENT) {
    lid0 = xcd_remap(blockIdx.x, gridDim.x);
    full = tiles / gridDim.x;
    rest = tiles - full * gridDim.x;
    ntiles = full + (blockIdx.x < rest);
  } else {
    const int splits = Cfg::SPLIT_K ? p.splits : 1;
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
    return Cfg::PERSISTENT && t == full ? full * (int)gridDim.x + xcd_remap(blockIdx.x, rest)
                                        : lid0 + t * (int)gridDim.x;
  };
  auto tile_origin = [&](int t, int &m0, int &n0) { tile_origin_id(lid_of(t), m0, n0); };
  // a partial last K tile is processed first (not with PERSISTENT)
  const bool tail = !Cfg::PERSISTENT && p.K % BK != 0 && kb + nkt == KT;
  auto ktile = [&](int t) { return tail ? (t == 0 ? KT - 1 : kb + t - 1) : kb + t; };

  // ---- Global -> LDS. Load i covers K_ROWS K-major rows starting at
  // K_ROWS * (i * NW + wave). For an MN-major operand, load j covers K rows
  // mn_row(wave) + mn_jrow(j) + lane / MN_LANES, chunk lane % MN_LANES. Its buffer ends after the
  // last K row (rows past K read as 0, so a partial K
  // tile needs no clearing); the bounds check works per dword, hence align(MN, 4).
  // A buffer access is a 64-bit base plus 32-bit offsets, and its size field is 32 bits. The large
  // part of an address goes into the base: the tile's origin, plus the K step for an MN-major
  // operand and for the scales; the offsets only span a tile (row strides below 16 MiB, checked by
  // the host API). The size counts from the base to the end of the matrix, so that reads past its
  // last row / K row still return 0. A base lies inside its matrix (the wave's rows go into the
  // offsets), so end - base > 0.
  // Rows and bases need no 16-byte alignment (the DMA takes any address), but the bounds check
  // works per dword from the base: the last row ends at align(K, 4), which a row stride that is a
  // multiple of 4 always contains. Past the last row these are 1-3 bytes beyond the matrix, which
  // stay in the aligned dword of its last byte because the host API requires a 4-byte aligned base.
  const uint32_t k4 = (uint32_t)(p.K + 3) & ~3u;
  const uint8_t *const a_end = p.A + (Cfg::A_MN_MAJOR ? (size_t)(p.K - 1) * p.lda + ((p.M + 3) & ~3)
                                                      : (size_t)(p.M - 1) * p.lda + k4);
  const uint8_t *const b_end = p.B + (Cfg::B_MN_MAJOR ? (size_t)(p.K - 1) * p.ldb + ((p.N + 3) & ~3)
                                                      : (size_t)(p.N - 1) * p.ldb + k4);
  // scale words: one per K tile, or (128-wide scale blocks, shift 2) one per 4 K tiles
  const int sa_words = (KT + (1 << Cfg::SFA_SHIFT) - 1) >> Cfg::SFA_SHIFT;
  const int sb_words = (KT + (1 << Cfg::SFB_SHIFT) - 1) >> Cfg::SFB_SHIFT;
  const uint8_t *const sa_end =
      (const uint8_t *)p.sfa + ((size_t)(sa_words - 1) * p.lda_sf + p.M) * 4;
  const uint8_t *const sb_end =
      (const uint8_t *)p.sfb + ((size_t)(sb_words - 1) * p.ldb_sf + p.N) * 4;
  // size = min(end - base, 2^32 - 1) from the high word. 64-bit compares are VALU only, and the
  // compiler turns hi != 0 back into one unless hi is opaque.
  auto rsrc_at = [&](const uint8_t *base, const uint8_t *end) {
    const uint64_t n = (uint64_t)end - (uint64_t)base;
    uint32_t hi = (uint32_t)(n >> 32);
    asm("" : "+s"(hi));
    return make_rsrc(base, hi ? 0xffffffffu : (uint32_t)n);
  };
  static_assert(NW == 4, "the keys of the rows of load i must not depend on i");
  constexpr int k_rows = Cfg::K_ROWS, k_lanes = Cfg::K_LANES, mn_lanes = Cfg::MN_LANES,
                load_bytes = Cfg::DMA_BYTES;
  const uint32_t mn_voff_row = (uint32_t)(Cfg::mn_row(wave) + lane / mn_lanes);
  const uint32_t mn_voff_chunk =
      (uint32_t)((lane % mn_lanes) ^ Cfg::keyMN((int)mn_voff_row)) * load_bytes;
  const uint32_t k_row = (uint32_t)(k_rows * wave + lane / k_lanes);
  const uint32_t a_voff =
      Cfg::A_MN_MAJOR ? mn_voff_row * p.lda + mn_voff_chunk
                      : k_row * p.lda + ((lane % k_lanes) ^ Cfg::key((int)k_row)) * load_bytes;
  const uint32_t b_voff =
      Cfg::B_MN_MAJOR ? mn_voff_row * p.ldb + mn_voff_chunk
                      : k_row * p.ldb + ((lane % k_lanes) ^ Cfg::keyB((int)k_row)) * load_bytes;
  const uint32_t a_step = k_rows * NW * p.lda, b_step = k_rows * NW * p.ldb;
  const uint32_t s_voff = (Cfg::WAVE_SIZE * wave + lane) * Cfg::SCALE_BYTES;
  // a tile's origin. The DMA bases are built from it per K step: kept 64-bit for two tiles (the
  // current and the next), they cost the persistent kernel SGPR spills.
  struct Src {
    int m0, n0;
  };
  auto src_of_id = [&](int lid) {
    Src s;
    tile_origin_id(lid, s.m0, s.n0);
    return s;
  };
  auto src_of = [&](int t) { return src_of_id(lid_of(t)); };
  // descriptor i (0: A, 1: B, 2, 3: their scales) of K tile kt (local index) of tile `ld`. Base:
  // the tile's first row (K-major) / column, plus K row kp BK (MN-major) or the scale word of kp.
  struct Rsrc {
    __amdgpu_buffer_rsrc_t r[4];
  };
  auto kp_of = [&](int kt) { return ktile(kt); };
  auto step_rsrc = [&](int i, const Src &ld, int kt) {
    const int kp = kp_of(kt);
    if (i == 0)
      return rsrc_at(
          p.A + (Cfg::A_MN_MAJOR ? (size_t)(kp * BK) * p.lda + ld.m0 : (size_t)ld.m0 * p.lda),
          a_end);
    if (i == 1)
      return rsrc_at(
          p.B + (Cfg::B_MN_MAJOR ? (size_t)(kp * BK) * p.ldb + ld.n0 : (size_t)ld.n0 * p.ldb),
          b_end);
    if (i == 2)
      return rsrc_at(
          (const uint8_t *)p.sfa + ((size_t)(kp >> Cfg::SFA_SHIFT) * p.lda_sf + ld.m0) * 4, sa_end);
    return rsrc_at((const uint8_t *)p.sfb + ((size_t)(kp >> Cfg::SFB_SHIFT) * p.ldb_sf + ld.n0) * 4,
                   sb_end);
  };
  auto rsrc_of = [&](const Src &ld, int kt) {
    return Rsrc{
        {step_rsrc(0, ld, kt), step_rsrc(1, ld, kt), step_rsrc(2, ld, kt), step_rsrc(3, ld, kt)}};
  };
  // DMA j of K tile kt (local index) with descriptors rs into the stage at stage_off
  auto dma = [&](int j, uint32_t stage_off, const Rsrc &rs, int kt) {
    uint8_t *st = smem + stage_off;
    const int kp = ktile(kt);
    // MN-major soffset: uniform, but the compiler loses track of it (K row x ld ends up in a VGPR
    // and every DMA gets a waterfall loop), hence readfirstlane
    if (j < Cfg::A_LOADS) {
      if constexpr (Cfg::A_MN_MAJOR)
        __builtin_amdgcn_raw_ptr_buffer_load_lds(
            rs.r[0], (lds_ptr_t)(st + Cfg::A_OFF + (Cfg::mn_row(wave) + Cfg::mn_jrow(j)) * Cfg::BM),
            Cfg::DMA_BYTES, a_voff, __builtin_amdgcn_readfirstlane(Cfg::mn_jrow(j) * p.lda), 0, 0);
      else
        __builtin_amdgcn_raw_ptr_buffer_load_lds(
            rs.r[0], (lds_ptr_t)(st + Cfg::A_OFF + (j * NW + wave) * Cfg::K_WAVE_BYTES),
            Cfg::DMA_BYTES, a_voff, j * a_step + kp * BK, 0, 0);
    } else if (j < Cfg::A_LOADS + Cfg::B_LOADS) {
      const int jb = j - Cfg::A_LOADS;
      if constexpr (Cfg::B_MN_MAJOR)
        __builtin_amdgcn_raw_ptr_buffer_load_lds(
            rs.r[1],
            (lds_ptr_t)(st + Cfg::B_OFF + (Cfg::mn_row(wave) + Cfg::mn_jrow(jb)) * Cfg::BN),
            Cfg::DMA_BYTES, b_voff, __builtin_amdgcn_readfirstlane(Cfg::mn_jrow(jb) * p.ldb), 0, 0);
      else
        __builtin_amdgcn_raw_ptr_buffer_load_lds(
            rs.r[1], (lds_ptr_t)(st + Cfg::B_OFF + (jb * NW + wave) * Cfg::K_WAVE_BYTES),
            Cfg::DMA_BYTES, b_voff, jb * b_step + kp * BK, 0, 0);
    } else if (j == Cfg::A_LOADS + Cfg::B_LOADS)
      __builtin_amdgcn_raw_ptr_buffer_load_lds(
          rs.r[2], (lds_ptr_t)(st + Cfg::SA_OFF + wave * Cfg::SCALE_WAVE_BYTES), Cfg::SCALE_BYTES,
          s_voff, 0, 0, 0);
    else
      __builtin_amdgcn_raw_ptr_buffer_load_lds(
          rs.r[3], (lds_ptr_t)(st + Cfg::SB_OFF + wave * Cfg::SCALE_WAVE_BYTES), Cfg::SCALE_BYTES,
          s_voff, 0, 0, 0);
  };

  // ---- LDS -> registers. Fragment f of A: rows wm*128 + 16 f + r16; of B: rows wn*128 + bfrag(f)
  // + brow(r16). A lane reads logical chunks g (half 0) and g + 4 (half 1) of its row (physical =
  // logical ^ key(row); the key only depends on r16) and byte g of the row's scale word: the code
  // of K block g. With 128-wide scale blocks all lanes read byte kp % 4, the code of K tile kp,
  // which goes into the (uniform) stage offset of the scale reads (scale_byte_offsets).
  const uint32_t a_row_offset = Cfg::A_OFF + (uint32_t)(wm * 128 + r16) * 128;
  const uint32_t b_row_offset = Cfg::B_OFF + (uint32_t)(wn * 128 + Cfg::brow(r16)) * 128;
  const uint32_t a_half0_offset = (uint32_t)(g ^ Cfg::key(r16)) * 16;
  const uint32_t a_half1_offset = (uint32_t)((g + 4) ^ Cfg::key(r16)) * 16;
  const uint32_t b_half0_offset = (uint32_t)(g ^ Cfg::keyB(Cfg::brow(r16))) * 16;
  const uint32_t b_half1_offset = (uint32_t)((g + 4) ^ Cfg::keyB(Cfg::brow(r16))) * 16;
  const uint32_t a_scale_offset =
      Cfg::SA_OFF + (uint32_t)(wm * 128 + r16) * 4 + (Cfg::SFA_SHIFT ? 0 : g);
  const uint32_t b_scale_offset =
      Cfg::SB_OFF + (uint32_t)(wn * 128 + Cfg::brow(r16)) * 4 + (Cfg::SFB_SHIFT ? 0 : g);
  // byte of A's and B's scale words that holds the code of local K tile kt; 0 for 32-wide scale
  // blocks
  struct ScaleByteOffsets {
    uint32_t a, b;
  };
  auto scale_byte_offsets = [&](int kt) {
    const uint32_t kp = Cfg::SFA_SHIFT || Cfg::SFB_SHIFT ? (uint32_t)kp_of(kt) & 3 : 0u;
    return ScaleByteOffsets{Cfg::SFA_SHIFT ? kp : 0u, Cfg::SFB_SHIFT ? kp : 0u};
  };
  // MN-major: fragment f covers the 16 MN rows of chunk 8 wm + f (A) / 8 wn + f (B). Transposing
  // read (h, s) gives lane (g, r16) the 8 bytes K = 64 h + 16 g + 8 s .. + 7 of its MN row r16
  // (half h = reads s = 0, 1); lane r16 points at K row 64 h + 16 g + 8 s + r16 / 2, half r16 % 2
  // of the chunk. The key of that row is (r16 / 2) | (g & 1) << 3 for every h, s, so fragment f is
  // base ^ (f << 4) and (h, s) is an immediate offset.
  const uint32_t mn_key = (uint32_t)((r16 >> 1) | ((g & 1) << 3));
  const uint32_t mn_lane_offset = (uint32_t)(16 * g + (r16 >> 1)) * 256 + 8 * (r16 & 1);
  const uint32_t a_transpose_base = Cfg::A_OFF + mn_lane_offset + ((8u * wm) ^ mn_key) * 16;
  const uint32_t b_transpose_base = Cfg::B_OFF + mn_lane_offset + ((8u * wn) ^ mn_key) * 16;

  v4i a_fragments[8][2], b_fragments[8][2];    // [fragment][half]: the MFMA operands
  uint32_t a_scale_codes[8], b_scale_codes[8]; // [fragment]: the lane's scale code, in byte 0
  // One operand half of an MN-major fragment, assembled whole from its two transposing reads:
  // writing half of a fragment that is still live costs registers.
  // The XOR goes after the stage offset (whose bits 4-6 are 0): base ^ (f << 4) alone is loop
  // invariant, and the compiler keeps all 8 of them in registers.
  static_assert((Cfg::STAGE_BYTES & 0x70) == 0, "the stage offset must not touch the chunk bits");
  auto read_transposed_half = [&](uint32_t stage_offset, uint32_t base, int fragment, int half) {
    const uint32_t addr = ((base + stage_offset) ^ (uint32_t)(fragment << 4)) + 64 * half * 256;
    const v2i lo = ds_read_tr8((const __attribute__((address_space(3))) v2i *)(smem + addr));
    const v2i hi =
        ds_read_tr8((const __attribute__((address_space(3))) v2i *)(smem + addr + 8 * 256));
    return v4i{lo[0], lo[1], hi[0], hi[1]};
  };
  // One LDS read for fragment f of A (is_b = false) or B: part 0 and 1 are its operand halves,
  // part 2 its scale code (the byte of the scale word that scale_bytes picks for the K tile).
  auto read_fragment_part = [&](bool is_b, int f, int part, uint32_t stage_offset,
                                const ScaleByteOffsets &scale_bytes) {
    const uint8_t *stage = smem + stage_offset;
    if (!is_b) {
      if (part < 2)
        a_fragments[f][part] = Cfg::A_MN_MAJOR
                                   ? read_transposed_half(stage_offset, a_transpose_base, f, part)
                                   : *(const v4i *)(stage + a_row_offset + f * 2048 +
                                                    (part ? a_half1_offset : a_half0_offset));
      else
        a_scale_codes[f] =
            *(const uint8_t *)(smem + (stage_offset + scale_bytes.a) + a_scale_offset + f * 64);
    } else {
      if (part < 2)
        b_fragments[f][part] = Cfg::B_MN_MAJOR
                                   ? read_transposed_half(stage_offset, b_transpose_base, f, part)
                                   : *(const v4i *)(stage + b_row_offset + Cfg::bfrag(f) * 128 +
                                                    (part ? b_half1_offset : b_half0_offset));
      else
        b_scale_codes[f] = *(const uint8_t *)(smem + (stage_offset + scale_bytes.b) +
                                              b_scale_offset + Cfg::bfrag(f) * 4);
    }
  };
  // Read i (0 .. READS_PER_FRAGMENT_GROUP - 1) of fragments 4-7 of this K step, in the order A 4-7,
  // B 4-7: part i % 3 of the (i / 3)-th fragment of that list.
  auto read_fragments_4_to_7 = [&](int i, uint32_t stage_offset,
                                   const ScaleByteOffsets &scale_bytes) {
    const int item = i / 3;
    read_fragment_part(item >= 4, 4 + item % 4, i % 3, stage_offset, scale_bytes);
  };
  // Read i of fragments 0-3 of a K step, in the order B 0, A 0-3, B 1-3: the order in which the
  // first quadrant of MFMAs (B 0-3 x A 0-3, B 0 first) needs them.
  auto read_fragments_0_to_3 = [&](int i, uint32_t stage_offset,
                                   const ScaleByteOffsets &scale_bytes) {
    const int item = i / 3;
    read_fragment_part(item == 0 || item >= 5,
                       item == 0   ? 0
                       : item <= 4 ? item - 1
                                   : item - 4,
                       i % 3, stage_offset, scale_bytes);
  };

  v4f acc[8][8];
  if constexpr (!Cfg::PERSISTENT) {
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
      for (int j = 0; j < 8; ++j)
        acc[i][j] = v4f{};
  }

  // MFMA of slot t. B goes into the MFMA's A slot, so acc[i][j][e] of lane l is
  //   D[wm*128 + 16 i + l % 16][wn*128 + bfrag(j) + brow(4 (l / 16) + e)]  (one row per lane).
  // (PERSISTENT: the first step of a tile accumulates onto 0 instead of zeroing 256 registers)
  auto mfma = [&](int t, bool zero) {
    const int q = t / 16, j = (q >> 1) * 4 + (t % 16) / 4, i = (q & 1) * 4 + t % 4;
    acc[i][j] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
        cat8(b_fragments[j][0], b_fragments[j][1]), cat8(a_fragments[i][0], a_fragments[i][1]),
        zero ? v4f{} : acc[i][j], kFmtFp8, kFmtFp8, 0, b_scale_codes[j], 0, a_scale_codes[i]);
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
    const Src ld{same ? src.m0 : src_nx.m0, same ? src.n0 : src_nx.n0};
    const int kl = same ? k + 2 : k + 2 - nkt;
    // scale bytes of this step's K tile and of the next step's (the next tile's first: local 0)
    const ScaleByteOffsets scale_bytes_this_step = scale_byte_offsets(k);
    const ScaleByteOffsets scale_bytes_next_step = scale_byte_offsets(k + 1 < nkt ? k + 1 : 0);
    // DMA descriptor i is built in rsrc_slot(i), the slot before its first DMA: its chain of
    // dependent scalar ops delays the MFMA of its slot, and in a DMA slot that costs ~2%
    Rsrc rs;
    static_for<0, 64>([&](auto T) {
      constexpr int t = decltype(T)::value;
      mfma(t, FIRST);
      if constexpr (t < Cfg::READS_PER_FRAGMENT_GROUP / 2) { // A / B 4-7 of this step
        read_fragments_4_to_7(2 * t, cur, scale_bytes_this_step);
        read_fragments_4_to_7(2 * t + 1, cur, scale_bytes_this_step);
      }
      static_for<0, 4>([&](auto I) {
        constexpr int i = decltype(I)::value;
        if constexpr (t == Cfg::rsrc_slot(i))
          rs.r[i] = step_rsrc(i, ld, kl);
      });
      if constexpr (t == Cfg::BAR1) // every wave has read all of stage cur
        asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
      if constexpr (Cfg::dma_at(t) >= 0)
        dma(Cfg::dma_at(t), cur, rs, kl);
      if constexpr (t == Cfg::BAR2) { // step s + 1 has landed (only the DMAs of step s + 2 may be
                                      // in flight)
        __builtin_amdgcn_s_waitcnt(waitcnt_vm(Cfg::dma_before_bar2()));
        asm volatile("s_barrier" ::: "memory");
      }
      if constexpr (!LAST && t >= Cfg::BAR2 &&
                    t < Cfg::BAR2 +
                            Cfg::READS_PER_FRAGMENT_GROUP / 2) { // A / B 0-3 of the next step
        read_fragments_0_to_3(2 * (t - Cfg::BAR2), nxt, scale_bytes_next_step);
        read_fragments_0_to_3(2 * (t - Cfg::BAR2) + 1, nxt, scale_bytes_next_step);
      }
      __builtin_amdgcn_sched_barrier(0);
    });
  };

  // Persistent epilogue. Full tiles stage BF16 values through LDS for coalesced
  // stores; edge tiles use bounds-checked stores from each lane's accumulators.
  const bool vec_ok = (p.ldd * (int)sizeof(OutT)) % 16 == 0 && ((uintptr_t)p.D & 15) == 0 &&
                      ((uintptr_t)p.C & 15) == 0;
  const bool fast_ok = !p.C && vec_ok;
  const uint8_t *const d_end = (const uint8_t *)p.D + (size_t)p.M * p.ldd * sizeof(OutT);
  auto rout_at = [&](int m0) {
    return rsrc_at((const uint8_t *)p.D + (size_t)m0 * p.ldd * sizeof(OutT), d_end);
  };
  // Persistent LDS epilogue: round i stages rows 16 i .. + 15 of both wave rows (32 rows x 256
  // columns); wave w then stores staging rows 8 w .. 8 w + 7, two per instruction (lane: row +
  // lane / 32, 16 B chunk lane % 32)
  // (without PAIRED_B_FRAGMENTS, lane (g, r16) owns columns 16 j + 4 g .. + 3 of fragment j: 8 B
  // per fragment)
  const uint32_t epi_wr = Cfg::EPI_OFF + (uint32_t)(wm * 16 + r16) * Cfg::EPI_PITCH +
                          (uint32_t)(wn * 128 + (Cfg::PAIRED_B_FRAGMENTS ? 8 : 4) * g) * 2;
  const uint32_t epi_rd =
      Cfg::EPI_OFF + (uint32_t)(8 * wave + lane / 32) * Cfg::EPI_PITCH + (lane % 32) * 16;
  const uint32_t epi_out =
      ((uint32_t)((wave / 2) * 128 + 8 * (wave % 2) + lane / 32) * p.ldd + (lane % 32) * 8) * 2;
  auto store_tile_lds = [&](int m0, int n0, auto unit_alpha) {
    const __amdgpu_buffer_rsrc_t rout = rout_at(m0);
    static_for<0, 8>([&](auto I) {
      constexpr int i = decltype(I)::value;
      if constexpr (Cfg::PAIRED_B_FRAGMENTS) {
        static_for<0, 4>([&](auto P) {
          constexpr int pp = decltype(P)::value;
          v4f lo = acc[i][2 * pp], hi = acc[i][2 * pp + 1];
          if constexpr (!decltype(unit_alpha)::value) {
            lo *= p.alpha;
            hi *= p.alpha;
          }
          *(v4u *)(smem + epi_wr + 64 * pp) =
              v4u{pack_bf16x2(lo[0], lo[1]), pack_bf16x2(lo[2], lo[3]), pack_bf16x2(hi[0], hi[1]),
                  pack_bf16x2(hi[2], hi[3])};
        });
      } else {
        static_for<0, 8>([&](auto J) {
          constexpr int j = decltype(J)::value;
          v4f v = acc[i][j];
          if constexpr (!decltype(unit_alpha)::value)
            v *= p.alpha;
          *(v2u *)(smem + epi_wr + 32 * j) = v2u{pack_bf16x2(v[0], v[1]), pack_bf16x2(v[2], v[3])};
        });
      }
      asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
      static_for<0, 4>([&](auto S) {
        constexpr int s = decltype(S)::value;
        const v4u v = *(const v4u *)(smem + epi_rd + 2 * s * Cfg::EPI_PITCH);
        const int soff = (int)(((uint32_t)(16 * i + 2 * s) * p.ldd + n0) * 2);
        buffer_store16(v, rout, epi_out, soff);
      });
      asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
    });
  };
  auto store_persistent_tile = [&](int lid) { // logical tile id
    int m0, n0;
    tile_origin_id(lid, m0, n0);
    if (fast_ok && m0 + BM <= p.M && n0 + BN <= p.N) {
      if constexpr (Cfg::PERSISTENT && std::is_same<OutT, uint16_t>::value) {
        if (p.alpha == 1.f)
          store_tile_lds(m0, n0, std::true_type{});
        else
          store_tile_lds(m0, n0, std::false_type{});
        return;
      }
    }
    if constexpr (Cfg::PAIRED_B_FRAGMENTS) {
#pragma unroll
      for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int pp = 0; pp < 4; ++pp) {
          store8<OutT>(p.D, p.C, p.ldd, p.M, p.N, m0 + (wm * 128 + 16 * i + r16),
                       n0 + wn * 128 + 32 * pp + 8 * g, acc[i][2 * pp] * p.alpha,
                       acc[i][2 * pp + 1] * p.alpha, vec_ok);
          // keeps the scheduler from hoisting all 256 accumulator reads (the registers are full)
          __builtin_amdgcn_sched_barrier(0);
        }
    } else {
#pragma unroll
      for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          store4<OutT>(p.D, p.C, p.ldd, p.M, p.N, m0 + (wm * 128 + 16 * i + r16),
                       n0 + wn * 128 + 16 * j + 4 * g, acc[i][j] * p.alpha, vec_ok);
          __builtin_amdgcn_sched_barrier(0);
        }
    }
  };

  // ---- prologue: steps 0 and 1 in flight, wait for step 0, read its A / B 0-3
  Src src = src_of(0), src_nx = src_of(min(1, ntiles - 1));
  {
    const Rsrc rs0 = rsrc_of(src, 0), rs1 = rsrc_of(nkt > 1 ? src : src_nx, nkt > 1 ? 1 : 0);
#pragma unroll
    for (int j = 0; j < Cfg::DMA_PER_TILE; ++j)
      dma(j, 0, rs0, 0);
#pragma unroll
    for (int j = 0; j < Cfg::DMA_PER_TILE; ++j)
      dma(j, Cfg::STAGE_BYTES, rs1, nkt > 1 ? 1 : 0);
  }
  __builtin_amdgcn_s_waitcnt(waitcnt_vm(Cfg::DMA_PER_TILE));
  asm volatile("s_barrier" ::: "memory");
  if (tail)
    clear_k_overhang<Cfg>(smem, tid, p.K, KT, Cfg::SFA_SHIFT == 0, Cfg::SFB_SHIFT == 0);
  // Fragments 0-3 of the first K step of a tile. Every other step gets them from the step before
  // (after its barrier 2); the prologue has no step before, and the last step of a tile skips them
  // so that they are not live across the epilogue.
  auto read_first_step_fragments_0_to_3 = [&](uint32_t stage_offset) {
    const ScaleByteOffsets scale_bytes = scale_byte_offsets(0);
    static_for<0, Cfg::READS_PER_FRAGMENT_GROUP>(
        [&](auto I) { read_fragments_0_to_3(decltype(I)::value, stage_offset, scale_bytes); });
  };
  read_first_step_fragments_0_to_3(0);
  constexpr std::false_type not_last{}, not_first{};
  constexpr std::true_type last{}, first{};

  if constexpr (Cfg::PERSISTENT) { // nkt >= 2: steps s + 2 are at most one tile ahead
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
      store_persistent_tile(lid_of(t));
      if (t + 1 < ntiles)
        read_first_step_fragments_0_to_3((s & 1) * Cfg::STAGE_BYTES);
    }
    return;
  }

  // MN-major: the last step stays in the loop (its 24 reads of the next step are wasted). As a
  // separate copy it gets its own register assignment, and the allocator moves accumulators between
  // AGPRs and VGPRs around the MFMAs there, which corrupted a few outputs.
  if constexpr (Cfg::A_MN_MAJOR || Cfg::B_MN_MAJOR) {
    for (int s = 0; s < nkt; ++s)
      iter(s, s, src, src, not_last, not_first);
  } else {
    for (int s = 0; s < nkt - 1; ++s)
      iter(s, s, src, src, not_last, not_first);
    iter(nkt - 1, nkt - 1, src, src, last, not_first);
  }
  asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)\n\ts_barrier" ::: "memory");

  // ---- Data-parallel and split-K epilogue. Stage 32 rows (fragments 2q and
  // 2q + 1) through each wave's LDS strip, then store sixteen bytes per lane.
  int m0, n0;
  tile_origin(0, m0, n0);
  using OutE = typename std::conditional<Cfg::SPLIT_K, float, OutT>::type;
  void *const out = Cfg::SPLIT_K ? (void *)(p.ws + (size_t)split * p.M * p.ldw) : p.D;
  const void *const cin = Cfg::SPLIT_K ? nullptr : p.C;
  const int ldo = Cfg::SPLIT_K ? p.ldw : p.ldd;
  const float alpha = Cfg::SPLIT_K ? 1.f : p.alpha;
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
      store8<OutE>(out, cin, ldo, p.M, p.N, m0 + wm * 128 + 32 * q + r, n0 + wn * 128 + ec, lo, hi,
                   vec_ok_e);
    }
  }
#endif
}
