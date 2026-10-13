// CDNA4 FP8 / FP4 GEMMs using 16x16x128 scaled MFMA instructions, and the BF16 GEMM on the same
// pipeline (16x16x32 BF16 MFMAs).
// Direct global-to-LDS loads feed two software-pipelined stages. MN-major
// operands stay in their original layout and use transposing LDS reads.
#pragma once
#include "epilogue.cuh"
#include "tile_scheduler.cuh"
#include <type_traits>

enum class Mfma16x16x128GemmMode { DataParallel, Persistent, SplitK };

// The masks use bit 0 for A and bit 1 for B. MN_MAJOR selects transposing
// operand reads; SF_GRAN_128 selects one packed scale code per 128 K values.
// TYPE maps tiles onto the groups of grouped GEMMs (tile_scheduler.cuh). A_TYPE / B_TYPE: FP8 and
// FP4 operands mix freely in the scaled MFMA; BF16 operands (no scales) run two 16x16x32 BF16
// MFMAs per slot, in place of one scaled MFMA.
template <Mfma16x16x128GemmMode Mode, int MN_MAJOR = 0, int SF_GRAN_128 = 0,
          GemmType TYPE = GemmType::Normal, OperandType A_TYPE = OperandType::Fp8,
          OperandType B_TYPE = OperandType::Fp8>
struct Mfma16x16x128GemmConfig {
  static constexpr bool BF16 = A_TYPE == OperandType::Bf16, SCALED = !BF16;
  static_assert(BF16 == (B_TYPE == OperandType::Bf16), "BF16 operands come in pairs");
  static constexpr bool A_FP4 = A_TYPE == OperandType::Fp4, B_FP4 = B_TYPE == OperandType::Fp4;
  // ES bytes per FP8 / BF16 element. A K step is BK elements: 128 bytes of a K-major FP8 or BF16
  // row, 64 bytes of an FP4 row (A_ROW / B_ROW: the LDS bytes of a K-major row of a stage).
  static constexpr int ES = BF16 ? 2 : 1, BK_BYTES = 128;
  static constexpr int BM = 256, BN = 256, BK = BK_BYTES / ES, NW = 4, NT = 256;
  static constexpr int A_ROW = A_FP4 ? BK_BYTES / 2 : BK_BYTES, B_ROW = B_FP4 ? BK_BYTES / 2 : BK_BYTES;
  // bytes of n values of A / B
  __host__ __device__ static constexpr int64_t a_bytes(int64_t n) { return A_FP4 ? (n + 1) / 2 : n * ES; }
  __host__ __device__ static constexpr int64_t b_bytes(int64_t n) { return B_FP4 ? (n + 1) / 2 : n * ES; }
  static constexpr int GROUP_M = 8;
  static constexpr bool PERSISTENT = Mode == Mfma16x16x128GemmMode::Persistent;
  static constexpr bool SPLIT_K = Mode == Mfma16x16x128GemmMode::SplitK;
  static constexpr GemmType kType = TYPE;
  using Scheduler = TileScheduler<TYPE, BM, BN, GROUP_M>;
  static constexpr bool K_GROUPED = Scheduler::K_GROUPED;
  static constexpr bool MASKED = Scheduler::MASKED;
  // m-grouped GEMMs index B and its scales by group
  static constexpr bool B_GROUPED = TYPE == GemmType::MGroupedContiguous ||
                                    TYPE == GemmType::MGroupedContiguousPsum || MASKED;
  static_assert(!SPLIT_K || TYPE == GemmType::Normal, "split-K only for normal GEMMs");
  static constexpr bool A_MN_MAJOR = MN_MAJOR & 1, B_MN_MAJOR = MN_MAJOR & 2;
  static constexpr int SFA_SHIFT = SF_GRAN_128 & 1 ? 2 : 0;
  static constexpr int SFB_SHIFT = SF_GRAN_128 & 2 ? 2 : 0;
  // Paired B fragments give each lane eight adjacent output columns. A
  // transposing read requires sixteen adjacent rows and uses unpaired fragments.
  static constexpr bool PAIRED_B_FRAGMENTS = PERSISTENT && !B_MN_MAJOR;
  static constexpr int A_OFF = 0, B_OFF = BM * A_ROW, SA_OFF = B_OFF + BN * B_ROW,
                       SB_OFF = SA_OFF + BM * 4;
  static constexpr int STAGE_BYTES = SCALED ? SB_OFF + BN * 4 : SA_OFF;
  // Persistent epilogue buffer: 32 rows (16 of each wave row) x 256 bf16, rows padded to 528 B
  // against bank conflicts
  static constexpr int EPI_OFF = 2 * STAGE_BYTES, EPI_PITCH = 2 * BN + 16;
  static constexpr int LDS_BYTES = EPI_OFF + (PERSISTENT ? 32 * EPI_PITCH : 0);
  static_assert(LDS_BYTES <= 160 * 1024, "CDNA4 LDS capacity");
  // Each lane moves DMA_BYTES with buffer_load_dwordx4. A_K_LANES lanes cover
  // one K-major row of A; a wave covers A_K_ROWS rows (B alike). An MN-major K row has
  // MN_ROW_BYTES bytes (FP8 / BF16), or mn_row_bytes(true) with FP4; mn_lanes lanes cover it.
  static constexpr int WAVE_SIZE = 64, DMA_BYTES = 16, SCALE_BYTES = sizeof(int32_t);
  static constexpr int A_K_LANES = A_ROW / DMA_BYTES, A_K_ROWS = WAVE_SIZE / A_K_LANES;
  static constexpr int B_K_LANES = B_ROW / DMA_BYTES, B_K_ROWS = WAVE_SIZE / B_K_LANES;
  static constexpr int MN_ROW_BYTES = BM * ES;
  static constexpr int MN_LANES = MN_ROW_BYTES / DMA_BYTES, MN_ROWS = WAVE_SIZE / MN_LANES;
  __host__ __device__ static constexpr int mn_row_bytes(bool fp4) { return fp4 ? BM / 2 : MN_ROW_BYTES; }
  __host__ __device__ static constexpr int mn_lanes(bool fp4) { return mn_row_bytes(fp4) / DMA_BYTES; }
  __host__ __device__ static constexpr int mn_rows(bool fp4) { return WAVE_SIZE / mn_lanes(fp4); }
  static constexpr int K_WAVE_BYTES = WAVE_SIZE * DMA_BYTES, SCALE_WAVE_BYTES = WAVE_SIZE * SCALE_BYTES;
  static constexpr int lg(int n) {
    int b = 0;
    while ((1 << b) < n)
      ++b;
    return b;
  }
  // FP8: ds_read_b64_tr_b8 reads TR_ROWS K rows (row bits 0 .. LG_TR-1). A fragment is two of
  // those, TR_ROWS apart, so bit HALF_BIT selects the half and keyMN ignores it. The next bit,
  // KEY_BIT, is the lane group. DMA index j uses only the bits keyMN does not read (HALF_BIT, then
  // KEY_BIT + 1 and up), which keeps a lane's swizzle key the same for every j.
  static constexpr int TR_ROWS = 8, LG_TR = lg(TR_ROWS), LG_MN_ROWS = lg(MN_ROWS);
  static constexpr int HALF_BIT = LG_TR, KEY_BIT = HALF_BIT + 1;
  static_assert(BN == BM && WAVE_SIZE % MN_LANES == 0, "A and B share the MN-major lane map");
  static_assert(BF16 || (LG_MN_ROWS + 1 == LG_TR && (1 << (LG_TR + 1)) == MN_LANES),
                "the lane takes bits 0..LG_MN_ROWS-1; one transpose bit is left for the wave, and "
                "the key covers every chunk");
  static constexpr int A_LOADS = BM / (A_K_ROWS * NW),
                       B_LOADS = BN / (B_K_ROWS * NW); // per wave, K_WAVE_BYTES
  static_assert((!A_MN_MAJOR || A_LOADS == BK / (NW * mn_rows(A_FP4))) &&
                    (!B_MN_MAJOR || B_LOADS == BK / (NW * mn_rows(B_FP4))),
                "MN-major K rows take one DMA per remaining group");
  // + one scale load for A and B
  static constexpr int DMA_PER_TILE = A_LOADS + B_LOADS + (SCALED ? 2 : 0);
  static constexpr int EPI_LD = 128 + 4, EPI_WAVE_BYTES = 32 * EPI_LD * 4;
  static_assert(NW * EPI_WAVE_BYTES <= LDS_BYTES,
                "epilogue staging must fit in the operand stages");
  // slots (MFMA numbers 0..63 of an iteration) of the two barriers; the DMAs of tile kt + 2 are
  // spread evenly over the SPAN slots that follow barrier 1
  static constexpr int BAR1 = 16, BAR2 = 48, SPAN = 48;
  // LDS reads that bring one group of fragments into registers (A and B fragments 0-3, or 4-7):
  // per fragment its operand halves (one with FP4: 16 bytes per lane and step) and, unless BF16, a
  // scale code. A half is one ds_read_b128 (K-major) or two transposing reads (MN-major).
  static constexpr int A_HALVES = A_FP4 ? 1 : 2, B_HALVES = B_FP4 ? 1 : 2;
  static constexpr int A_PARTS = A_HALVES + SCALED, B_PARTS = B_HALVES + SCALED;
  static constexpr int READS_PER_FRAGMENT_GROUP = 4 * (A_PARTS + B_PARTS);
  // read i of a fragment group: part < halves is an operand half, the next part the scale code
  struct FragmentRead {
    bool is_b;
    int fragment, part;
  };
  // fragments 4-7 in the order A 4-7, B 4-7
  __host__ __device__ static constexpr FragmentRead read_of_fragments_4_to_7(int i) {
    return i < 4 * A_PARTS ? FragmentRead{false, 4 + i / A_PARTS, i % A_PARTS}
                           : FragmentRead{true, 4 + (i - 4 * A_PARTS) / B_PARTS, (i - 4 * A_PARTS) % B_PARTS};
  }
  // fragments 0-3 in the order B 0, A 0-3, B 1-3: the order in which the first quadrant of MFMAs
  // (B 0-3 x A 0-3, B 0 first) needs them
  __host__ __device__ static constexpr FragmentRead read_of_fragments_0_to_3(int i) {
    return i < B_PARTS                 ? FragmentRead{true, 0, i}
           : i < B_PARTS + 4 * A_PARTS ? FragmentRead{false, (i - B_PARTS) / A_PARTS, (i - B_PARTS) % A_PARTS}
                                       : FragmentRead{true, 1 + (i - B_PARTS - 4 * A_PARTS) / B_PARTS,
                                                      (i - B_PARTS - 4 * A_PARTS) % B_PARTS};
  }
  static_assert(BAR1 >= READS_PER_FRAGMENT_GROUP / 2 && BAR1 + SPAN <= 64,
                "DMA window: after the reads of stage cur, within the iteration");
  static_assert(SPAN >= DMA_PER_TILE, "at most one DMA per slot");
  __host__ __device__ static constexpr int dma_slot(int j) {
    return BAR1 + j * SPAN / DMA_PER_TILE;
  }
  // descriptors: A, B, and with FP8 the scales of A and B
  static constexpr int RSRCS = SCALED ? 4 : 2;
  // slot that builds DMA descriptor i: the one before its first DMA
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
  // MN-major tile, FP8: keyMN(row) is bits 0..LG_TR-1, plus KEY_BIT packed above them: the bits a
  // transposing read's lane fixes (r16 / 2, and the low bit of its lane group).
  // BF16: ds_read_b64_tr_b16 gives each of 16 lanes a column of 4 K rows; the 16 lanes read 32 B
  // of each row. keyMN moves those 32 B by row bits 0-1 (2 chunks per step) and the lane group
  // pairs apart by row bit 3 (8 chunks), so the rows a read serves hit different banks.
  // FP4 (fp4): ds_read_b64_tr_b4 gives each of 16 lanes a column of 16 K rows, 8 bytes of each row.
  // The 32 lanes of a read cycle read 32 rows at the same 8-byte half of a chunk; the key (row bits
  // 1-3) spreads the 16 rows of a parity over the 8 chunks: two rows per chunk, a 2-way conflict.
  __host__ __device__ static constexpr int keyMN(int r, bool fp4 = false) {
    return fp4    ? (r >> 1) & 7
           : BF16 ? 2 * (r & 3) | (((r >> 3) & 1) << 3)
                  : (r & ((1 << LG_TR) - 1)) | (((r >> KEY_BIT) & 1) << LG_TR);
  }
  // DMA j (0 .. A_LOADS-1) of wave w loads mn_rows K rows x mn_row_bytes bytes, at base row
  // mn_row(w) + mn_jrow(j). The lane adds 0 .. mn_rows-1. FP8: wave bit 0 takes the remaining
  // transpose bit and wave bit 1 takes KEY_BIT; j takes HALF_BIT and the bits above KEY_BIT.
  // BF16: the lane takes row bit 0, the wave bits 1 and 3 (the rest of keyMN), j bits 2, 4, 5.
  // FP4: the lane takes row bits 0-2, the wave bits 3 (the rest of the key) and 4, j bits 5-6.
  __host__ __device__ static constexpr int mn_row(int w, bool fp4 = false) {
    return fp4    ? ((w & 1) << 3) | ((w >> 1) << 4)
           : BF16 ? ((w & 1) << 1) | ((w >> 1) << 3)
                  : ((w & 1) << LG_MN_ROWS) | ((w >> 1) << KEY_BIT);
  }
  __host__ __device__ static constexpr int mn_jrow(int j, bool fp4 = false) {
    return fp4    ? j << 5
           : BF16 ? ((j & 1) << 2) | ((j >> 1) << 4)
                  : ((j & 1) << HALF_BIT) | ((j >> 1) << (KEY_BIT + 1));
  }
  // swizzle keys (physical chunk = logical chunk ^ key(row)). The 16 lanes that one ds_read_b128
  // cycle serves read rows r16 = 0..15 of a fragment and must hit 16 different 16 B bank groups,
  // i.e. different (row % 2, key) pairs for 128-byte rows, (row % 4, key) pairs for 64-byte FP4
  // rows. Rows 0..15 -> (r >> 1) & 7, or (r >> 2) & 3. B with PAIRED_B_FRAGMENTS: rows
  // {8a + b : a, b < 4} -> bit 1 and bits 3-4 of the row, or bits 3-4.
  __host__ __device__ static constexpr int row_key(int r, bool fp4) { return fp4 ? (r >> 2) & 3 : (r >> 1) & 7; }
  __host__ __device__ static constexpr int key(int r) { return row_key(r, A_FP4); }
  __host__ __device__ static constexpr int keyB(int r) {
    return !PAIRED_B_FRAGMENTS ? row_key(r, B_FP4)
           : B_FP4             ? (r >> 3) & 3
                               : ((r >> 1) & 1) | (((r >> 3) & 3) << 1);
  }
};

// The partial last K tile sits in stage 0 (it is processed first): only its first kv K values,
// kv_bytes of each row, are data. Zero the rest (it holds the next row, padding, or 0 past the end
// of the buffer), and set the scale codes of K blocks without data to 127 (= 1.0) so that padding
// codes cannot turn 0 * scale into NaN. An MN-major operand needs no clearing: its K rows past K
// are outside its buffer and read as 0. With 128-wide scale blocks (fix_sa / fix_sb false) the
// tile has one code, and the other bytes of the word belong to other K tiles.
template <class Cfg>
__device__ __forceinline__ void clear_k_overhang(uint8_t *smem, int tid, int K, int KT,
                                                 bool fix_sa = true, bool fix_sb = true) {
  constexpr int BM = Cfg::BM, BN = Cfg::BN;
  const int kv = K - (KT - 1) * Cfg::BK, nb = (kv + 31) / 32;
  for (int r = tid; r < BM + BN; r += Cfg::NT) {
    const bool is_a = r < BM;
    const int row_bytes = is_a ? Cfg::A_ROW : Cfg::B_ROW;
    const int kv_bytes = (int)(is_a ? Cfg::a_bytes(kv) : Cfg::b_bytes(kv));
    uint8_t *row = smem + (is_a ? Cfg::A_OFF + r * Cfg::A_ROW : Cfg::B_OFF + (r - BM) * Cfg::B_ROW);
    for (int c = kv_bytes / 16; c < row_bytes / 16 && !(is_a ? Cfg::A_MN_MAJOR : Cfg::B_MN_MAJOR); ++c) {
      v4i *q = (v4i *)(row + (c ^ (is_a ? Cfg::key(r) : Cfg::keyB(r - BM))) * 16); // logical chunk c
      v4i v = *q;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int keep = kv_bytes - c * 16 - 4 * i; // bytes of dword i that are data
        v[i] = keep >= 4 ? v[i] : keep <= 0 ? 0 : (int)((uint32_t)v[i] & ((1u << (8 * keep)) - 1));
      }
      *q = v;
    }
    if (Cfg::SCALED && nb < 4 && (is_a ? fix_sa : fix_sb)) {
      uint32_t *w =
          (uint32_t *)(smem + (is_a ? Cfg::SA_OFF + r * 4 : Cfg::SB_OFF + (r - BM) * 4));
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
  using Scheduler = typename Cfg::Scheduler;
  using Src = GroupTile;
  extern __shared__ __attribute__((aligned(1024))) uint8_t smem[];

  const int tid = threadIdx.x, lane = tid % 64, g = lane / 16, r16 = lane % 16;
  const int wave = __builtin_amdgcn_readfirstlane(tid / 64);
  const int wm = wave / 2, wn = wave % 2;
  // ---- K range of the current tile. A K-grouped GEMM takes it from the tile's group; its K-major
  // operands are [M, kg] / [N, kg] blocks, whose row strides are kg.
  int K = p.K, lda = p.lda, ldb = p.ldb;
  int KT = (K + BK - 1) / BK;
  KGroup kgrp{0, p.K, 0};
  // ---- work of this workgroup: ntiles output tiles with logical ids lid0, lid0 + gridDim.x, ...
  // (PERSISTENT; otherwise one tile and one K split), each over K tiles kb .. kb + nkt - 1
  int lid0 = 0, ntiles = 1, split = 0, kb = 0, nkt = KT, full = 0, rest = 0;
  // Spread the partial last round across the participating workgroups.
  auto lid_of = [&](int t) {
    return Cfg::PERSISTENT && t == full ? full * (int)gridDim.x + xcd_remap(blockIdx.x, rest)
                                        : lid0 + t * (int)gridDim.x;
  };
  auto src_of = [&](int t) { return Scheduler::tile(p, lid_of(t)); };
  // a partial last K tile is processed first (not with PERSISTENT)
  bool tail = false;
  auto ktile = [&](int t) { return tail ? (t == 0 ? KT - 1 : kb + t - 1) : kb + t; };

  // ---- Global -> LDS. Load i of A covers A_K_ROWS K-major rows starting at
  // A_K_ROWS * (i * NW + wave) (B alike). For an MN-major operand, load j covers K rows
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
  // Grouped operands: the matrix is the tile's group (its base from *_group below), and the
  // extents are the bytes from a group's base to the end of its data.
  static_assert(NW == 4, "the keys of the rows of load i must not depend on i");
  constexpr int load_bytes = Cfg::DMA_BYTES;
  constexpr int a_mn_lanes = Cfg::mn_lanes(Cfg::A_FP4), b_mn_lanes = Cfg::mn_lanes(Cfg::B_FP4);
  const uint32_t a_mn_row = (uint32_t)(Cfg::mn_row(wave, Cfg::A_FP4) + lane / a_mn_lanes);
  const uint32_t b_mn_row = (uint32_t)(Cfg::mn_row(wave, Cfg::B_FP4) + lane / b_mn_lanes);
  const uint32_t a_mn_chunk =
      (uint32_t)((lane % a_mn_lanes) ^ Cfg::keyMN((int)a_mn_row, Cfg::A_FP4)) * load_bytes;
  const uint32_t b_mn_chunk =
      (uint32_t)((lane % b_mn_lanes) ^ Cfg::keyMN((int)b_mn_row, Cfg::B_FP4)) * load_bytes;
  const uint32_t a_k_row = (uint32_t)(Cfg::A_K_ROWS * wave + lane / Cfg::A_K_LANES);
  const uint32_t b_k_row = (uint32_t)(Cfg::B_K_ROWS * wave + lane / Cfg::B_K_LANES);
  const uint32_t s_voff = (Cfg::WAVE_SIZE * wave + lane) * Cfg::SCALE_BYTES;
  uint32_t a_voff, b_voff, a_step, b_step;
  int64_t a_extent, b_extent, sa_extent, sb_extent;
  auto set_geometry = [&] {
    KT = (K + BK - 1) / BK;
    a_extent = Cfg::A_MN_MAJOR ? (int64_t)(K - 1) * lda + ((Cfg::a_bytes(p.M) + 3) & ~3)
                               : (int64_t)(p.M - 1) * lda + ((Cfg::a_bytes(K) + 3) & ~3);
    b_extent = Cfg::B_MN_MAJOR ? (int64_t)(K - 1) * ldb + ((Cfg::b_bytes(p.N) + 3) & ~3)
                               : (int64_t)(p.N - 1) * ldb + ((Cfg::b_bytes(K) + 3) & ~3);
    // scale words: one per K tile, or (128-wide scale blocks, shift 2) one per 4 K tiles
    const int sa_words = (KT + (1 << Cfg::SFA_SHIFT) - 1) >> Cfg::SFA_SHIFT;
    const int sb_words = (KT + (1 << Cfg::SFB_SHIFT) - 1) >> Cfg::SFB_SHIFT;
    sa_extent = ((int64_t)(sa_words - 1) * p.lda_sf + p.M) * 4;
    sb_extent = ((int64_t)(sb_words - 1) * p.ldb_sf + p.N) * 4;
    a_voff = Cfg::A_MN_MAJOR
                 ? a_mn_row * lda + a_mn_chunk
                 : a_k_row * lda + ((lane % Cfg::A_K_LANES) ^ Cfg::key((int)a_k_row)) * load_bytes;
    b_voff = Cfg::B_MN_MAJOR
                 ? b_mn_row * ldb + b_mn_chunk
                 : b_k_row * ldb + ((lane % Cfg::B_K_LANES) ^ Cfg::keyB((int)b_k_row)) * load_bytes;
    a_step = Cfg::A_K_ROWS * NW * lda;
    b_step = Cfg::B_K_ROWS * NW * ldb;
  };
  // size = min(end - base, 2^32 - 1) from the high word. 64-bit compares are VALU only, and the
  // compiler turns hi != 0 back into one unless hi is opaque.
  auto rsrc_at = [&](const uint8_t *base, const uint8_t *end) {
    const uint64_t n = (uint64_t)end - (uint64_t)base;
    uint32_t hi = (uint32_t)(n >> 32);
    asm("" : "+s"(hi));
    return make_rsrc(base, hi ? 0xffffffffu : (uint32_t)n);
  };
  // Group bases of A, B and their packed scales
  auto a_group = [&](const Src &s) -> const uint8_t * {
    if constexpr (Cfg::MASKED)
      return p.A + (size_t)s.group * p.a_group_stride;
    else if constexpr (Cfg::K_GROUPED && Cfg::A_FP4)
      return p.A + (size_t)kgrp.k0 / 2;
    else if constexpr (Cfg::K_GROUPED)
      return p.A + (size_t)kgrp.k0 * (Cfg::A_MN_MAJOR ? lda : p.M * Cfg::ES);
    else
      return p.A;
  };
  auto b_group = [&](const Src &s) -> const uint8_t * {
    if constexpr (Cfg::B_GROUPED)
      return p.B + (size_t)s.group * p.b_group_stride;
    else if constexpr (Cfg::K_GROUPED && Cfg::B_FP4)
      return p.B + (size_t)kgrp.k0 / 2;
    else if constexpr (Cfg::K_GROUPED)
      return p.B + (size_t)kgrp.k0 * (Cfg::B_MN_MAJOR ? ldb : p.N * Cfg::ES);
    else
      return p.B;
  };
  auto sa_group = [&](const Src &s) -> const uint8_t * {
    const uint8_t *base = (const uint8_t *)p.sfa;
    if constexpr (Cfg::MASKED)
      return base + (size_t)s.group * p.sfa_group_stride * 4;
    else if constexpr (Cfg::K_GROUPED)
      return base + (size_t)kgrp.w0 * p.lda_sf * 4;
    else
      return base;
  };
  auto sb_group = [&](const Src &s) -> const uint8_t * {
    const uint8_t *base = (const uint8_t *)p.sfb;
    if constexpr (Cfg::B_GROUPED)
      return base + (size_t)s.group * p.sfb_group_stride * 4;
    else if constexpr (Cfg::K_GROUPED)
      return base + (size_t)kgrp.w0 * p.ldb_sf * 4;
    else
      return base;
  };
  // a tile's origin. The DMA bases are built from it per K step: kept 64-bit for two tiles (the
  // current and the next), they cost the persistent kernel SGPR spills.
  // descriptor i (0: A, 1: B, 2, 3: their scales) of K tile kt (local index) of tile `ld`. Base:
  // the tile's first row (K-major) / column, plus K row kp BK (MN-major) or the scale word of kp.
  struct Rsrc {
    __amdgpu_buffer_rsrc_t r[4];
  };
  auto kp_of = [&](int kt) { return ktile(kt); };
  auto step_rsrc = [&](int i, const Src &ld, int kt) {
    const int kp = kp_of(kt);
    if (i == 0) {
      const uint8_t *base = a_group(ld);
      return rsrc_at(base + (Cfg::A_MN_MAJOR ? (size_t)(kp * BK) * lda + (size_t)Cfg::a_bytes(ld.m0)
                                             : (size_t)ld.m0 * lda),
                     base + a_extent);
    }
    if (i == 1) {
      const uint8_t *base = b_group(ld);
      return rsrc_at(base + (Cfg::B_MN_MAJOR ? (size_t)(kp * BK) * ldb + (size_t)Cfg::b_bytes(ld.n0)
                                             : (size_t)ld.n0 * ldb),
                     base + b_extent);
    }
    if (i == 2) {
      const uint8_t *base = sa_group(ld);
      return rsrc_at(base + ((size_t)(kp >> Cfg::SFA_SHIFT) * p.lda_sf + ld.m0) * 4,
                     base + sa_extent);
    }
    const uint8_t *base = sb_group(ld);
    return rsrc_at(base + ((size_t)(kp >> Cfg::SFB_SHIFT) * p.ldb_sf + ld.n0) * 4,
                   base + sb_extent);
  };
  auto rsrc_of = [&](const Src &ld, int kt) {
    Rsrc rs;
    static_for<0, Cfg::RSRCS>([&](auto I) { rs.r[decltype(I)::value] = step_rsrc(decltype(I)::value, ld, kt); });
    return rs;
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
            rs.r[0],
            (lds_ptr_t)(st + Cfg::A_OFF + (Cfg::mn_row(wave, Cfg::A_FP4) + Cfg::mn_jrow(j, Cfg::A_FP4)) *
                                              Cfg::mn_row_bytes(Cfg::A_FP4)),
            Cfg::DMA_BYTES, a_voff, __builtin_amdgcn_readfirstlane(Cfg::mn_jrow(j, Cfg::A_FP4) * lda), 0, 0);
      else
        __builtin_amdgcn_raw_ptr_buffer_load_lds(
            rs.r[0], (lds_ptr_t)(st + Cfg::A_OFF + (j * NW + wave) * Cfg::K_WAVE_BYTES),
            Cfg::DMA_BYTES, a_voff, j * a_step + kp * Cfg::A_ROW, 0, 0);
    } else if (j < Cfg::A_LOADS + Cfg::B_LOADS) {
      const int jb = j - Cfg::A_LOADS;
      if constexpr (Cfg::B_MN_MAJOR)
        __builtin_amdgcn_raw_ptr_buffer_load_lds(
            rs.r[1],
            (lds_ptr_t)(st + Cfg::B_OFF + (Cfg::mn_row(wave, Cfg::B_FP4) + Cfg::mn_jrow(jb, Cfg::B_FP4)) *
                                              Cfg::mn_row_bytes(Cfg::B_FP4)),
            Cfg::DMA_BYTES, b_voff, __builtin_amdgcn_readfirstlane(Cfg::mn_jrow(jb, Cfg::B_FP4) * ldb), 0, 0);
      else
        __builtin_amdgcn_raw_ptr_buffer_load_lds(
            rs.r[1], (lds_ptr_t)(st + Cfg::B_OFF + (jb * NW + wave) * Cfg::K_WAVE_BYTES),
            Cfg::DMA_BYTES, b_voff, jb * b_step + kp * Cfg::B_ROW, 0, 0);
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
  // (an FP4 row has logical chunks 0-3 and one half: chunk g)
  const uint32_t a_row_offset = Cfg::A_OFF + (uint32_t)(wm * 128 + r16) * Cfg::A_ROW;
  const uint32_t b_row_offset = Cfg::B_OFF + (uint32_t)(wn * 128 + Cfg::brow(r16)) * Cfg::B_ROW;
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
  // BF16: fragment f covers the 16 MN columns of chunks 16 wm + 2 f, + 1 (8 columns each).
  // Transposing read (h, s) gives lane (g, r16) K = 32 h + 8 g + 4 s .. + 3 of its column r16; lane
  // r16 points at K row 32 h + 8 g + 4 s + r16 / 4, columns 4 (r16 % 4) .. + 3. The key of that row
  // is 2 (r16 / 4) | (g & 1) << 3 for every h, s, so fragment f is base ^ (f << 5).
  const uint32_t mn_key =
      Cfg::BF16 ? (uint32_t)(2 * (r16 >> 2) | ((g & 1) << 3)) : (uint32_t)((r16 >> 1) | ((g & 1) << 3));
  const uint32_t mn_lane_offset =
      Cfg::BF16 ? (uint32_t)(8 * g + (r16 >> 2)) * Cfg::MN_ROW_BYTES + 8 * (r16 & 1)
                : (uint32_t)(16 * g + (r16 >> 1)) * 256 + 8 * (r16 & 1);
  const uint32_t mn_lane_chunk = Cfg::BF16 ? (uint32_t)((r16 & 3) >> 1) : 0u;
  const uint32_t mn_wave_chunk = Cfg::BF16 ? 16u : 8u;
  // FP4: fragment f covers the 16 MN columns of half f % 2 of chunk 4 w + f / 2 (16 bytes hold 32
  // columns). Transposing read s gives lane (g, r16) K = 32 g + 16 s .. + 15 of its column r16; lane
  // r16 points at K row 32 g + 16 s + r16, whose key (r16 / 2) % 8 is the same for every s.
  const uint32_t fp4_key = (uint32_t)((r16 >> 1) & 7);
  const uint32_t fp4_lane_offset = (uint32_t)(32 * g + r16) * (uint32_t)Cfg::mn_row_bytes(true);
  const uint32_t a_transpose_base =
      Cfg::A_FP4 ? Cfg::A_OFF + fp4_lane_offset + ((4u * wm) ^ fp4_key) * 16
                 : Cfg::A_OFF + mn_lane_offset + ((mn_wave_chunk * wm + mn_lane_chunk) ^ mn_key) * 16;
  const uint32_t b_transpose_base =
      Cfg::B_FP4 ? Cfg::B_OFF + fp4_lane_offset + ((4u * wn) ^ fp4_key) * 16
                 : Cfg::B_OFF + mn_lane_offset + ((mn_wave_chunk * wn + mn_lane_chunk) ^ mn_key) * 16;

  v4i a_fragments[8][2], b_fragments[8][2];    // [fragment][half]: the MFMA operands
  uint32_t a_scale_codes[8], b_scale_codes[8]; // [fragment]: the lane's scale code, in byte 0
  // One operand half of an MN-major fragment, assembled whole from its two transposing reads:
  // writing half of a fragment that is still live costs registers.
  // The XOR goes after the stage offset (whose bits 4-6 are 0): base ^ (f << 4) alone is loop
  // invariant, and the compiler keeps all 8 of them in registers.
  static_assert((Cfg::STAGE_BYTES & 0xf0) == 0, "the stage offset must not touch the chunk bits");
  auto read_transposed_half = [&](uint32_t stage_offset, uint32_t base, int fragment, int half, auto fp4) {
    typedef const __attribute__((address_space(3))) v2i *lds_v2i;
    if constexpr (decltype(fp4)::value) { // the operand's only half
      const uint32_t addr =
          ((base + stage_offset) ^ (uint32_t)((fragment >> 1) << 4)) + 8 * (uint32_t)(fragment & 1);
      const v2i lo = ds_read_tr4((lds_v2i)(smem + addr));
      const v2i hi = ds_read_tr4((lds_v2i)(smem + addr + 16 * Cfg::mn_row_bytes(true)));
      return v4i{lo[0], lo[1], hi[0], hi[1]};
    } else if constexpr (Cfg::BF16) {
      const uint32_t addr =
          ((base + stage_offset) ^ (uint32_t)(fragment << 5)) + 32 * half * Cfg::MN_ROW_BYTES;
      const v2i lo = ds_read_tr16((lds_v2i)(smem + addr));
      const v2i hi = ds_read_tr16((lds_v2i)(smem + addr + 4 * Cfg::MN_ROW_BYTES));
      return v4i{lo[0], lo[1], hi[0], hi[1]};
    } else {
      const uint32_t addr = ((base + stage_offset) ^ (uint32_t)(fragment << 4)) + 64 * half * 256;
      const v2i lo = ds_read_tr8((lds_v2i)(smem + addr));
      const v2i hi = ds_read_tr8((lds_v2i)(smem + addr + 8 * 256));
      return v4i{lo[0], lo[1], hi[0], hi[1]};
    }
  };
  // One LDS read for fragment f of A (is_b = false) or B: part 0 and 1 are its operand halves,
  // part 2 its scale code (the byte of the scale word that scale_bytes picks for the K tile).
  auto read_fragment_part = [&](bool is_b, int f, int part, uint32_t stage_offset,
                                const ScaleByteOffsets &scale_bytes) {
    const uint8_t *stage = smem + stage_offset;
    if (!is_b) {
      if (part < Cfg::A_HALVES)
        a_fragments[f][part] = Cfg::A_MN_MAJOR
                                   ? read_transposed_half(stage_offset, a_transpose_base, f, part,
                                                          std::bool_constant<Cfg::A_FP4>{})
                                   : *(const v4i *)(stage + a_row_offset + f * 16 * Cfg::A_ROW +
                                                    (part ? a_half1_offset : a_half0_offset));
      else
        a_scale_codes[f] =
            *(const uint8_t *)(smem + (stage_offset + scale_bytes.a) + a_scale_offset + f * 64);
    } else {
      if (part < Cfg::B_HALVES)
        b_fragments[f][part] = Cfg::B_MN_MAJOR
                                   ? read_transposed_half(stage_offset, b_transpose_base, f, part,
                                                          std::bool_constant<Cfg::B_FP4>{})
                                   : *(const v4i *)(stage + b_row_offset + Cfg::bfrag(f) * Cfg::B_ROW +
                                                    (part ? b_half1_offset : b_half0_offset));
      else
        b_scale_codes[f] = *(const uint8_t *)(smem + (stage_offset + scale_bytes.b) +
                                              b_scale_offset + Cfg::bfrag(f) * 4);
    }
  };
  // Read i (0 .. READS_PER_FRAGMENT_GROUP - 1) of fragments 4-7 / 0-3 of a K step
  auto read_fragments_4_to_7 = [&](int i, uint32_t stage_offset,
                                   const ScaleByteOffsets &scale_bytes) {
    const auto r = Cfg::read_of_fragments_4_to_7(i);
    read_fragment_part(r.is_b, r.fragment, r.part, stage_offset, scale_bytes);
  };
  auto read_fragments_0_to_3 = [&](int i, uint32_t stage_offset,
                                   const ScaleByteOffsets &scale_bytes) {
    const auto r = Cfg::read_of_fragments_0_to_3(i);
    read_fragment_part(r.is_b, r.fragment, r.part, stage_offset, scale_bytes);
  };

  v4f acc[8][8];

  // MFMA of slot t. B goes into the MFMA's A slot, so acc[i][j][e] of lane l is
  //   D[wm*128 + 16 i + l % 16][wn*128 + bfrag(j) + brow(4 (l / 16) + e)]  (one row per lane).
  // (PERSISTENT: the first step of a tile accumulates onto 0 instead of zeroing 256 registers)
  // BF16: the two operand halves are the K = 0..31 and 32..63 of two 16x16x32 MFMAs.
  auto mfma = [&](int t, bool zero) {
    const int q = t / 16, j = (q >> 1) * 4 + (t % 16) / 4, i = (q & 1) * 4 + t % 4;
    if constexpr (Cfg::BF16) {
      acc[i][j] = mfma_bf16_16x16x32(b_fragments[j][0], a_fragments[i][0], zero ? v4f{} : acc[i][j]);
      acc[i][j] = mfma_bf16_16x16x32(b_fragments[j][1], a_fragments[i][1], acc[i][j]);
    } else {
      acc[i][j] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
          scaled_operand<Cfg::B_FP4>(b_fragments[j]), scaled_operand<Cfg::A_FP4>(a_fragments[i]),
          zero ? v4f{} : acc[i][j], Cfg::B_FP4 ? kFmtFp4 : kFmtFp8, Cfg::A_FP4 ? kFmtFp4 : kFmtFp8, 0,
          b_scale_codes[j], 0, a_scale_codes[i]);
    }
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
    const Src ld{same ? src.m0 : src_nx.m0, same ? src.n0 : src_nx.n0,
                 same ? src.group : src_nx.group, 0};
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
      static_for<0, Cfg::RSRCS>([&](auto I) {
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

  // Output group bases (D and C share their layout)
  auto d_group = [&](const Src &s) -> uint8_t * {
    if constexpr (Cfg::MASKED || Cfg::K_GROUPED)
      return (uint8_t *)p.D + (size_t)s.group * p.d_group_stride * sizeof(OutT);
    else
      return (uint8_t *)p.D;
  };
  auto c_group = [&](const Src &s) -> const uint8_t * {
    if (!p.C)
      return nullptr;
    if constexpr (Cfg::MASKED || Cfg::K_GROUPED)
      return (const uint8_t *)p.C + (size_t)s.group * p.d_group_stride * sizeof(OutT);
    else
      return (const uint8_t *)p.C;
  };

  // Persistent epilogue. Full tiles stage BF16 values through LDS for coalesced
  // stores; edge tiles use bounds-checked stores from each lane's accumulators.
  const bool vec_ok = (p.ldd * (int)sizeof(OutT)) % 16 == 0 && ((uintptr_t)p.D & 15) == 0 &&
                      ((uintptr_t)p.C & 15) == 0 &&
                      (p.d_group_stride * (int64_t)sizeof(OutT)) % 16 == 0;
  const bool fast_ok = !p.C && vec_ok;
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
  auto store_tile_lds = [&](const Src &s, auto unit_alpha) {
    uint8_t *const dg = d_group(s);
    const __amdgpu_buffer_rsrc_t rout =
        rsrc_at(dg + (size_t)s.m0 * p.ldd * sizeof(OutT), dg + (size_t)p.M * p.ldd * sizeof(OutT));
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
        constexpr int ss = decltype(S)::value;
        const v4u v = *(const v4u *)(smem + epi_rd + 2 * ss * Cfg::EPI_PITCH);
        const int soff = (int)(((uint32_t)(16 * i + 2 * ss) * p.ldd + s.n0) * 2);
        buffer_store16(v, rout, epi_out, soff);
      });
      asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
    });
  };
  auto store_persistent_tile = [&](const Src &s) {
    if (fast_ok && s.m0 + BM <= s.m_end && s.n0 + BN <= p.N) {
      if constexpr (Cfg::PERSISTENT && std::is_same<OutT, uint16_t>::value) {
        if (p.alpha == 1.f)
          store_tile_lds(s, std::true_type{});
        else
          store_tile_lds(s, std::false_type{});
        return;
      }
    }
    void *const dg = d_group(s);
    const void *const cg = c_group(s);
    if constexpr (Cfg::PAIRED_B_FRAGMENTS) {
#pragma unroll
      for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int pp = 0; pp < 4; ++pp) {
          store8<OutT>(dg, cg, p.ldd, s.m_end, p.N, s.m0 + (wm * 128 + 16 * i + r16),
                       s.n0 + wn * 128 + 32 * pp + 8 * g, acc[i][2 * pp] * p.alpha,
                       acc[i][2 * pp + 1] * p.alpha, vec_ok);
          // keeps the scheduler from hoisting all 256 accumulator reads (the registers are full)
          __builtin_amdgcn_sched_barrier(0);
        }
    } else {
#pragma unroll
      for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          store4<OutT>(dg, cg, p.ldd, s.m_end, p.N, s.m0 + (wm * 128 + 16 * i + r16),
                       s.n0 + wn * 128 + 16 * j + 4 * g, acc[i][j] * p.alpha, vec_ok);
          __builtin_amdgcn_sched_barrier(0);
        }
    }
  };

  // Fragments 0-3 of the first K step of a tile. Every other step gets them from the step before
  // (after its barrier 2); the prologue has no step before, and the last step of a tile skips them
  // so that they are not live across the epilogue.
  auto read_first_step_fragments_0_to_3 = [&](uint32_t stage_offset) {
    const ScaleByteOffsets scale_bytes = scale_byte_offsets(0);
    static_for<0, Cfg::READS_PER_FRAGMENT_GROUP>(
        [&](auto I) { read_fragments_0_to_3(decltype(I)::value, stage_offset, scale_bytes); });
  };
  // ---- prologue: steps 0 and 1 in flight, wait for step 0, read its A / B 0-3
  auto prologue = [&](const Src &src, const Src &src_nx) {
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
      clear_k_overhang<Cfg>(smem, tid, K, KT, Cfg::SFA_SHIFT == 0, Cfg::SFB_SHIFT == 0);
    read_first_step_fragments_0_to_3(0);
  };
  constexpr std::false_type not_last{}, not_first{};
  constexpr std::true_type last{}, first{};

  if constexpr (Cfg::PERSISTENT) { // nkt >= 2: steps s + 2 are at most one tile ahead
    set_geometry();
    nkt = KT;
    const int tiles = Scheduler::num_tiles(p);
    lid0 = xcd_remap(blockIdx.x, gridDim.x);
    full = tiles / gridDim.x;
    rest = tiles - full * gridDim.x;
    ntiles = full + (blockIdx.x < rest);
    if (ntiles == 0)
      return;
    Src src = src_of(0), src_nx = src_of(min(1, ntiles - 1));
    prologue(src, src_nx);
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
      store_persistent_tile(src);
      if (t + 1 < ntiles)
        read_first_step_fragments_0_to_3((s & 1) * Cfg::STAGE_BYTES);
    }
    return;
  }

  // ---- Data-parallel and split-K: one tile (and K split) at a time
  auto run_tile = [&](const Src &src) {
    if constexpr (Cfg::K_GROUPED) {
      kgrp = Scheduler::k_group(p, src.group, 128 << Cfg::SFA_SHIFT);
      K = kgrp.kg;
      // (FP4 operands are column ranges of [M, sum_k / 2] / [N, sum_k / 2] rows)
      if constexpr (!Cfg::A_MN_MAJOR && !Cfg::A_FP4)
        lda = K * Cfg::ES;
      if constexpr (!Cfg::B_MN_MAJOR && !Cfg::B_FP4)
        ldb = K * Cfg::ES;
      set_geometry();
      kb = 0;
      nkt = KT;
    }
    tail = K % BK != 0 && kb + nkt == KT;
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
      for (int j = 0; j < 8; ++j)
        acc[i][j] = v4f{};
    if (nkt > 0) {
      prologue(src, src);
      // MN-major: the last step stays in the loop (its 24 reads of the next step are wasted). As a
      // separate copy it gets its own register assignment, and the allocator moves accumulators
      // between AGPRs and VGPRs around the MFMAs there, which corrupted a few outputs.
      if constexpr (Cfg::A_MN_MAJOR || Cfg::B_MN_MAJOR) {
        for (int s = 0; s < nkt; ++s)
          iter(s, s, src, src, not_last, not_first);
      } else {
        for (int s = 0; s < nkt - 1; ++s)
          iter(s, s, src, src, not_last, not_first);
        iter(nkt - 1, nkt - 1, src, src, last, not_first);
      }
      asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)\n\ts_barrier" ::: "memory");
    }

    // Stage 32 rows (fragments 2q and 2q + 1) through each wave's LDS strip, then store sixteen
    // bytes per lane.
    using OutE = typename std::conditional<Cfg::SPLIT_K, float, OutT>::type;
    void *const out = Cfg::SPLIT_K ? (void *)(p.ws + (size_t)split * p.M * p.ldw) : d_group(src);
    const void *const cin = Cfg::SPLIT_K ? nullptr : c_group(src);
    const int ldo = Cfg::SPLIT_K ? p.ldw : p.ldd;
    const float alpha = Cfg::SPLIT_K ? 1.f : p.alpha;
    float *stg = (float *)(smem + wave * Cfg::EPI_WAVE_BYTES);
    const int er = lane / 16, ec = (lane % 16) * 8;
    const bool vec_ok_e = (ldo * (int)sizeof(OutE)) % 16 == 0 && ((uintptr_t)out & 15) == 0 &&
                          (!cin || ((uintptr_t)cin & 15) == 0);
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      // the round's C is loaded first; staging the accumulators overlaps the loads
      v4f c_lo[8], c_hi[8];
      if (cin) {
#pragma unroll
        for (int k = 0; k < 8; ++k)
          load_c8<OutE>(cin, ldo, src.m_end, p.N, src.m0 + wm * 128 + 32 * q + er + 4 * k,
                        src.n0 + wn * 128 + ec, vec_ok_e, c_lo[k], c_hi[k]);
      }
#pragma unroll
      for (int ii = 0; ii < 2; ++ii)
#pragma unroll
        for (int j = 0; j < 8; ++j)
          *(v4f *)(stg + (16 * ii + r16) * Cfg::EPI_LD + 16 * j + 4 * g) =
              acc[2 * q + ii][j] * alpha;
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const int r = er + 4 * k;
        v4f lo = *(const v4f *)(stg + r * Cfg::EPI_LD + ec),
            hi = *(const v4f *)(stg + r * Cfg::EPI_LD + ec + 4);
        if (cin) {
          lo += c_lo[k];
          hi += c_hi[k];
        }
        store8<OutE>(out, nullptr, ldo, src.m_end, p.N, src.m0 + wm * 128 + 32 * q + r,
                     src.n0 + wn * 128 + ec, lo, hi, vec_ok_e);
      }
    }
  };

  if constexpr (!Cfg::K_GROUPED)
    set_geometry();
  if constexpr (Scheduler::DYNAMIC) {
    // device-side tile count: walk it in rounds of the grid
    const int tiles = Scheduler::num_tiles(p);
    const int first_lid = xcd_remap(blockIdx.x, gridDim.x);
    for (int lid = first_lid; lid < tiles; lid += gridDim.x) {
      nkt = KT;
      run_tile(Scheduler::tile(p, lid));
      asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
    }
  } else {
    const int tiles = Scheduler::num_tiles(p);
    const int splits = Cfg::SPLIT_K ? p.splits : 1;
    const int uid = xcd_remap(blockIdx.x, tiles * splits);
    const int kps = (KT + splits - 1) / splits;
    split = uid / tiles;
    lid0 = uid - split * tiles;
    kb = split * kps;
    nkt = min(KT - kb, kps);
    run_tile(Scheduler::tile(p, lid0));
  }
#endif
}
