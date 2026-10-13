// CDNA4 FP8 / FP4 and BF16 GEMMs for small M, where streaming B bounds the time.
// Every wave loads its scaled-MFMA operands straight from global memory into a ring of DEPTH
// register stages: no LDS and no barriers in the K loop. The waves of a workgroup take equal
// parts of the K range of the workgroup's output tile and add their tiles in LDS in wave order,
// so the result does not depend on timing. Grouped GEMMs (contiguous, PSUM and masked layouts)
// map tiles onto groups with the shared tile scheduler.
#pragma once
#include "epilogue.cuh"
#include "tile_scheduler.cuh"

// FM A fragments (M <= 16 FM per tile), NF B fragments (16 NF output columns per workgroup),
// DEPTH K steps in flight per wave, WAVES waves splitting the K range of a workgroup.
// SF_GRAN_128: bit 0 for A, bit 1 for B. K_TAIL: K is not a multiple of a K step.
// A_TYPE / B_TYPE: FP8 and FP4 operands mix in the scaled MFMA; BF16 operands (no scales) run two
// 16x16x32 BF16 MFMAs per fragment pair and step.
template <int FM_, int NF_, int DEPTH_, int WAVES_, int SF_GRAN_128 = 0,
          GemmType TYPE = GemmType::Normal, bool K_TAIL_ = false,
          OperandType A_TYPE = OperandType::Fp8, OperandType B_TYPE = OperandType::Fp8>
struct Mfma16x16x128SmallMConfig {
  static constexpr int FM = FM_, NF = NF_, DEPTH = DEPTH_;
  static constexpr bool BF16 = A_TYPE == OperandType::Bf16, K_TAIL = K_TAIL_;
  static_assert(BF16 == (B_TYPE == OperandType::Bf16), "BF16 operands come in pairs");
  static constexpr bool A_FP4 = A_TYPE == OperandType::Fp4, B_FP4 = B_TYPE == OperandType::Fp4;
  // A K step is BK K values: 128 bytes of an FP8 or BF16 row, 64 bytes of an FP4 row
  static constexpr int ES = BF16 ? 2 : 1, BK = 128 / ES;
  static constexpr int A_STEP = A_FP4 ? 64 : 128, B_STEP = B_FP4 ? 64 : 128;
  __host__ __device__ static constexpr int a_bytes(int k) { return A_FP4 ? (k + 1) / 2 : k * ES; }
  __host__ __device__ static constexpr int b_bytes(int k) { return B_FP4 ? (k + 1) / 2 : k * ES; }
  static constexpr int WAVES = WAVES_, NT = 64 * WAVES;
  static constexpr int BM = 16 * FM, BN = 16 * NF;
  static constexpr int SFA_SHIFT = SF_GRAN_128 & 1 ? 2 : 0, SFB_SHIFT = SF_GRAN_128 & 2 ? 2 : 0;
  static constexpr GemmType kType = TYPE;
  // all m tiles of a column block run next to each other, so that B is read from HBM once
  static constexpr int GROUP_M = 4096;
  using Scheduler = TileScheduler<TYPE, BM, BN, GROUP_M>;
  static constexpr bool MASKED = Scheduler::MASKED;
  static constexpr bool B_GROUPED = TYPE == GemmType::MGroupedContiguous ||
                                    TYPE == GemmType::MGroupedContiguousPsum || MASKED;
  static_assert(!Scheduler::K_GROUPED, "K-grouped GEMMs use the 256x256 kernel");
  // per-wave output tiles for the in-workgroup reduction, rows padded against bank conflicts
  static constexpr int TILE_LD = BN + 4;
  static constexpr int LDS_BYTES = WAVES * BM * TILE_LD * 4;
};

// Zero the bytes of a 16-byte K chunk at or past `valid` (a byte count, may be <= 0 or >= 16)
__device__ __forceinline__ v4i keep_first_bytes(v4i v, int valid) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    // a 64-bit shift: the mask is computed for every step, and a 32-bit shift by 32 is undefined
    const int keep = min(max(valid - 4 * i, 0), 4);
    v[i] &= (int)(uint32_t)((1ull << (8 * keep)) - 1);
  }
  return v;
}

template <class Cfg, class OutT>
__device__ __forceinline__ void fp8_gemm_mfma_16x16x128_small_m_impl(GemmArgs p) {
#if defined(__HIP_DEVICE_COMPILE__)
  constexpr int FM = Cfg::FM, NF = Cfg::NF, DEPTH = Cfg::DEPTH, BM = Cfg::BM, BN = Cfg::BN;
  constexpr int BK = Cfg::BK, WAVES = Cfg::WAVES;
  using Scheduler = typename Cfg::Scheduler;
  extern __shared__ __attribute__((aligned(16))) float tiles[];

  const int tid = threadIdx.x, lane = tid % 64, g = lane / 16, r16 = lane % 16;
  const int wave = __builtin_amdgcn_readfirstlane(tid / 64);

  // ---- K steps of this wave (the same for every tile)
  const int k_steps = (p.K + BK - 1) / BK;
  const int steps_per_wave = (k_steps + WAVES - 1) / WAVES;
  const int wave_begin = wave * steps_per_wave;
  const int nsteps = max(0, min(k_steps, wave_begin + steps_per_wave) - wave_begin);
  // K values of the last K step that hold data
  const int k_tail = p.K - (k_steps - 1) * BK;
  const uint64_t a_k4 = (uint64_t)((Cfg::a_bytes(p.K) + 3) & ~3);
  const uint64_t b_k4 = (uint64_t)((Cfg::b_bytes(p.K) + 3) & ~3);
  const int sa_words = (k_steps + (1 << Cfg::SFA_SHIFT) - 1) >> Cfg::SFA_SHIFT;
  const int sb_words = (k_steps + (1 << Cfg::SFB_SHIFT) - 1) >> Cfg::SFB_SHIFT;

  // ---- buffer descriptors: rows past the tile's valid rows, past N and past the matrix read 0
  auto rsrc_at = [&](const uint8_t *base, uint64_t bytes) {
    uint32_t hi = (uint32_t)(bytes >> 32);
    asm("" : "+s"(hi));
    return make_rsrc(base, hi ? 0xffffffffu : (uint32_t)bytes);
  };
  const __amdgpu_buffer_rsrc_t empty_rsrc = make_rsrc(p.A, 0);

  // One output tile: rows t.m0 .. t.m0 + BM - 1 of group t.group, columns t.n0 .. t.n0 + BN - 1.
  // Rows >= t.m_data are not read (zero products), rows >= t.m_end are not stored; a tile of PSUM
  // zero padding only stores zeros.
  auto run_tile = [&](const GroupTile &t) {
    const int m0 = t.m0, n0 = t.n0, rows = min(BM, t.m_data - m0);
    const size_t ga = Cfg::MASKED ? (size_t)t.group : 0, gb = Cfg::B_GROUPED ? (size_t)t.group : 0;
    const __amdgpu_buffer_rsrc_t a_rsrc = rsrc_at(
        p.A + ga * p.a_group_stride + (size_t)m0 * p.lda, (uint64_t)(rows - 1) * p.lda + a_k4);
    const __amdgpu_buffer_rsrc_t b_rsrc = rsrc_at(
        p.B + gb * p.b_group_stride + (size_t)n0 * p.ldb, (uint64_t)(p.N - n0 - 1) * p.ldb + b_k4);
    const __amdgpu_buffer_rsrc_t sa_rsrc =
        rsrc_at((const uint8_t *)(p.sfa + ga * p.sfa_group_stride + m0),
                ((uint64_t)(sa_words - 1) * p.lda_sf + rows) * 4);
    const __amdgpu_buffer_rsrc_t sb_rsrc =
        rsrc_at((const uint8_t *)(p.sfb + gb * p.sfb_group_stride + n0),
                ((uint64_t)(sb_words - 1) * p.ldb_sf + p.N - n0) * 4);

    // Lane (g, r16) holds row r16 of a fragment: bytes 16 g .. + 15 (half 0) and 64 + 16 g .. + 15
    // (half 1) of the step's 128 bytes, or of an FP4 row bytes 16 g .. + 15 (K values 32 g ..
    // + 31) of the step's 64, and the scale word of its row. Steps past the wave's range load
    // through an empty descriptor: zeros, no memory traffic, so the K loop has no branches (a
    // branch around loads makes every later wait drain all loads).
    v4i a_frag[DEPTH][FM][2], b_frag[DEPTH][NF][2];
    uint32_t a_code[DEPTH][FM], b_code[DEPTH][NF];
    auto load_step = [&](auto slot, int s) {
      constexpr int d = decltype(slot)::value;
      const bool in_range = s < nsteps;
      const int step = wave_begin + s;
      const __amdgpu_buffer_rsrc_t ar = in_range ? a_rsrc : empty_rsrc;
      const __amdgpu_buffer_rsrc_t br = in_range ? b_rsrc : empty_rsrc;
      const __amdgpu_buffer_rsrc_t sar = in_range ? sa_rsrc : empty_rsrc;
      const __amdgpu_buffer_rsrc_t sbr = in_range ? sb_rsrc : empty_rsrc;
      const int a_koff = step * Cfg::A_STEP, b_koff = step * Cfg::B_STEP;
      const int sa_off = (step >> Cfg::SFA_SHIFT) * p.lda_sf * 4;
      const int sb_off = (step >> Cfg::SFB_SHIFT) * p.ldb_sf * 4;
#pragma unroll
      for (int j = 0; j < NF; ++j) {
        const uint32_t row = (uint32_t)(16 * j + r16) * p.ldb + 16 * g;
        b_frag[d][j][0] = (v4i)__builtin_amdgcn_raw_buffer_load_b128(br, row, b_koff, 0);
        if constexpr (!Cfg::B_FP4)
          b_frag[d][j][1] = (v4i)__builtin_amdgcn_raw_buffer_load_b128(br, row + 64, b_koff, 0);
        if constexpr (!Cfg::BF16)
          b_code[d][j] = __builtin_amdgcn_raw_buffer_load_b32(sbr, (16 * j + r16) * 4, sb_off, 0);
      }
#pragma unroll
      for (int i = 0; i < FM; ++i) {
        const uint32_t row = (uint32_t)(16 * i + r16) * p.lda + 16 * g;
        a_frag[d][i][0] = (v4i)__builtin_amdgcn_raw_buffer_load_b128(ar, row, a_koff, 0);
        if constexpr (!Cfg::A_FP4)
          a_frag[d][i][1] = (v4i)__builtin_amdgcn_raw_buffer_load_b128(ar, row + 64, a_koff, 0);
        if constexpr (!Cfg::BF16)
          a_code[d][i] = __builtin_amdgcn_raw_buffer_load_b32(sar, (16 * i + r16) * 4, sa_off, 0);
      }
    };
    // The last K step of a K that is not a multiple of a step reads past K (the next row or row
    // padding, which may hold NaN codes): those bytes become 0. The masks apply to every step (all
    // ones before the last), as selects in the loop are cheaper than the copies a conditional
    // block costs.
    auto clear_tail = [&](auto slot, bool last) {
      constexpr int d = decltype(slot)::value;
      const int a_tail = Cfg::a_bytes(k_tail), b_tail = Cfg::b_bytes(k_tail);
      const int a0 = last ? a_tail - 16 * g : 16, a1 = last ? a_tail - 64 - 16 * g : 16;
      const int b0 = last ? b_tail - 16 * g : 16, b1 = last ? b_tail - 64 - 16 * g : 16;
#pragma unroll
      for (int i = 0; i < FM; ++i) {
        a_frag[d][i][0] = keep_first_bytes(a_frag[d][i][0], a0);
        if constexpr (!Cfg::A_FP4)
          a_frag[d][i][1] = keep_first_bytes(a_frag[d][i][1], a1);
      }
#pragma unroll
      for (int j = 0; j < NF; ++j) {
        b_frag[d][j][0] = keep_first_bytes(b_frag[d][j][0], b0);
        if constexpr (!Cfg::B_FP4)
          b_frag[d][j][1] = keep_first_bytes(b_frag[d][j][1], b1);
      }
    };

    // acc[i][j][e] of lane (g, r16): D[m0 + 16 i + r16][n0 + 16 j + 4 g + e]
    v4f acc[FM][NF];
#pragma unroll
    for (int i = 0; i < FM; ++i)
#pragma unroll
      for (int j = 0; j < NF; ++j)
        acc[i][j] = v4f{};
    // The MFMA reads the low byte of a scale register: the code of K block g, or (128-wide
    // blocks) the step's byte, shifted down. In the last step of a partial K, K blocks without
    // data get code 127 (scale 1) so that a padding code cannot give NaN.
    const bool no_data_block = 32 * g >= k_tail;
    auto compute_step = [&](auto slot, int s) {
      constexpr int d = decltype(slot)::value;
      if constexpr (Cfg::BF16) {
#pragma unroll
        for (int j = 0; j < NF; ++j)
#pragma unroll
          for (int i = 0; i < FM; ++i) {
            acc[i][j] = mfma_bf16_16x16x32(b_frag[d][j][0], a_frag[d][i][0], acc[i][j]);
            acc[i][j] = mfma_bf16_16x16x32(b_frag[d][j][1], a_frag[d][i][1], acc[i][j]);
          }
        return;
      }
      const int step = wave_begin + s;
      const bool no_data = Cfg::K_TAIL && step == k_steps - 1 && no_data_block;
      const int a_shift = 8 * (Cfg::SFA_SHIFT ? step & 3 : g), b_shift = 8 * (Cfg::SFB_SHIFT ? step & 3 : g);
      uint32_t a_c[FM], b_c[NF];
#pragma unroll
      for (int i = 0; i < FM; ++i) {
        a_c[i] = a_code[d][i] >> a_shift;
        if constexpr (Cfg::K_TAIL && !Cfg::SFA_SHIFT)
          a_c[i] = no_data ? 127u : a_c[i];
      }
#pragma unroll
      for (int j = 0; j < NF; ++j) {
        b_c[j] = b_code[d][j] >> b_shift;
        if constexpr (Cfg::K_TAIL && !Cfg::SFB_SHIFT)
          b_c[j] = no_data ? 127u : b_c[j];
      }
#pragma unroll
      for (int j = 0; j < NF; ++j)
#pragma unroll
        for (int i = 0; i < FM; ++i)
          acc[i][j] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
              scaled_operand<Cfg::B_FP4>(b_frag[d][j]), scaled_operand<Cfg::A_FP4>(a_frag[d][i]),
              acc[i][j], Cfg::B_FP4 ? kFmtFp4 : kFmtFp8, Cfg::A_FP4 ? kFmtFp4 : kFmtFp8, 0, b_c[j], 0,
              a_c[i]);
    };

    // ---- K loop: step s sits in register stage s % DEPTH; steps past nsteps add zeros. Loads
    // complete in issue order, so stage s is waited for while DEPTH - 1 stages stay in flight, as
    // long as the scheduler keeps the stages in order (sched_barrier).
    if (rows > 0) {
      static_for<0, DEPTH>([&](auto D) {
        load_step(D, decltype(D)::value);
        __builtin_amdgcn_sched_barrier(0);
      });
      for (int s0 = 0; s0 < nsteps; s0 += DEPTH) {
        static_for<0, DEPTH>([&](auto D) {
          const int s = s0 + decltype(D)::value;
          if constexpr (Cfg::K_TAIL)
            clear_tail(D, wave_begin + s == k_steps - 1);
          compute_step(D, s);
          load_step(D, s + DEPTH);
          __builtin_amdgcn_sched_barrier(0);
        });
      }
    }

    // ---- in-workgroup reduction: wave tiles added in wave order
    float *const my_tile = tiles + wave * BM * Cfg::TILE_LD;
#pragma unroll
    for (int i = 0; i < FM; ++i)
#pragma unroll
      for (int j = 0; j < NF; ++j)
        *(v4f *)(my_tile + (16 * i + r16) * Cfg::TILE_LD + 16 * j + 4 * g) = acc[i][j];
    __syncthreads();
    uint8_t *const out = (uint8_t *)p.D + (Cfg::MASKED ? ga * p.d_group_stride * sizeof(OutT) : 0);
    const bool vec_ok = (p.ldd * (int)sizeof(OutT)) % 16 == 0 && ((uintptr_t)out & 15) == 0 &&
                        ((uintptr_t)p.C & 15) == 0;
    for (int e = tid; e < BM * BN / 4; e += Cfg::NT) {
      const int row = e / (BN / 4), col = (e % (BN / 4)) * 4;
      v4f s = *(const v4f *)(tiles + row * Cfg::TILE_LD + col);
#pragma unroll
      for (int w = 1; w < WAVES; ++w)
        s += *(const v4f *)(tiles + (w * BM + row) * Cfg::TILE_LD + col);
      store4<OutT>(out, p.C, p.ldd, t.m_end, p.N, m0 + row, n0 + col, s * p.alpha, vec_ok);
    }
  };

  if constexpr (Scheduler::DYNAMIC) {
    // workgroup (group, slot, column block): m tiles slot, slot + group_m_slots, ... of the group,
    // whose m tile count is known only on the device
    const int tiles_n = (p.N + BN - 1) / BN, slots = p.group_m_slots;
    const int tn = blockIdx.x % tiles_n, rest = blockIdx.x / tiles_n;
    const int slot = rest % slots, group = rest / slots;
    const int tiles_m = Scheduler::group_tiles_m(p, group);
    GroupTile t{0, tn * BN, group, 0, 0};
    int start = 0;
    if constexpr (Cfg::MASKED) {
      t.m_end = t.m_data = p.grouped_layout[group];
    } else {
      start = Scheduler::psum_start(p, group);
      t.m_end = Scheduler::psum_stored_end(p, group);
      t.m_data = p.grouped_layout[group];
    }
    for (int tm = slot; tm < tiles_m; tm += slots) {
      t.m0 = start + tm * BM;
      run_tile(t);
      __syncthreads();
    }
  } else {
    const GroupTile t = Scheduler::tile(p, blockIdx.x);
    // tiles of padding rows only
    if constexpr (Cfg::kType == GemmType::MGroupedContiguous)
      if (p.grouped_layout[t.m0] < 0)
        return;
    run_tile(t);
  }
#endif
}
