// AMD GPU projection experiment with wider native MFMA output fragments.
// FP8 scale semantics remain one UE8M0 code per 32 K values.
#pragma once
#include <deep_gemm/impls/cdna4/device_common.h>
namespace deep_gemm::cdna4::projection {
using acc32_t = float __attribute__((ext_vector_type(16)));
template <class T>
__device__ __forceinline__ T projection_lds_read(const __attribute__((address_space(3))) T *__restrict__ p) {
    return *p;
}
#if defined(__HIP_DEVICE_COMPILE__)
template <bool MANUAL> __device__ __forceinline__ auto projection_resource(const void *base, uint32_t bytes) {
    if constexpr (MANUAL) {
        const uintptr_t address = uintptr_t(base);
        return v4i{int(uint32_t(address)), int(uint32_t(address >> 32)), int(bytes), 0x00020000};
    } else
        return make_rsrc(base, bytes);
}
template <bool MANUAL, int BYTES, int CACHE = 0, class Resource>
__device__ __forceinline__ void projection_dma(Resource resource, lds_ptr_t destination, unsigned vector_offset,
                                               unsigned scalar_offset) {
    if constexpr (!MANUAL) {
        static_assert(CACHE != 256 && CACHE != 512 && CACHE != 1024,
                      "manual modifier tags are not intrinsic policy bits");
        static_assert(BYTES == 4 || BYTES == 16);
        if constexpr (BYTES == 16)
            __builtin_amdgcn_raw_ptr_buffer_load_lds(resource, destination, 16, vector_offset, scalar_offset, 0, CACHE);
        else
            __builtin_amdgcn_raw_ptr_buffer_load_lds(resource, destination, 4, vector_offset, scalar_offset, 0, CACHE);
    } else {
        // Manual-only tags outside intrinsic policy values:256=sc0,512=sc1,1024=nt.
        static_assert((CACHE == 0 || CACHE == 256 || CACHE == 512 || CACHE == 1024) && (BYTES == 4 || BYTES == 16));
        static_assert(BYTES == 16 || CACHE == 0);
        const unsigned address = __builtin_amdgcn_readfirstlane(uintptr_t(destination));
        // Completion and cross-wave joins are owned by the enclosing pipeline.
        // Descriptor bounds and lane-relative LDS destinations stay unchanged.
        if constexpr (BYTES == 16 && CACHE == 256)
            asm volatile("s_mov_b32 m0, %0\n\ts_nop 0\n\tbuffer_load_dwordx4 %1, %2, %3 offen sc0 lds" ::"s"(address),
                         "v"(vector_offset), "s"(resource), "s"(scalar_offset)
                         : "m0", "memory");
        else if constexpr (BYTES == 16 && CACHE == 512)
            asm volatile("s_mov_b32 m0, %0\n\ts_nop 0\n\tbuffer_load_dwordx4 %1, %2, %3 offen sc1 lds" ::"s"(address),
                         "v"(vector_offset), "s"(resource), "s"(scalar_offset)
                         : "m0", "memory");
        else if constexpr (BYTES == 16 && CACHE == 1024)
            asm volatile("s_mov_b32 m0, %0\n\ts_nop 0\n\tbuffer_load_dwordx4 %1, %2, %3 offen nt lds" ::"s"(address),
                         "v"(vector_offset), "s"(resource), "s"(scalar_offset)
                         : "m0", "memory");
        else if constexpr (BYTES == 16)
            asm volatile("s_mov_b32 m0, %0\n\ts_nop 0\n\tbuffer_load_dwordx4 %1, %2, %3 offen lds" ::"s"(address),
                         "v"(vector_offset), "s"(resource), "s"(scalar_offset)
                         : "m0", "memory");
        else
            asm volatile("s_mov_b32 m0, %0\n\ts_nop 0\n\tbuffer_load_dword %1, %2, %3 offen lds" ::"s"(address),
                         "v"(vector_offset), "s"(resource), "s"(scalar_offset)
                         : "m0", "memory");
    }
}
template <bool MANUAL, int COUNT> __device__ __forceinline__ void projection_wait_vm() {
    // Inline-assembly copies are opaque to compiler VMEM tracking. Preserve
    // their software-owned waits as explicit instructions as well.
    if constexpr (MANUAL)
        asm volatile("s_waitcnt %0" ::"n"(waitcnt_vm(COUNT)) : "memory");
    else
        __builtin_amdgcn_s_waitcnt(waitcnt_vm(COUNT));
}
template <bool EXACT, bool MANUAL, int COUNT> __device__ __forceinline__ void projection_wait_ready(int wave) {
    if constexpr (EXACT) {
        static_assert(MANUAL && COUNT == 4);
        // In the restricted layout below, waves 0/1 copy one scale plane in
        // addition to four operand vectors; waves 2/3 copy only the operands.
        if (wave < 2)
            projection_wait_vm<true, 5>();
        else
            projection_wait_vm<true, 4>();
    } else
        projection_wait_vm<MANUAL, COUNT>();
}
#endif
template <bool BF16, int KP = 1, int F = 1, int GROUP_M = 1, bool DMA_SCALES = false, int SPLITS = 1,
          bool FULL_BANK_SWIZZLE = false, int STAGES = 2, int STRIPES = 1, int WAVE_M = 2, int CACHE_A = 0,
          int CACHE_B = 0, int REGISTER_TILE = 0, int FN = F, bool SCALAR_K_OFFSET = false, bool PACKED_B = false,
          bool UNIFORM_SCALE_DMA = false, int WAVES = 4, bool PACKED_A = false, int UNROLL = 1,
          bool SCHEDULE_FENCE = false, bool UNIFORM_SCALE_LOOP = false, bool LATE_SCALE_ADDRESS = false,
          int ACC_CHAINS = 1, bool READ_FIRST = false, bool FORCE_ACC = false, int FRAG_DEPTH = 0, int GROUP_N = 0,
          int B_ROW_PADDING = 0, int IGLP = -1, int COALESCED_OUTPUT = 0, bool PACKED_OUTPUT = false,
          bool WAVE_OUTPUT = false, bool SEPARATE_DMA = false, bool REBASE_INPUTS = false, bool SHARED_B_SCALES = false,
          bool DISTRIBUTED_B_SCALES = false, bool EXPLICIT_LDS_READS = false, bool MANUAL_LDS_DMA = false,
          bool EXACT_WAVE_WAIT = false, int WAIT_FENCE = 0, bool FRAGMENT_SCHEDULE_FENCES = true>
__device__ __forceinline__ void run_m32(const void *ap, const void *bp, const int *sa, const int *sb, void *dp, int m,
                                        int n, int k, int sap, int sbp, int ldd) {
#if defined(__HIP_DEVICE_COMPILE__)
    static_assert((WAVES == 1 || WAVES == 2 || WAVES == 4 || WAVES == 8) &&
                  (WAVE_M == 1 || WAVE_M == 2 || WAVE_M == 4) && WAVES % WAVE_M == 0);
    constexpr int THREADS = 64 * WAVES, WAVE_N = WAVES / WAVE_M, BM = 32 * WAVE_M * F, BN = 32 * WAVE_N * FN,
                  ROWS = BM + BN, CHUNKS = 8 * KP, ES = BF16 ? 2 : 1, BK = 128 * KP / ES;
    static_assert(!UNIFORM_SCALE_DMA || (BM == 64 && BN == 64 && DMA_SCALES));
    static_assert(!MANUAL_LDS_DMA || (SEPARATE_DMA && UNIFORM_SCALE_LOOP && DMA_SCALES && !SHARED_B_SCALES));
    static_assert(!(MANUAL_LDS_DMA && UNIFORM_SCALE_DMA) ||
                  (!BF16 && KP == 1 && WAVES == 4 && STAGES == 2 && REGISTER_TILE && !EXACT_WAVE_WAIT));
    constexpr int READY_COPIES = ROWS * CHUNKS / THREADS + ((MANUAL_LDS_DMA && UNIFORM_SCALE_DMA) ? 2 : 0);
    static_assert(!EXACT_WAVE_WAIT || (!BF16 && KP == 1 && BM == 64 && BN == 64 && WAVES == 4 && STAGES == 2 &&
                                       REGISTER_TILE && MANUAL_LDS_DMA));
    constexpr int SCALE_A = UNIFORM_SCALE_DMA ? 256 : (BM + 63) / 64 * 64;
    constexpr int SCALE_ROWS = SCALE_A + (UNIFORM_SCALE_DMA ? 256 : (BN + 63) / 64 * 64);
    static_assert(!(PACKED_A || PACKED_B) || (BM == 64 && BN == 64 && !FULL_BANK_SWIZZLE));
    static_assert(STAGES == 2 || (STAGES == 3 && DMA_SCALES) || ((STAGES == 1 || STAGES == 4) && REGISTER_TILE));
    static_assert(UNROLL == 1 || ((UNROLL == 2 || UNROLL == 3 || UNROLL == 4 || UNROLL == 6 || UNROLL == 8 ||
                                   UNROLL == 10 || UNROLL == 16 || UNROLL == 20 || UNROLL == 40) &&
                                  UNROLL % STAGES == 0));
    static_assert(!REGISTER_TILE || (STAGES >= 1 && STAGES <= 4 && DMA_SCALES));
    static_assert(REGISTER_TILE >= 0 && REGISTER_TILE <= 2);
    static_assert(REGISTER_TILE != 2 || (STAGES == 2 && !EXACT_WAVE_WAIT && !SHARED_B_SCALES));
    static_assert(!READ_FIRST || (!REGISTER_TILE && F == 1 && FN == 1 && STAGES == 2));
    static_assert(WAIT_FENCE >= 0 && WAIT_FENCE <= 2);
    static_assert(IGLP < 0 || ((IGLP == 0 || IGLP == 1) && UNROLL == 1 && !SCHEDULE_FENCE && !READ_FIRST &&
                               FRAG_DEPTH == 0 && WAIT_FENCE == 0));
    static_assert(STAGES * (ROWS * CHUNKS * 16 + SCALE_ROWS * KP * 4) <= 160 * 1024);
    __shared__ __attribute__((aligned(16))) v4i tiles[STAGES][ROWS * CHUNKS];
    __shared__ unsigned scales[STAGES][SCALE_ROWS * KP];
    // Keep the LDS address space explicit across the read helper. All waits and
    // joins below remain mandatory; this does not weaken stage ownership.
    auto read_tile = [&](int phase, int index) {
        if constexpr (EXPLICIT_LDS_READS)
            return projection_lds_read((const __attribute__((address_space(3))) v4i *)(&tiles[phase][index]));
        else
            return tiles[phase][index];
    };
    auto read_scale = [&](int phase, int index) {
        if constexpr (EXPLICIT_LDS_READS)
            return projection_lds_read((const __attribute__((address_space(3))) unsigned *)(&scales[phase][index]));
        else
            return scales[phase][index];
    };
    auto join = []() {
        if constexpr (WAVES == 1)
            __builtin_amdgcn_wave_barrier();
        else
            asm volatile("s_barrier" ::: "memory");
    };
    const int tid = threadIdx.x, wave = __builtin_amdgcn_readfirstlane(tid / 64), lane = tid % 64, r = lane % 32,
              g = lane / 32;
    int linear = blockIdx.y * gridDim.x + blockIdx.x;
    if constexpr (STRIPES > 1) {
        // A bijective partition/interleave permutation. This is only an ordering
        // hint; correctness never assumes any physical block placement.
        const int count = gridDim.x * gridDim.y, per = (count + STRIPES - 1) / STRIPES;
        const int tall = count % STRIPES ? count % STRIPES : STRIPES;
        const int stripe = linear % STRIPES, local = linear / STRIPES;
        linear = stripe < tall ? stripe * per + local : tall * per + (stripe - tall) * (per - 1) + local;
    }
    const int first_m = (linear / (GROUP_M * gridDim.x)) * GROUP_M;
    const int group_size = min(int(gridDim.y) - first_m, GROUP_M);
    const int within = linear % (GROUP_M * gridDim.x);
    int tile_m = first_m + within % group_size, tile_n = within / group_size;
    if constexpr (GROUP_N > 0) {
        // Subdivide a row-group into compact rectangles, including the partial
        // final column group. This remains a bijection for any tile dimensions.
        const int first_n = (within / (group_size * GROUP_N)) * GROUP_N;
        const int columns = min(int(gridDim.x) - first_n, GROUP_N);
        const int offset = within % (group_size * GROUP_N);
        tile_m = first_m + offset / columns;
        tile_n = first_n + offset % columns;
    }
    const int m0 = tile_m * BM, n0 = tile_n * BN, wm = wave / WAVE_N, wn = wave % WAVE_N;
    static_assert(B_ROW_PADDING >= 0 && B_ROW_PADDING % 16 == 0 && (!PACKED_B || B_ROW_PADDING == 0));
    static_assert(!REBASE_INPUTS || (SEPARATE_DMA && !PACKED_A && !PACKED_B));
    auto ar = projection_resource<MANUAL_LDS_DMA>((const uint8_t *)ap + (REBASE_INPUTS ? size_t(m0) * k * ES : 0),
                                                  uint32_t(size_t(REBASE_INPUTS ? min(BM, m - m0) : m) * k * ES));
    auto br = projection_resource<MANUAL_LDS_DMA>(
        (const uint8_t *)bp + (REBASE_INPUTS ? size_t(n0) * (k + B_ROW_PADDING) * ES : 0),
        uint32_t(size_t(REBASE_INPUTS ? min(BN, n - n0) : (PACKED_B ? (n + 63) / 64 * 64 : n)) * (k + B_ROW_PADDING) *
                 ES));
    auto sar = projection_resource<MANUAL_LDS_DMA>(sa, uint32_t((uint64_t(k / 128 - 1) * sap + m) * 4));
    auto sbr = projection_resource<MANUAL_LDS_DMA>(sb, uint32_t((uint64_t(k / 128 - 1) * sbp + n) * 4));
    const auto bank_key = [](int row) {
        // Wider K rows remove the alternating bank-half offset supplied by
        // 128-byte rows. Include that row bit in the XOR instead of dropping it.
        if constexpr (FULL_BANK_SWIZZLE && KP > 1)
            return row & (CHUNKS - 1);
        else
            return (row >> 1) & 7;
    };
    // Opt-in ABI: B scale rows are compact [ceil(N/32),K/128] packed words.
    // The host must validate shared scale rows before selecting this contract.
    static_assert(!SHARED_B_SCALES || (!BF16 && F == 1 && FN == 1 && DMA_SCALES && !READ_FIRST));
    unsigned shared_b_codes[STAGES][KP];
    // Additional opt-in ABI: compact row-major B scale words. Each lane owns
    // one K word, so K/128 must fit in one wave (validated by the host).
    static_assert(!DISTRIBUTED_B_SCALES || (SHARED_B_SCALES && KP == 1));
    unsigned distributed_b_code = 0x7f7f7f7fu;
    if constexpr (DISTRIBUTED_B_SCALES)
        if (lane < k / 128)
            distributed_b_code = sb[((n0 + wn * 32) / 32) * sbp + lane];
    auto dma = [&](int step, int phase) {
        if constexpr (SHARED_B_SCALES) {
#pragma unroll
            for (int part = 0; part < KP; ++part) {
                if constexpr (DISTRIBUTED_B_SCALES)
                    shared_b_codes[phase][part] = __shfl(distributed_b_code, step * KP + part);
                else
                    shared_b_codes[phase][part] = sb[(step * KP + part) * sbp + (n0 + wn * 32) / 32];
            }
        }
        if constexpr (SEPARATE_DMA) {
            static_assert(!PACKED_A && BM * CHUNKS % THREADS == 0 && BN * CHUNKS % THREADS == 0);
            const unsigned scalar = __builtin_amdgcn_readfirstlane(step * 128 * KP);
#pragma unroll
            for (int i = 0; i < BM * CHUNKS / THREADS; ++i) {
                const int row = (tid + i * THREADS) / CHUNKS, chunk = (tid % CHUNKS) ^ bank_key(row);
                projection_dma<MANUAL_LDS_DMA, 16, CACHE_A>(ar, (lds_ptr_t)(&tiles[phase][i * THREADS + wave * 64]),
                                                            ((REBASE_INPUTS ? 0 : m0) + row) * k * ES + chunk * 16,
                                                            scalar);
            }
#pragma unroll
            for (int i = 0; i < BN * CHUNKS / THREADS; ++i) {
                const int row = (tid + i * THREADS) / CHUNKS, chunk = (tid % CHUNKS) ^ bank_key(BM + row);
                if constexpr (PACKED_B) {
                    const unsigned packed_base =
                        __builtin_amdgcn_readfirstlane(((n0 / 64) * (k / BK) + step) * 64 * CHUNKS * 16);
                    projection_dma<MANUAL_LDS_DMA, 16, CACHE_B>(
                        br, (lds_ptr_t)(&tiles[phase][BM * CHUNKS + i * THREADS + wave * 64]), (tid + i * THREADS) * 16,
                        packed_base);
                } else {
                    projection_dma<MANUAL_LDS_DMA, 16, CACHE_B>(
                        br, (lds_ptr_t)(&tiles[phase][BM * CHUNKS + i * THREADS + wave * 64]),
                        ((REBASE_INPUTS ? 0 : n0) + row) * (k + B_ROW_PADDING) * ES + chunk * 16, scalar);
                }
            }
        } else {
#pragma unroll
            for (int i = 0; i < ROWS * CHUNKS / THREADS; ++i) {
                int row = (tid + i * THREADS) / CHUNKS, chunk = (tid % CHUNKS) ^ bank_key(row);
                bool a = __builtin_amdgcn_readfirstlane(row < BM);
                unsigned off = ((a ? m0 + row : n0 + row - BM) * (a ? k : k + B_ROW_PADDING) * ES) + chunk * 16;
                unsigned scalar = 0;
                if constexpr (SCALAR_K_OFFSET)
                    scalar = __builtin_amdgcn_readfirstlane(step * 128 * KP);
                else
                    off += step * 128 * KP;
                if (a) {
                    if constexpr (PACKED_A) {
                        off = (row * CHUNKS + (tid % CHUNKS)) * 16;
                        scalar = __builtin_amdgcn_readfirstlane(((m0 / 64) * (k / BK) + step) * 64 * CHUNKS * 16);
                    }
                    __builtin_amdgcn_raw_ptr_buffer_load_lds(ar, (lds_ptr_t)(&tiles[phase][i * THREADS + wave * 64]),
                                                             16, off, scalar, 0, CACHE_A);
                } else {
                    if constexpr (PACKED_B) {
                        // Immutable weights are stored in exactly the physical LDS order.
                        // Every wave now transfers contiguous vectors from a packed tile.
                        off = ((row - BM) * CHUNKS + (tid % CHUNKS)) * 16;
                        scalar = __builtin_amdgcn_readfirstlane(((n0 / 64) * (k / BK) + step) * 64 * CHUNKS * 16);
                    }
                    __builtin_amdgcn_raw_ptr_buffer_load_lds(br, (lds_ptr_t)(&tiles[phase][i * THREADS + wave * 64]),
                                                             16, off, scalar, 0, CACHE_B);
                }
            }
        }
        if constexpr (!BF16) {
            if constexpr (SHARED_B_SCALES) {
#pragma unroll
                for (int base = wave * 64; base < SCALE_A * KP; base += THREADS) {
                    const int part = base / SCALE_A, local = base % SCALE_A;
                    __builtin_amdgcn_raw_ptr_buffer_load_lds(
                        sar, (lds_ptr_t)(&scales[phase][part * SCALE_ROWS + local]), 4, lane * 4,
                        ((step * KP + part) * sap + m0 + local) * 4, 0, 0);
                }
            } else if constexpr (UNIFORM_SCALE_DMA) {
                // Each wave writes its own private padding rows. Only the first
                // wave's 64 rows are consumed, with no overlapping LDS writes.
#pragma unroll
                for (int part = 0; part < KP; ++part) {
                    const int off = __builtin_amdgcn_readfirstlane(step * KP + part);
                    projection_dma<MANUAL_LDS_DMA, 4>(sar, (lds_ptr_t)(&scales[phase][part * SCALE_ROWS + wave * 64]),
                                                      (m0 + lane) * 4, off * sap * 4);
                    projection_dma<MANUAL_LDS_DMA, 4>(
                        sbr, (lds_ptr_t)(&scales[phase][part * SCALE_ROWS + SCALE_A + wave * 64]), (n0 + lane) * 4,
                        off * sbp * 4);
                }
            } else if constexpr (UNIFORM_SCALE_LOOP) {
                static_assert(DMA_SCALES);
                // Plane sizes are wave-aligned. Branch on an explicitly uniform
                // wave base, not the per-thread index, to avoid divergent EXEC masks.
#pragma unroll
                for (int base = wave * 64; base < SCALE_ROWS * KP; base += THREADS) {
                    const int local = base % SCALE_ROWS, part = base / SCALE_ROWS;
                    const bool a = local < SCALE_A;
                    unsigned scalar;
                    if constexpr (LATE_SCALE_ADDRESS) {
                        // Keep transient column offsets local to their copy instruction.
                        // Arithmetic is still exact unsigned byte addressing, not fast math.
                        const unsigned pitch = a ? sap : sbp, column = step * KP + part;
                        const unsigned row = (a ? m0 : n0) + local - (a ? 0 : SCALE_A);
                        asm volatile("s_mul_i32 %0, %1, %2\n\ts_add_u32 %0, %0, %3\n\ts_lshl_b32 %0, %0, 2"
                                     : "=&s"(scalar)
                                     : "s"(pitch), "s"(column), "s"(row)
                                     : "scc");
                    } else
                        scalar = ((step * KP + part) * (a ? sap : sbp) + (a ? m0 : n0) + local - (a ? 0 : SCALE_A)) * 4;
                    projection_dma<MANUAL_LDS_DMA, 4>(a ? sar : sbr, (lds_ptr_t)(&scales[phase][base]), lane * 4,
                                                      scalar);
                }
            } else {
#pragma unroll
                for (int index = tid; index < SCALE_ROWS * KP; index += THREADS) {
                    int local = index % SCALE_ROWS, part = index / SCALE_ROWS,
                        row = local < SCALE_A ? m0 + local : n0 + local - SCALE_A;
                    if constexpr (DMA_SCALES) {
                        const bool a = __builtin_amdgcn_readfirstlane(local < SCALE_A);
                        // All lanes of a wave cover contiguous rows in one scale column.
                        // The destination is wave-uniform; the intrinsic adds lane*4.
                        __builtin_amdgcn_raw_ptr_buffer_load_lds(
                            a ? sar : sbr, (lds_ptr_t)(&scales[phase][(index / 64) * 64]), 4, row * 4,
                            __builtin_amdgcn_readfirstlane((step * KP + part) * (a ? sap : sbp) * 4), 0, 0);
                    } else
                        scales[phase][index] =
                            row < (local < SCALE_A ? m : n)
                                ? unsigned((local < SCALE_A
                                                ? sa
                                                : sb)[(step * KP + part) * (local < SCALE_A ? sap : sbp) + row])
                                : 0x7f7f7f7fu;
                }
            }
        }
    };
    static_assert(ACC_CHAINS == 1 || ACC_CHAINS == 2 || ACC_CHAINS == 4);
    acc32_t accum[ACC_CHAINS][F][FN] = {};
    const int begin = SPLITS == 1 ? 0 : (k / BK) * blockIdx.z / SPLITS;
    const int end = SPLITS == 1 ? k / BK : (k / BK) * (blockIdx.z + 1) / SPLITS;
    dma(begin, 0);
    if constexpr (REGISTER_TILE && STAGES > 2) {
        // Fill the complete lookahead ring before retiring the first stage.
#pragma unroll
        for (int phase = 1; phase < STAGES; ++phase)
            if (begin + phase < end)
                dma(begin + phase, phase);
        if (begin + STAGES - 1 < end)
            projection_wait_vm<MANUAL_LDS_DMA, (STAGES - 1) * ROWS * CHUNKS / THREADS>();
        else
            projection_wait_vm<MANUAL_LDS_DMA, 0>();
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        join();
    } else if constexpr (STAGES == 3 || (REGISTER_TILE == 1 && STAGES == 2)) {
        if (begin + 1 < end)
            dma(begin + 1, 1);
        // Each wave issues at least this many vector DMA operations per stage.
        // Waiting below that count completes the older stage, even on waves
        // which additionally copied scales. No vector stores are in this loop.
        if (begin + 1 < end)
            projection_wait_ready<EXACT_WAVE_WAIT, MANUAL_LDS_DMA, READY_COPIES>(wave);
        else
            projection_wait_vm<MANUAL_LDS_DMA, 0>();
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        join();
    } else {
        asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
        join();
    }
    auto iteration = [&](int step, int phase) {
        const int future = step + (REGISTER_TILE == 1 ? STAGES : STAGES - 1);
        if constexpr (REGISTER_TILE == 2) {
            // The previous iteration's end join retired this other buffer. Refill
            // it before reading the current stage, never overwriting live inputs.
            if (future < end)
                dma(future, phase ^ 1);
        }
        v4i first_a, first_b, first_ah, first_bh;
        unsigned first_ac = 0, first_bc = 0;
        if constexpr (READ_FIRST) {
            const int arow = wm * 32 + r, brow = BM + wn * 32 + r;
            first_a = read_tile(phase, arow * CHUNKS + (g ^ bank_key(arow)));
            first_b = read_tile(phase, brow * CHUNKS + (g ^ bank_key(brow)));
            if constexpr (!BF16) {
                first_ah = read_tile(phase, arow * CHUNKS + ((g + 2) ^ bank_key(arow)));
                first_bh = read_tile(phase, brow * CHUNKS + ((g + 2) ^ bank_key(brow)));
                first_ac = read_scale(phase, arow) >> (8 * g);
                first_bc = read_scale(phase, SCALE_A + brow - BM) >> (8 * g);
            }
            // Compiler scheduling boundary only; no new runtime wait or barrier.
            __builtin_amdgcn_sched_barrier(0);
        }
        v4i prefetched_a[F][4 * KP], prefetched_b[FN][4 * KP];
        unsigned prefetched_sa[F][KP], prefetched_sb[FN][KP];
        if constexpr (REGISTER_TILE) {
            // Retire each distinct operand fragment once, then reuse it across
            // the wave's output tiles while the next LDS stage is in flight.
#pragma unroll
            for (int i = 0; i < F; ++i) {
                const int arow = wm * (32 * F) + i * 32 + r;
#pragma unroll
                for (int q = 0; q < 4 * KP; ++q) {
                    const int chunk = q * 2 + g;
                    prefetched_a[i][q] = read_tile(phase, arow * CHUNKS + (chunk ^ bank_key(arow)));
                }
                if constexpr (!BF16) {
#pragma unroll
                    for (int part = 0; part < KP; ++part)
                        prefetched_sa[i][part] = read_scale(phase, part * SCALE_ROWS + arow);
                }
            }
#pragma unroll
            for (int j = 0; j < FN; ++j) {
                const int brow = BM + wn * (32 * FN) + j * 32 + r;
#pragma unroll
                for (int q = 0; q < 4 * KP; ++q) {
                    const int chunk = q * 2 + g;
                    prefetched_b[j][q] = read_tile(phase, brow * CHUNKS + (chunk ^ bank_key(brow)));
                }
                if constexpr (!BF16) {
#pragma unroll
                    for (int part = 0; part < KP; ++part) {
                        if constexpr (SHARED_B_SCALES)
                            prefetched_sb[j][part] = shared_b_codes[phase][part];
                        else
                            prefetched_sb[j][part] = read_scale(phase, part * SCALE_ROWS + SCALE_A + brow - BM);
                    }
                }
            }
            // All waves finish their current-stage LDS reads before it is reused.
            // Operand registers remain live across the refill and matrix work.
            if constexpr (REGISTER_TILE == 1) {
                asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
                join();
                if (future < end)
                    dma(future, phase);
            }
        } else if (future < end)
            dma(future, (phase + STAGES - 1) % STAGES);
        if constexpr (FRAG_DEPTH > 0) {
            static_assert(BF16 && F == 1 && FN == 1 && !REGISTER_TILE && !READ_FIRST && !FORCE_ACC && ACC_CHAINS == 1);
            static_assert(FRAG_DEPTH == 2 || FRAG_DEPTH == 4 || FRAG_DEPTH == 8);
            static_assert(FRAG_DEPTH <= 4 * KP, "fragment lookahead must fit the K stage");
            v4i af[FRAG_DEPTH], bf[FRAG_DEPTH];
            const int arow = wm * 32 + r, brow = BM + wn * 32 + r;
            auto read_fragment = [&](auto Slot, int h) {
                constexpr int slot = decltype(Slot)::value;
                const int chunk = h * 2 + g;
                af[slot] = read_tile(phase, arow * CHUNKS + (chunk ^ bank_key(arow)));
                bf[slot] = read_tile(phase, brow * CHUNKS + (chunk ^ bank_key(brow)));
            };
            static_for<0, FRAG_DEPTH>([&](auto I) { read_fragment(I, decltype(I)::value); });
            if constexpr (FRAGMENT_SCHEDULE_FENCES)
                __builtin_amdgcn_sched_barrier(0);
            static_for<0, 4 * KP>([&](auto H) {
                constexpr int h = decltype(H)::value, slot = h % FRAG_DEPTH;
                using bf8 = __bf16 __attribute__((ext_vector_type(8)));
                accum[0][0][0] = __builtin_amdgcn_mfma_f32_32x32x16_bf16(
                    __builtin_bit_cast(bf8, bf[slot]), __builtin_bit_cast(bf8, af[slot]), accum[0][0][0], 0, 0, 0);
                if constexpr (h + FRAG_DEPTH < 4 * KP)
                    read_fragment(std::integral_constant<int, slot>{}, h + FRAG_DEPTH);
                if constexpr (FRAGMENT_SCHEDULE_FENCES)
                    __builtin_amdgcn_sched_barrier(0);
            });
        } else {
#pragma unroll
            for (int part = 0; part < KP; ++part) {
#pragma unroll
                for (int half = 0; half < (BF16 ? 4 : 2); ++half) {
                    const int chunk = part * 8 + (BF16 ? half * 2 : half * 4) + g;
#pragma unroll
                    for (int i = 0; i < F; ++i) {
                        int arow = wm * (32 * F) + i * 32 + r;
                        v4i a0;
                        if constexpr (REGISTER_TILE)
                            a0 = prefetched_a[i][(chunk - g) / 2];
                        else if constexpr (READ_FIRST) {
                            if (part == 0 && half == 0)
                                a0 = first_a;
                            else
                                a0 = read_tile(phase, arow * CHUNKS + (chunk ^ bank_key(arow)));
                        } else
                            a0 = read_tile(phase, arow * CHUNKS + (chunk ^ bank_key(arow)));
#pragma unroll
                        for (int j = 0; j < FN; ++j) {
                            int brow = BM + wn * (32 * FN) + j * 32 + r;
                            v4i b0;
                            if constexpr (REGISTER_TILE)
                                b0 = prefetched_b[j][(chunk - g) / 2];
                            else if constexpr (READ_FIRST) {
                                if (part == 0 && half == 0)
                                    b0 = first_b;
                                else
                                    b0 = read_tile(phase, brow * CHUNKS + (chunk ^ bank_key(brow)));
                            } else
                                b0 = read_tile(phase, brow * CHUNKS + (chunk ^ bank_key(brow)));
                            if constexpr (BF16) {
                                using bf8 = __bf16 __attribute__((ext_vector_type(8)));
                                auto &value = accum[(part * 4 + half) % ACC_CHAINS][i][j];
                                if constexpr (FORCE_ACC)
                                    asm volatile("v_mfma_f32_32x32x16_bf16 %0, %1, %2, %0"
                                                 : "+a"(value)
                                                 : "v"(b0), "v"(a0));
                                else
                                    value = __builtin_amdgcn_mfma_f32_32x32x16_bf16(
                                        __builtin_bit_cast(bf8, b0), __builtin_bit_cast(bf8, a0), value, 0, 0, 0);
                            } else {
                                v4i a1, b1;
                                unsigned ac, bc;
                                if constexpr (REGISTER_TILE) {
                                    a1 = prefetched_a[i][(chunk - g) / 2 + 1];
                                    b1 = prefetched_b[j][(chunk - g) / 2 + 1];
                                    ac = prefetched_sa[i][part] >> (8 * (half * 2 + g));
                                    bc = prefetched_sb[j][part] >> (8 * (half * 2 + g));
                                } else {
                                    a1 = read_tile(phase, arow * CHUNKS + ((chunk + 2) ^ bank_key(arow)));
                                    b1 = read_tile(phase, brow * CHUNKS + ((chunk + 2) ^ bank_key(brow)));
                                    ac = read_scale(phase, part * SCALE_ROWS + arow) >> (8 * (half * 2 + g));
                                    if constexpr (SHARED_B_SCALES)
                                        bc = shared_b_codes[phase][part] >> (8 * (half * 2 + g));
                                    else
                                        bc = read_scale(phase, part * SCALE_ROWS + SCALE_A + brow - BM) >>
                                             (8 * (half * 2 + g));
                                }
                                if constexpr (READ_FIRST)
                                    if (part == 0 && half == 0) {
                                        a1 = first_ah;
                                        b1 = first_bh;
                                        ac = first_ac;
                                        bc = first_bc;
                                    }
                                auto &value = accum[(part * 2 + half) % ACC_CHAINS][i][j];
                                value = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(cat8(b0, b1), cat8(a0, a1),
                                                                                        value, 0, 0, 0, bc, 0, ac);
                            }
                        }
                    }
                }
            }
        }
        if constexpr (IGLP >= 0)
            __builtin_amdgcn_iglp_opt(IGLP);
        if constexpr (WAIT_FENCE == 2) {
            static_assert(F == 1 && FN == 1 && ACC_CHAINS == 1);
            // Explicit compiler dependency only: preserve the accumulator before
            // the stage wait without issuing another device synchronization.
            asm volatile("" ::"a"(accum[0][0][0]) : "memory");
        }
        if constexpr (WAIT_FENCE > 0)
            __builtin_amdgcn_sched_barrier(0);
        if (step + 1 < end) {
            if constexpr (STAGES == 3 || REGISTER_TILE) {
                if constexpr (STAGES == 1 || REGISTER_TILE == 2)
                    projection_wait_vm<MANUAL_LDS_DMA, 0>();
                else {
                    if (future < end)
                        projection_wait_ready<EXACT_WAVE_WAIT, MANUAL_LDS_DMA,
                                              (REGISTER_TILE ? STAGES - 1 : 1) * READY_COPIES>(wave);
                    else
                        projection_wait_vm<MANUAL_LDS_DMA, 0>();
                }
                asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
                join();
            } else {
                asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
                join();
            }
        }
    };
    if constexpr (UNROLL == 1) {
        for (int step = begin; step < end; ++step) {
            iteration(step, (step - begin) % STAGES);
            if constexpr (SCHEDULE_FENCE)
                __builtin_amdgcn_sched_barrier(0);
        }
    } else {
        for (int base = begin; base < end; base += UNROLL) {
            static_for<0, UNROLL>([&](auto I) {
                constexpr int i = decltype(I)::value;
                if (base + i < end) {
                    iteration(base + i, i % STAGES);
                    if constexpr (SCHEDULE_FENCE)
                        __builtin_amdgcn_sched_barrier(0);
                }
            });
        }
    }
    static_assert(!PACKED_OUTPUT || COALESCED_OUTPUT);
    static_assert(COALESCED_OUTPUT >= 0 && COALESCED_OUTPUT <= 2);
    static_assert(COALESCED_OUTPUT != 2 || (BF16 && !PACKED_OUTPUT && !WAVE_OUTPUT));
    if constexpr (WAVE_OUTPUT) {
        static_assert(F == 1 && FN == 1 && WAVE_M == 2 && WAVES == 4 && SPLITS == 1);
        acc32_t value = accum[0][0][0];
#pragma unroll
        for (int chain = 1; chain < ACC_CHAINS; ++chain)
            value += accum[chain][0][0];
        const int row = m0 + wm * 32 + lane / 2, col = n0 + wn * 32 + (lane & 1) * 16;
        if constexpr (BF16) {
            float output[16];
#pragma unroll
            for (int q = 0; q < 16; ++q) {
                const int source = lane / 2 + 32 * ((q / 4) % 2), component = (q / 8) * 4 + q % 4;
                const float lo = __shfl(value[component], source), hi = __shfl(value[component + 8], source);
                output[q] = (lane & 1) ? hi : lo;
            }
            if (row < m) {
#pragma unroll
                for (int q = 0; q < 4; ++q)
                    if (col + 4 * q + 3 < n)
                        *(v4f *)((float *)dp + size_t(row) * ldd + col + 4 * q) =
                            v4f{output[4 * q], output[4 * q + 1], output[4 * q + 2], output[4 * q + 3]};
            }
        } else {
            unsigned packed[8], output[8];
#pragma unroll
            for (int q = 0; q < 8; ++q)
                packed[q] = pack_bf16x2(value[2 * q], value[2 * q + 1]);
#pragma unroll
            for (int q = 0; q < 8; ++q) {
                const int source = lane / 2 + 32 * ((q / 2) % 2), component = (q / 4) * 2 + q % 2;
                const unsigned lo = __shfl(packed[component], source), hi = __shfl(packed[component + 4], source);
                output[q] = (lane & 1) ? hi : lo;
            }
            if (row < m) {
#pragma unroll
                for (int q = 0; q < 2; ++q) {
                    const size_t offset = size_t(row) * ldd + col + q * 8;
                    if (col + q * 8 + 7 < n && ldd % 8 == 0 && (uintptr_t(dp) & 15) == 0)
                        *(v4u *)((uint16_t *)dp + offset) =
                            v4u{output[4 * q], output[4 * q + 1], output[4 * q + 2], output[4 * q + 3]};
                    else {
#pragma unroll
                        for (int j = 0; j < 4; ++j)
                            if (col + q * 8 + 2 * j + 1 < n)
                                *(uint32_t *)((uint16_t *)dp + offset + 2 * j) = output[4 * q + j];
                    }
                }
            }
        }
    } else if constexpr (PACKED_OUTPUT && !BF16) {
        static_assert(F == 1 && FN == 1 && WAVE_M == 2 && WAVES == 4 && SPLITS == 1);
        constexpr int PITCH = 72;
        static_assert(sizeof(tiles) >= 64 * PITCH * sizeof(uint16_t));
        asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
        join();
        uint16_t *output_tile = reinterpret_cast<uint16_t *>(&tiles[0][0]);
        acc32_t value = accum[0][0][0];
#pragma unroll
        for (int chain = 1; chain < ACC_CHAINS; ++chain)
            value += accum[chain][0][0];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            int row = wm * 32 + r, col = wn * 32 + q * 8 + g * 4;
            *(v2u *)(output_tile + row * PITCH + col) =
                v2u{pack_bf16x2(value[q * 4], value[q * 4 + 1]), pack_bf16x2(value[q * 4 + 2], value[q * 4 + 3])};
        }
        __syncthreads();
        for (int e = tid * 8; e < 4096; e += THREADS * 8) {
            int row = e / 64, col = e % 64;
            if (m0 + row < m && n0 + col < n) {
                size_t offset = size_t(m0 + row) * ldd + n0 + col;
                if (n0 + col + 7 < n && ldd % 8 == 0 && (uintptr_t(dp) & 15) == 0)
                    *(v4u *)((uint16_t *)dp + offset) = *(v4u *)(output_tile + row * PITCH + col);
                else {
#pragma unroll
                    for (int j = 0; j < 8; ++j)
                        if (n0 + col + j < n)
                            ((uint16_t *)dp)[offset + j] = output_tile[row * PITCH + col + j];
                }
            }
        }
    } else if constexpr (COALESCED_OUTPUT) {
        static_assert(F == 1 && FN == 1 && WAVE_M == 2 && WAVES == 4 && SPLITS == 1);
        constexpr int PITCH = 68;
        static_assert(sizeof(tiles) >= 64 * PITCH * sizeof(float));
        // Retire all prior DMA/LDS reads before aliasing the existing operand tile.
        asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
        join();
        float *output_tile = reinterpret_cast<float *>(&tiles[0][0]);
        acc32_t value = accum[0][0][0];
#pragma unroll
        for (int chain = 1; chain < ACC_CHAINS; ++chain)
            value += accum[chain][0][0];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            int row = wm * 32 + r, col = wn * 32 + q * 8 + g * 4;
            *(v4f *)(output_tile + row * PITCH + col) =
                v4f{value[q * 4], value[q * 4 + 1], value[q * 4 + 2], value[q * 4 + 3]};
        }
        __syncthreads();
        for (int e = tid * 4; e < 4096; e += THREADS * 4) {
            int row = e / 64, col = e % 64;
            if constexpr (COALESCED_OUTPUT == 2) {
                // Opt-in transposed output ABI: D is [n,m], row stride ldd. The
                // original M tail becomes the contiguous output-column tail.
                if (n0 + row < n && m0 + col + 3 < m) {
                    v4f x{output_tile[col * PITCH + row], output_tile[(col + 1) * PITCH + row],
                          output_tile[(col + 2) * PITCH + row], output_tile[(col + 3) * PITCH + row]};
                    *(v4f *)((float *)dp + size_t(n0 + row) * ldd + m0 + col) = x;
                }
            } else if (m0 + row < m && n0 + col + 3 < n) {
                v4f x = *(v4f *)(output_tile + row * PITCH + col);
                size_t offset = size_t(m0 + row) * ldd + n0 + col;
                if constexpr (BF16)
                    *(v4f *)((float *)dp + offset) = x;
                else
                    *(v2u *)((uint16_t *)dp + offset) = v2u{pack_bf16x2(x[0], x[1]), pack_bf16x2(x[2], x[3])};
            }
        }
    } else {
#pragma unroll
        for (int i = 0; i < F; ++i) {
            int row = m0 + wm * 32 * F + i * 32 + r;
            if (row >= m)
                continue;
#pragma unroll
            for (int j = 0; j < FN; ++j) {
                acc32_t value = accum[0][i][j];
#pragma unroll
                for (int chain = 1; chain < ACC_CHAINS; ++chain)
                    value += accum[chain][i][j];
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    int col = n0 + wn * 32 * FN + j * 32 + q * 8 + g * 4;
                    if (col + 3 < n) {
                        size_t off = size_t(row) * ldd + col;
                        if constexpr (BF16 || SPLITS > 1)
                            *(v4f *)((float *)dp + off + (SPLITS > 1 ? size_t(blockIdx.z) * m * ldd : 0)) =
                                v4f{value[q * 4], value[q * 4 + 1], value[q * 4 + 2], value[q * 4 + 3]};
                        else
                            *(v2u *)((uint16_t *)dp + off) = v2u{pack_bf16x2(value[q * 4], value[q * 4 + 1]),
                                                                 pack_bf16x2(value[q * 4 + 2], value[q * 4 + 3])};
                    }
                }
            }
        }
    }
#endif
}
} // namespace deep_gemm::cdna4::projection
