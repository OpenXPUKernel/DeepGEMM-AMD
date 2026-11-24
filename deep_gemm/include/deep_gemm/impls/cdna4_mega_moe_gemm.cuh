#pragma once
#include <deep_gemm/impls/cdna4/device_common.h>
#include <deep_gemm/impls/cdna4/gemm_args.h>
#if defined(CDNA4_SHORT_MFMA16) && !defined(CDNA4_UP_WEIGHT_AHEAD) && !defined(CDNA4_MOE_PACKED_EPILOGUE)
#include <deep_gemm/impls/cdna4/resident_mfma16.cuh>
#define CDNA4_USE_SHORT_MFMA16 1
#endif
using v16f = float __attribute__((ext_vector_type(16)));
// Scaled MFMA tile for routed FP4 and shared FP8 experts. Paired routed
// columns reuse activation fragments across two weight panels.
template <int K, int N, bool First, bool PackedB, bool WeightPanels, int TileColumns, int PairStride, int ReadyGroups, int RowCap = 128>
__device__ __forceinline__ void pipeline_native_tile(deep_gemm::cdna4::GemmArgs p, int expert, int m0, int end, int n0,
    const uint32_t *local_rows, uint32_t input_bytes, const int *completion, const int32_t *original_sf = nullptr,
    const int *row_weights = nullptr, bool up_ahead = false) {
    constexpr int KT = K / 128, S = PackedB ? 3 : 2;
#ifdef CDNA4_UP_WEIGHT_CACHE
    constexpr int WeightCache = PackedB ? (First ? CDNA4_UP_WEIGHT_CACHE : CDNA4_DOWN_WEIGHT_CACHE) : 0;
#else
    constexpr int WeightCache = 0;
#endif

    static_assert(S >= 2 && S <= 3);
    static_assert(!WeightPanels || PackedB, "only routed packed-FP4 weights use panel storage");

    static_assert(TileColumns == 256 || TileColumns == 512);

    static_assert(!PairStride || TileColumns == 512);

    constexpr int BM = 128, BN = TileColumns;

    // A paired tile uses all four waves for columns. A proven half-full row
    // tile can issue two row fragments per wave with the normal MF=2 pipeline.
#ifdef CDNA4_HALF_ROW_PAIR
    constexpr int MappedRows = RowCap == 64 && First && PairStride != 0 ? 64 : BM;
#else
    constexpr int MappedRows = BM;
#endif
    constexpr int WM_ROWS = MappedRows / (TileColumns == 512 ? 1 : 2), MF = WM_ROWS / 32;
    static_assert(MF > 0, "every native matrix wave requires a complete row fragment");

    // On short single panels, each wave computes the same 64 rows and
    // distinct 64-column strips. Keep the 128-row external buffer layout.
#ifdef CDNA4_SHORT_COLUMN_WAVES
#ifdef CDNA4_MOE_PACKED_EPILOGUE
    constexpr bool packed_output = First && (CDNA4_MOE_PACKED_EPILOGUE & 1);
#else
    constexpr bool packed_output = false;
#endif
    constexpr bool short_columns = RowCap == 64 && TileColumns == 256 && PackedB && WeightPanels && !packed_output &&
                                   (CDNA4_SHORT_COLUMN_WAVES & (First ? 1 : 2));
#else
    constexpr bool short_columns = false;
#endif
    constexpr int NF = short_columns ? 2 : 4;
    constexpr int AL = BM / (32), BL = (BN / (PackedB ? 64 : 32));
    constexpr int BROW = PackedB ? 64 : 128, BSTRIDE = PackedB ? K / 2 : K;
    // Each wave owns its scale DMA destinations, including unused dummy rows.
    // Widening the workgroup must not let those dummy writes alias B scales.
    constexpr int ScaleRows = 256;
    constexpr int DMAS = AL + BL + 1 + (((BN + ScaleRows - 1) / ScaleRows));
    constexpr int AO = 0, BO = BM * 128, SAO = BO + (BN * BROW), SBO = SAO + ScaleRows * 4,
                  STEP = SBO + (((BN > ScaleRows ? BN : ScaleRows) * 4));
    constexpr int EPI = S * STEP, PITCH = 2 * BN + 16;

    static_assert(K % 128 == 0 && K >= 128 * S && N % (PairStride ? 256 : BN) == 0);
    static_assert((EPI > BM * PITCH ? EPI : BM * PITCH) <= 163840);
    extern __shared__ __attribute__((aligned(1024))) uint8_t sm[];
    const int tid = threadIdx.x, lane = tid % 64, wave = __builtin_amdgcn_readfirstlane(tid / 64);
    const int wm = short_columns ? 0 : ((TileColumns == 512) ? wave / 4 : wave / 2);
    const int wn = short_columns ? wave : ((TileColumns == 512) ? wave % 4 : wave % 2), r = lane % 32, g = lane / 32;
    const int ar = 8 * wave + lane / 8, br = PackedB ? 16 * wave + lane / 4 : 8 * wave + lane / 8;
    const int bkey_dma = PackedB ? ((br >> 2) & 3) : ((br >> 1) & 7);
    const int brow = (r);
    const int av = ar * K + ((lane % 8) ^ ((ar >> 1) & 7)) * 16;
#ifdef CDNA4_SHARED_FP8_PANELS
    constexpr bool dense_panel = !PackedB && (CDNA4_SHARED_FP8_PANELS & (First ? 1 : 2)) != 0;
#else
    constexpr bool dense_panel = false;
#endif
    const int bv = br * (WeightPanels || dense_panel ? BROW : BSTRIDE) + ((lane % (PackedB ? 4 : 8)) ^ bkey_dma) * 16;
    const int sv = (wave * 64 + lane) * 4, sa_v = sv;
    constexpr int NPerWave = short_columns ? 64 : ((TileColumns == 512) ? 128 : (BN / 2)), PanelStep = PairStride ? PairStride : 1;
    constexpr int SpanN = PairStride ? (PairStride + 1) * 256 : BN;
    const int aoff = AO + (wm * WM_ROWS + r) * 128, boff = BO + (wn * NPerWave + brow) * BROW;
    const int akey = (r >> 1) & 7, bkey = PackedB ? ((r >> 2) & 3) : ((r >> 1) & 7);
    const int aso = SAO + (wm * WM_ROWS + r) * 4, bso = SBO + (wn * NPerWave + brow) * 4;

    constexpr int tile_phase = 0;
    uint64_t ready_cache = 0;
#ifdef CDNA4_MOE_LOCAL_INPUT
    uint32_t row_bytes[AL];
    static_for<0, AL>([&](auto J) {
        constexpr int j = decltype(J)::value;
        row_bytes[j] = local_rows ? local_rows[j * 32 + ar] + (av - ar * K) : av + j * 32 * K;
    });

    auto ra = make_rsrc(local_rows ? p.A : p.A + int64_t(m0) * K, local_rows ? input_bytes : min(end - m0, BM) * K);
#else
    auto ra = make_rsrc(p.A + int64_t(m0) * K, min(end - m0, BM) * K);
#endif
    const int64_t bbase =
        WeightPanels ? int64_t(n0 / 256) * 256 * BSTRIDE + int64_t(n0 % 256) * BROW : int64_t(n0) * BSTRIDE;
    auto rb = make_rsrc(p.B + int64_t(expert) * p.b_group_stride + bbase,
                        (PairStride ? SpanN : (WeightPanels ? (BN > 256 ? BN : 256) : BN)) * BSTRIDE);
    uint32_t sa_offset = sa_v, sa_step = p.lda_sf * 4u;
    const int32_t *scale_base = p.sfa + m0;
    uint32_t scale_bytes = uint32_t((uint64_t(K / 128 - 1) * p.lda_sf + min(end - m0, BM)) * 4);
#if defined(CDNA4_MOE_EAGER_LOCAL) || defined(CDNA4_SHARED_DURING_ROUTE)
    if constexpr (First) {
        if (original_sf) {
            scale_base = original_sf;
            scale_bytes = input_bytes / 32;
            if constexpr (PackedB)
                sa_offset = tid < BM ? local_rows[tid] / 32 : scale_bytes;
            else
                sa_offset = tid < BM && m0 + tid < end ? uint32_t(m0 + tid) * (K / 32) : scale_bytes;
            sa_step = 4;
        }
    }
#endif
    auto rsa = make_rsrc(scale_base, scale_bytes);
    auto rsb = make_rsrc(p.sfb + int64_t(expert) * p.sfb_group_stride + n0,
                         uint32_t((uint64_t(K / 128 - 1) * p.ldb_sf + SpanN) * 4));
    auto dma_one = [&](auto J, int stage, int k) {
        constexpr int j = decltype(J)::value;
        uint8_t *s = sm + stage * STEP;
        if constexpr (j < AL) {
#ifdef CDNA4_MOE_LOCAL_INPUT
            __builtin_amdgcn_raw_ptr_buffer_load_lds(ra, (lds_ptr_t)(s + AO + (j * 4 + wave) * 1024), 16, row_bytes[j], k * 128, 0, (16));
#else
            __builtin_amdgcn_raw_ptr_buffer_load_lds(ra, (lds_ptr_t)(s + AO + (j * 4 + wave) * 1024), 16, av, j * 32 * K + k * 128, 0, 16);
#endif
        } else if constexpr (j < AL + BL)
            __builtin_amdgcn_raw_ptr_buffer_load_lds(rb, (lds_ptr_t)(s + BO + ((j - AL) * 4 + wave) * 1024), 16, bv,
                dense_panel    ? (j - AL) * 32 * BSTRIDE + k * 32 * BROW
                : WeightPanels ? ((((j - AL) * (64)) / 256) * PanelStep * 256 * BSTRIDE + (((j - AL) * (64)) % 256) * BROW + k * 256 * BROW)
                               : ((((j - AL) * (PackedB ? 64 : 32)) / 256) * PanelStep * 256 * BSTRIDE +
                                     (((j - AL) * (PackedB ? 64 : 32)) % 256) * BSTRIDE + k * BROW),
                0, WeightCache);
        else if constexpr (j == AL + BL)
            __builtin_amdgcn_raw_ptr_buffer_load_lds(rsa, (lds_ptr_t)(s + SAO + wave * 256), 4, sa_offset, uint32_t(k) * sa_step, 0, (16));
        else
            __builtin_amdgcn_raw_ptr_buffer_load_lds(
                rsb, (lds_ptr_t)(s + SBO + (wave + (j - AL - BL - 1) * 4) * 256), 4, sv,
                uint32_t(k) * uint32_t(p.ldb_sf) * 4u + (j - AL - BL - 1) * uint32_t(ScaleRows) * PanelStep * 4u +
                    uint32_t(wave / 4) * (PanelStep - 1) * 256u * 4u,
                0, 0);
    };
    auto dma = [&](int stage, int k) { static_for<0, DMAS>([&](auto J) { dma_one(J, stage, k); }); };
#ifdef CDNA4_USE_SHORT_MFMA16
    if constexpr(First && PackedB && RowCap>0 && RowCap<=(CDNA4_SHORT_MFMA16==1 ? 32 : 64)){
        // Every wave now owns useful columns, including the 256-column tile
        // where the full-row geometry assigns two waves to the padded half.
        constexpr bool split_rows=CDNA4_SHORT_MFMA16==3 && RowCap==64;
        constexpr int WaveRows=split_rows ? 32 : RowCap;
        constexpr int WaveCols=split_rows ? BN/2 : BN/4;
        const int row_group=split_rows ? wave/2 : 0;
        const int column_group=split_rows ? wave%2 : wave;
        constexpr int half_m=WaveRows/32,half_n=WaveCols/32,quadrant=half_m*half_n;
        auto ready=[](int){};
        if constexpr(CDNA4_SHORT_MFMA16==4 && quadrant>=half_m+half_n && 3*quadrant>=DMAS)
            quadrant_resident_mfma16<KT,S,DMAS,STEP,BO,SAO,SBO,WaveRows,WaveCols,PITCH>(
                    sm,row_group,column_group,dma_one,ready);
        else
            resident_mfma16<KT,S,DMAS,STEP,BO,SAO,SBO,WaveRows,WaveCols,PITCH>(
                    sm,row_group,column_group,dma,ready);
        return;
    }
#endif
    v16f acc[MF][NF];
    v8i af[2][MF], bf[2][NF];
    uint32_t ac[(2)][(MF)], bc[(2)][(NF)];

    // The wave's 32-row output fragment is independent of all other
    // fragments. Only omit arithmetic when every row of the fragment is
    // padded. All waves still issue their original VMEM traffic and joins.
    auto active_fragment = [&](int fragment) {
        if constexpr (RowCap == 128)
            return true;
        const int start = (TileColumns == 512 ? 0 : wm * WM_ROWS) + 32 * fragment;
        if constexpr (RowCap == 0)
            return start < end - m0;
        return start < RowCap;
    };
    auto read_frag = [&](auto Bank, auto F, int half, int stage, int kt = 0) {
        constexpr int bank = decltype(Bank)::value, f = decltype(F)::value;
        const uint8_t *st = sm + stage * STEP;
        if constexpr (f >= 1 && f <= MF) {
            constexpr int i = f - 1;
#if defined(CDNA4_ACTIVE_SLABS) && CDNA4_ACTIVE_SLABS >= 2
            if constexpr (RowCap != 128) {
                if (!active_fragment(i)) {
                    af[bank][i] = v8i{};
                    ac[bank][i] = 0;
                    return;
                }
            }
#endif
            const v4i lo = *reinterpret_cast<const v4i *>(st + aoff + i * 32 * 128 + ((half * 4 + g) ^ akey) * 16);
            const v4i hi = *reinterpret_cast<const v4i *>(st + aoff + i * 32 * 128 + ((half * 4 + g + 2) ^ akey) * 16);
            af[bank][i] = v8i{lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]};
            {
                const uint32_t code = st[aso + i * 32 * 4 + half * 2 + g];
                ac[bank][i] = code;
            }
        } else {
            constexpr int i = f == 0 ? 0 : f - MF;
            if constexpr (PackedB) {
                const v4i b = *reinterpret_cast<const v4i *>(st + boff + i * 32 * BROW + ((half * 2 + g) ^ bkey) * 16);
                bf[bank][i] = v8i{b[0], b[1], b[2], b[3], 0, 0, 0, 0};
            } else {
                const v4i lo = *reinterpret_cast<const v4i *>(st + boff + i * 32 * BROW + ((half * 4 + g) ^ bkey) * 16);
                const v4i hi =
                    *reinterpret_cast<const v4i *>(st + boff + i * 32 * BROW + ((half * 4 + g + 2) ^ bkey) * 16);
                bf[bank][i] = v8i{lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]};
            }
            {
                const uint32_t code = st[bso + i * 32 * 4 + half * 2 + g];
                bc[bank][i] = code;
            }
        }
    };
#if defined(CDNA4_PROGRESSIVE_K) && CDNA4_PROGRESSIVE_K <= 2
    unsigned ready_panels = 0;
    auto wait_panels = [&](unsigned needed) {
        if constexpr (!First && PackedB && ReadyGroups > 0) {
            if (completion && (ready_panels & needed) != needed) {
                __shared__ unsigned observed;
                if (tid == 0) {
                    unsigned mask;
                    do {
                        mask =
                            __hip_atomic_load(reinterpret_cast<const unsigned *>(completion), __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
                        if ((mask & needed) != needed)
                            __builtin_amdgcn_s_sleep(4);
                    } while ((mask & needed) != needed);
                    observed = mask;
                }
                __syncthreads();
                ready_panels = observed;
                asm volatile("" ::: "memory");
                __syncthreads();
            }
        }
    };
#endif
    auto step = [&](int k, auto first, auto last, auto ExplicitStage, auto NewerStages) {
        constexpr int newer = decltype(NewerStages)::value;
#if defined(CDNA4_MOE_EARLY_K) && CDNA4_MOE_EARLY_K == 2 && !defined(CDNA4_PROGRESSIVE_K)
        constexpr int prefix = 2 * ReadyGroups - (2 * K / 256);
        if constexpr (!First && PackedB && ReadyGroups > 0 && prefix >= S && 2 * K / 256 > ReadyGroups) {
            // The prologue only prefetches [0,S). Drain the row's remaining
            // writers before this iteration can issue the first later K DMA.
            if (completion && k == prefix - S) {
                if (threadIdx.x == 0)
                    while (__hip_atomic_load(completion, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < ReadyGroups)
                        __builtin_amdgcn_s_sleep(4);
                __syncthreads();
                asm volatile("" ::: "memory");
                __syncthreads();
            }
        }
#endif
        {
            const int stage = int(ExplicitStage);
            const int next = stage + 1 == S ? 0 : stage + 1, loadk = k + S < KT ? k + S : k + S - KT;
            {

                static_for<0, 2 * MF * NF>([&](auto T) {
                    constexpr int t = decltype(T)::value, half = t / (MF * NF);
                    constexpr int i = ((t % (MF * NF)) / NF);
                    constexpr int j = (t % NF);
                    constexpr int code_bank = (half);
                    const int bcode = (bc[code_bank][j]);
                    const int acode = (ac[code_bank][i]);
                    if (active_fragment(i)) {
                        acc[i][j] = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(bf[half][j], af[half][i],
                            decltype(first)::value && half == 0 ? v16f{} : acc[i][j], PackedB ? 4 : 0, 0, (0), bcode, (0), acode);
                    } else if constexpr (decltype(first)::value && half == 0) {
                        acc[i][j] = v16f{};
                    }

                    constexpr int Reads = MF + NF, Half = MF * NF;
                    if constexpr ((t >= Half - Reads) && t < Half)
                        read_frag(std::integral_constant<int, 1>{}, std::integral_constant<int, t - (Half - Reads)>{},
                                  1, stage, k);
                    if constexpr ((!decltype(last)::value)) {
                        constexpr int FirstDma = Half + NF, Span = 2 * Half - FirstDma, NextRead = 2 * Half - Reads;
                        if constexpr (newer < 0) {
                            if constexpr ((!PackedB) && MF >= 2) {
                                if constexpr (t == Half) {
                                    // All current-stage operand fragments have been issued by Half-1.
                                    // Drain those LDS reads before recycling their storage. Issue the full
                                    // replacement before waiting: keeping (S-1) DMA groups pending then
                                    // proves that the older next-stage operand group has completed.
                                    asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
                                    dma(stage, loadk);
                                }
                            } else {
                                if constexpr (t == FirstDma - 1)
                                    asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
#ifdef CDNA4_PROGRESSIVE_K
                                if constexpr (!First && PackedB && ReadyGroups > 0 && t == FirstDma) {
#if CDNA4_PROGRESSIVE_K <= 2
                                    if (completion)
                                        wait_panels(1u << loadk);
#else
                                    constexpr int prefix = 2 * ReadyGroups - 2 * K / 256;
                                    if (completion && k == prefix - S) {
                                        if (tid == 0)
                                            while (__hip_atomic_load(completion, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < ReadyGroups)
                                                __builtin_amdgcn_s_sleep(4);
                                        __syncthreads();
                                        asm volatile("" ::: "memory");
                                        __syncthreads();
                                    }
#endif
                                }
#endif
                                static_for<0, DMAS>([&](auto J) {
                                    constexpr int j = decltype(J)::value;
                                    if constexpr (t == FirstDma + j * Span / DMAS)
                                        dma_one(J, stage, loadk);
                                });
                            }
                        }
                        if constexpr (t == NextRead) {
#if defined(CDNA4_UP_WEIGHT_AHEAD) && CDNA4_UP_WEIGHT_AHEAD == 2
                            if (First && PackedB && up_ahead && k < S - 1)
                                __builtin_amdgcn_s_waitcnt(waitcnt_vm(0));
#endif
                            // The short tile reaches this rendezvous before issuing k+2 loads:
                            // drain the pending k+1 stage. The original tile keeps its exact wait.
                            // In the partial register policy, the bound counts only newer A DMA.
                            // Additional B reads are deliberately excluded: this can strengthen the
                            // wait, not permit more newer requests than the proven A-only bound.
                            // Next-step B reads start only after this rendezvous. Keep the fully
                            // drained register policy as a separate numerical/performance control.
                            if constexpr (newer >= 0)
                                __builtin_amdgcn_s_waitcnt(waitcnt_vm(newer * DMAS));
                            else
                                __builtin_amdgcn_s_waitcnt(waitcnt_vm(
                                    (((!PackedB) && MF >= 2
                                            ? (S - 1) * DMAS
                                            : (S - 2) * DMAS +
                                                  (NextRead >= FirstDma ? ((NextRead - FirstDma + 1) * DMAS + Span - 1) / Span : 0)))));
                            asm volatile("s_barrier" ::: "memory");
                        }
                        if constexpr ((t >= NextRead) && !decltype(last)::value)
                            read_frag(std::integral_constant<int, 0>{}, std::integral_constant<int, t - NextRead>{}, 0,
                                      next, k + 1);
                    }

                    __builtin_amdgcn_sched_barrier(0);
                });
            }
        }
    };

    {
#ifdef CDNA4_UP_WEIGHT_AHEAD
        if (First && PackedB && up_ahead) {
            static_for<0, S>([&](auto Stage) {
                constexpr int k = decltype(Stage)::value;
                if constexpr (k == 0 || CDNA4_UP_WEIGHT_AHEAD == 2) {
                    static_for<0, AL>([&](auto J) { dma_one(J, k, k); });
                    dma_one(std::integral_constant<int, AL + BL>{}, k, k);
                } else
                    dma(k, k);
            });
            if constexpr (CDNA4_UP_WEIGHT_AHEAD == 2)
                __builtin_amdgcn_s_waitcnt(waitcnt_vm(0));
        } else
#endif
#if defined(CDNA4_PROGRESSIVE_K) && CDNA4_PROGRESSIVE_K <= 2
            if (!First && PackedB && completion) {
#if CDNA4_PROGRESSIVE_K == 1
            wait_panels((1u << S) - 1u);
#endif
            static_for<0, S>([&](auto Stage) {
                constexpr int k = decltype(Stage)::value;
#if CDNA4_PROGRESSIVE_K == 2
                wait_panels(1u << k);
#endif
                dma(k, k);
            });
        } else
#endif
            static_for<0, S>([&](auto I) { dma(decltype(I)::value, decltype(I)::value); });
        __builtin_amdgcn_s_waitcnt(waitcnt_vm(((S - 1) * DMAS)));
        asm volatile("s_barrier" ::: "memory");
    }
    {

        static_for<0, MF + NF>([&](auto F) { read_frag(std::integral_constant<int, 0>{}, F, 0, tile_phase); });

        using Refill = std::integral_constant<int, -1>;
#ifdef CDNA4_MOE_DRAIN_TAIL
        constexpr bool peel = KT > S;
#else
        constexpr bool peel = false;
#endif
        constexpr int limit = peel ? KT - S : KT - 1;
#ifdef CDNA4_MOE_ROLL_STAGES
        // Retain the same fragment lifetimes, per-stage waits, ordered MFMA
        // sequence and specialized tail. The steady-state code rotates one
        // LDS base rather than duplicating its full instruction body S times.
        constexpr bool compact = (CDNA4_MOE_ROLL_MASK & (PackedB ? (First ? 1 : 2) : 4)) != 0;
        if constexpr (compact) {
            constexpr bool common_first = CDNA4_MOE_ROLL_STAGES == 2;
            if constexpr (common_first) {
                static_for<0, MF>(
                    [&](auto Row) { static_for<0, NF>([&](auto Col) { acc[decltype(Row)::value][decltype(Col)::value] = v16f{}; }); });
            } else
                step(0, std::true_type{}, std::false_type{}, 0, Refill{});
            int stage = common_first ? 0 : 1;
#pragma unroll 1
            for (int k = common_first ? 0 : 1; k < limit; ++k) {
                step(k, std::false_type{}, std::false_type{}, stage, Refill{});
                stage = stage + 1 == S ? 0 : stage + 1;
            }
        } else
#endif
        {
            step(0, std::true_type{}, std::false_type{}, std::integral_constant<int, 0>{}, Refill{});
#pragma unroll 1
            for (int base = 1; base < limit; base += S)
                static_for<0, S>([&](auto J) {
                    constexpr int j = decltype(J)::value;
                    if (base + j < limit)
                        step(base + j, std::false_type{}, std::false_type{}, std::integral_constant<int, (1 + j) % S>{}, Refill{});
                });
        }
        if constexpr (peel) {
            // Each future operand stage is already in flight. No replacement
            // should reread the beginning of K; count only stages newer than k+1.
            static_for<0, S - 1>([&](auto J) {
                constexpr int k = KT - S + decltype(J)::value;
                step(k, std::false_type{}, std::false_type{}, std::integral_constant<int, k % S>{},
                    std::integral_constant<int, KT - k - 2>{});
            });
        }
        step(KT - 1, std::false_type{}, std::true_type{}, std::integral_constant<int, (KT - 1) % S>{}, Refill{});
    }
    asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)\n\ts_barrier" ::: "memory");

#ifdef CDNA4_MOE_PACKED_EPILOGUE
    if constexpr (First && PackedB && (CDNA4_MOE_PACKED_EPILOGUE & (PairStride ? 2 : 1))) {
        constexpr int PackedPitch = BN / 2 + 16, WordsPerRow = BN / 256;
        auto *scales = reinterpret_cast<uint16_t *>(sm + BM * PackedPitch);
        static_for<0, MF>([&](auto I) {
            constexpr int i = decltype(I)::value;
            const int relative = wm * WM_ROWS + i * 32 + r, row = m0 + relative;
            float clamp, unused;
            unpack_bf16x2(pack_bf16x2(p.moe_clamp, 0.f), clamp, unused);
            const float weight = row < end ? __int_as_float(row_weights[relative]) : 0.f;
            uint32_t scale_half = 0;
            static_for<0, NF / 2>([&](auto P) {
                constexpr int pair = decltype(P)::value;
                float values[2][2][4];
                float amax = 0.f;
                static_for<0, 2>([&](auto J) {
                    constexpr int jj = decltype(J)::value, j = pair * 2 + jj;
                    static_for<0, 2>([&](auto Q) {
                        constexpr int q = decltype(Q)::value;
                        static_for<0, 2>([&](auto Z) {
                            constexpr int z = decltype(Z)::value;
                            float gate[2], up[2];
                            unpack_bf16x2(pack_bf16x2(acc[i][j][q * 8 + z * 2], acc[i][j][q * 8 + z * 2 + 1]), gate[0], gate[1]);
                            unpack_bf16x2(pack_bf16x2(acc[i][j][q * 8 + 4 + z * 2], acc[i][j][q * 8 + 5 + z * 2]), up[0], up[1]);
                            static_for<0, 2>([&](auto T) {
                                constexpr int t = decltype(T)::value;
                                const float x = fminf(gate[t], clamp), y = fminf(fmaxf(up[t], -clamp), clamp);
                                const float denominator = 1.f + (p.moe_fast_math ? __expf(-x) : expf(-x));
                                const float silu = p.moe_fast_math ? x * __builtin_amdgcn_rcpf(denominator) : x / denominator;
                                const float value = silu * y * weight;
                                values[jj][q][z * 2 + t] = value;
                                amax = fmaxf(amax, fabsf(value));
                            });
                        });
                    });
                });
                amax = fmaxf(amax, __shfl_xor(amax, 32));
                const uint32_t rounded = (__float_as_uint(amax) + 0x7fffffu - (0x60u << 16)) >> 23;
                const uint32_t exponent = max(rounded, 113u) - 8u;
                const float inverse = __uint_as_float((254u - exponent) << 23);
                scale_half |= exponent << (pair * 8);
                static_for<0, 2>([&](auto J) {
                    constexpr int jj = decltype(J)::value, j = pair * 2 + jj;
                    static_for<0, 2>([&](auto Q) {
                        constexpr int q = decltype(Q)::value;
                        const int lo = __builtin_amdgcn_cvt_pk_fp8_f32(values[jj][q][0] * inverse, values[jj][q][1] * inverse, 0, false);
                        const uint32_t packed =
                            __builtin_amdgcn_cvt_pk_fp8_f32(values[jj][q][2] * inverse, values[jj][q][3] * inverse, lo, true);
                        const int column = wn * 64 + j * 16 + q * 8 + g * 4;
                        *reinterpret_cast<uint32_t *>(sm + relative * PackedPitch + column) = packed;
                    });
                });
            });

            if (g == 0)
                scales[relative * (BN / 128) + wn] = uint16_t(scale_half);
        });
        asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
        // Repartition the compact plane so peer-visible vector stores write
        // complete contiguous groups. No FP32 partial or BF16 tile escapes.
        for (int v = tid; v < BM * (BN / 32); v += 256) {
            const int relative = v / (BN / 32), col = v % (BN / 32) * 16, row = m0 + relative;
            if (row < end) {
                const int dest = n0 / 2 + (PairStride ? (col / 128) * PairStride * 128 + col % 128 : col);
                auto *ptr = p.moe_acts + int64_t(row) * (N / 2) + dest;
                const v4u value = *reinterpret_cast<const v4u *>(sm + relative * PackedPitch + col);
                asm volatile("global_store_dwordx4 %0, %1, off sc1\n\ts_nop 0" ::"v"(reinterpret_cast<uint64_t>(ptr)), "v"(value)
                    : "memory");
            }
        }
        for (int v = tid; v < BM * WordsPerRow; v += 256) {
            const int relative = v % BM, half = v / BM, row = m0 + relative;
            if (row < end) {
                const uint32_t value = reinterpret_cast<const uint32_t *>(scales)[relative * WordsPerRow + half];
                auto *dst = p.moe_sf + int64_t(n0 / 256 + (PairStride ? PairStride : 1) * half) * p.moe_sf_pitch + row;
                __hip_atomic_store(dst, int32_t(value), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            }
        }
        return;
    }
#endif
    {
        {
            // Alias the fully drained operand LDS with the complete BF16 output tile.
            // Accumulators die before the caller runs SwiGLU/quantization.
            static_for<0, MF>([&](auto I) {
                constexpr int i = decltype(I)::value;
                static_for<0, NF>([&](auto J) {
                    constexpr int j = decltype(J)::value;
                    static_for<0, 4>([&](auto Q) {
                        constexpr int q = decltype(Q)::value;
                        const int dst =
                            (wm * WM_ROWS + i * 32 + r) * PITCH + (wn * NPerWave + j * 32 + g * 4 + q * 8) * 2;
                        *reinterpret_cast<v2u *>(sm + dst) =
                            v2u{pack_bf16x2(acc[i][j][q * 4], acc[i][j][q * 4 + 1]),
                                pack_bf16x2(acc[i][j][q * 4 + 2], acc[i][j][q * 4 + 3])};
                        __builtin_amdgcn_sched_barrier(0);
                    });
                });
            });
            asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
        }
        if constexpr (!First && !PackedB) {
            uint32_t etid = tid;
            asm volatile("v_mov_b32 %0, %0" : "+v"(etid)::"memory");
            const int row0 = etid / (BN / 8), col = (etid % (BN / 8)) * 8;
            auto out = make_rsrc(reinterpret_cast<uint16_t *>(p.D) + int64_t(m0) * N + n0,
                                 uint32_t((min(end - m0, BM) - 1) * N * 2 + SpanN * 2));
            const uint32_t voff = row0 * N * 2 + (PairStride ? (col % 256 + (col / 256) * PairStride * 256) : col) * 2;
            constexpr int OutputRows = 256 / (BN / 8);
            static_for<0, BM / OutputRows>([&](auto Q) {
                constexpr int q = decltype(Q)::value;
                const v4u value = *reinterpret_cast<const v4u *>(sm + (row0 + q * OutputRows) * PITCH + col * 2);
                buffer_store16<!First ? 17 : 0>(value, out, voff, q * OutputRows * N * 2);
            });
            asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)\n\ts_barrier" ::: "memory");
        }
    }
}
