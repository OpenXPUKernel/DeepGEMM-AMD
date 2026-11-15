#pragma once
#include <deep_gemm/impls/cdna4/device_common.h>
#include <deep_gemm/impls/cdna4/gemm_args.h>
using v16f = float __attribute__((ext_vector_type(16)));
// Scaled MFMA tile for routed FP4 and shared FP8 experts. Paired routed
// columns reuse activation fragments across two weight panels.
template <int K, int N, bool First, bool PackedB, bool WeightPanels, int TileColumns, int PairStride>
__device__ __forceinline__ void pipeline_native_tile(deep_gemm::cdna4::GemmArgs p, int expert, int m0, int end,
                                                     int n0) {
    constexpr int KT = K / 128, S = PackedB ? 3 : 2;

    static_assert(S >= 2 && S <= 3);
    static_assert(!WeightPanels || PackedB, "only routed packed-FP4 weights use panel storage");

    static_assert(TileColumns == 256 || TileColumns == 512);

    static_assert(!PairStride || TileColumns == 512);

    constexpr int BM = 128, BN = TileColumns;

    constexpr int WM_ROWS = BM / (TileColumns == 512 ? 1 : 2), MF = WM_ROWS / 32;
    static_assert(MF > 0, "every native matrix wave requires a complete row fragment");

    constexpr int NF = 4;
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
    const int wm = ((TileColumns == 512) ? wave / 4 : wave / 2);
    const int wn = ((TileColumns == 512) ? wave % 4 : wave % 2), r = lane % 32, g = lane / 32;
    const int ar = 8 * wave + lane / 8, br = PackedB ? 16 * wave + lane / 4 : 8 * wave + lane / 8;
    const int bkey_dma = PackedB ? ((br >> 2) & 3) : ((br >> 1) & 7);
    const int brow = (r);
    const int av = ar * K + ((lane % 8) ^ ((ar >> 1) & 7)) * 16;
    const int bv = br * (WeightPanels ? BROW : BSTRIDE) + ((lane % (PackedB ? 4 : 8)) ^ bkey_dma) * 16;
    const int sv = (wave * 64 + lane) * 4, sa_v = sv;
    constexpr int NPerWave = ((TileColumns == 512) ? 128 : (BN / 2)), PanelStep = PairStride ? PairStride : 1;
    constexpr int SpanN = PairStride ? (PairStride + 1) * 256 : BN;
    const int aoff = AO + (wm * WM_ROWS + r) * 128, boff = BO + (wn * NPerWave + brow) * BROW;
    const int akey = (r >> 1) & 7, bkey = PackedB ? ((r >> 2) & 3) : ((r >> 1) & 7);
    const int aso = SAO + (wm * WM_ROWS + r) * 4, bso = SBO + (wn * NPerWave + brow) * 4;

    constexpr int tile_phase = 0;
    uint64_t ready_cache = 0;
    const uint32_t *row_offsets = nullptr;

    auto ra = make_rsrc(p.A + ((int64_t(m0) * K)), (min(end - m0, BM) * K));
    const int64_t bbase =
        WeightPanels ? int64_t(n0 / 256) * 256 * BSTRIDE + int64_t(n0 % 256) * BROW : int64_t(n0) * BSTRIDE;
    auto rb = make_rsrc(p.B + int64_t(expert) * p.b_group_stride + bbase,
                        (PairStride ? SpanN : (WeightPanels ? (BN > 256 ? BN : 256) : BN)) * BSTRIDE);
    auto rsa = make_rsrc(p.sfa + m0, uint32_t((uint64_t(K / 128 - 1) * p.lda_sf + min(end - m0, BM)) * 4));
    auto rsb = make_rsrc(p.sfb + int64_t(expert) * p.sfb_group_stride + n0,
                         uint32_t((uint64_t(K / 128 - 1) * p.ldb_sf + SpanN) * 4));
    auto dma_one = [&](auto J, int stage, int k) {
        constexpr int j = decltype(J)::value;
        uint8_t *s = sm + stage * STEP;
        if constexpr (j < AL) {
            __builtin_amdgcn_raw_ptr_buffer_load_lds(ra, (lds_ptr_t)(s + AO + (j * 4 + wave) * 1024), 16, av,
                                                     j * (32) * K + k * 128, 0, (16));
        } else if constexpr (j < AL + BL)
            __builtin_amdgcn_raw_ptr_buffer_load_lds(
                rb, (lds_ptr_t)(s + BO + ((j - AL) * 4 + wave) * 1024), 16, bv,
                WeightPanels ? ((((j - AL) * (64)) / 256) * PanelStep * 256 * BSTRIDE +
                                (((j - AL) * (64)) % 256) * BROW + k * 256 * BROW)
                             : ((((j - AL) * (PackedB ? 64 : 32)) / 256) * PanelStep * 256 * BSTRIDE +
                                (((j - AL) * (PackedB ? 64 : 32)) % 256) * BSTRIDE + k * BROW),
                0, 0);
        else if constexpr (j == AL + BL)
            __builtin_amdgcn_raw_ptr_buffer_load_lds(rsa, (lds_ptr_t)(s + SAO + wave * 256), 4, sa_v,
                                                     uint32_t(k) * uint32_t(p.lda_sf) * 4u, 0, (16));
        else
            __builtin_amdgcn_raw_ptr_buffer_load_lds(
                rsb, (lds_ptr_t)(s + SBO + (wave + (j - AL - BL - 1) * 4) * 256), 4, sv,
                uint32_t(k) * uint32_t(p.ldb_sf) * 4u + (j - AL - BL - 1) * uint32_t(ScaleRows) * PanelStep * 4u +
                    uint32_t(wave / 4) * (PanelStep - 1) * 256u * 4u,
                0, 0);
    };
    auto dma = [&](int stage, int k) { static_for<0, DMAS>([&](auto J) { dma_one(J, stage, k); }); };
    v16f acc[MF][NF];
    v8i af[2][MF], bf[2][NF];
    uint32_t ac[(2)][(MF)], bc[(2)][(NF)];

    auto read_frag = [&](auto Bank, auto F, int half, int stage, int kt = 0) {
        constexpr int bank = decltype(Bank)::value, f = decltype(F)::value;
        const uint8_t *st = sm + stage * STEP;
        if constexpr (f >= 1 && f <= MF) {
            constexpr int i = f - 1;
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
    auto step = [&](int k, auto first, auto last, auto ExplicitStage) {
        {
            const int stage = (decltype(ExplicitStage)::value);
            const int next = (stage + 1) % S, loadk = k + S < KT ? k + S : k + S - KT;
            {

                static_for<0, 2 * MF * NF>([&](auto T) {
                    constexpr int t = decltype(T)::value, half = t / (MF * NF);
                    constexpr int i = ((t % (MF * NF)) / NF);
                    constexpr int j = (t % NF);
                    constexpr int code_bank = (half);
                    const int bcode = (bc[code_bank][j]);
                    const int acode = (ac[code_bank][i]);
                    acc[i][j] = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
                        bf[half][j], af[half][i], decltype(first)::value && half == 0 ? v16f{} : acc[i][j],
                        PackedB ? 4 : 0, 0, (0), bcode, (0), acode);

                    constexpr int Reads = MF + NF, Half = MF * NF;
                    if constexpr ((t >= Half - Reads) && t < Half)
                        read_frag(std::integral_constant<int, 1>{}, std::integral_constant<int, t - (Half - Reads)>{},
                                  1, stage, k);
                    if constexpr ((!decltype(last)::value)) {
                        constexpr int FirstDma = Half + NF, Span = 2 * Half - FirstDma, NextRead = 2 * Half - Reads;
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
                            static_for<0, DMAS>([&](auto J) {
                                constexpr int j = decltype(J)::value;
                                if constexpr (t == FirstDma + j * Span / DMAS)
                                    dma_one(J, stage, loadk);
                            });
                        }
                        if constexpr (t == NextRead) {
                            // The short tile reaches this rendezvous before issuing k+2 loads:
                            // drain the pending k+1 stage. The original tile keeps its exact wait.
                            // In the partial register policy, the bound counts only newer A DMA.
                            // Additional B reads are deliberately excluded: this can strengthen the
                            // wait, not permit more newer requests than the proven A-only bound.
                            // Next-step B reads start only after this rendezvous. Keep the fully
                            // drained register policy as a separate numerical/performance control.
                            __builtin_amdgcn_s_waitcnt(waitcnt_vm(
                                (((!PackedB) && MF >= 2
                                      ? (S - 1) * DMAS
                                      : (S - 2) * DMAS + (NextRead >= FirstDma
                                                              ? ((NextRead - FirstDma + 1) * DMAS + Span - 1) / Span
                                                              : 0)))));
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
        static_for<0, S>([&](auto I) { dma(decltype(I)::value, decltype(I)::value); });
        __builtin_amdgcn_s_waitcnt(waitcnt_vm(((S - 1) * DMAS)));
        asm volatile("s_barrier" ::: "memory");
    }
    {

        static_for<0, MF + NF>([&](auto F) { read_frag(std::integral_constant<int, 0>{}, F, 0, tile_phase); });

        step(0, std::true_type{}, std::false_type{}, std::integral_constant<int, 0>{});
        {
            // Each loop body has a compile-time LDS base. Preserve increasing K,
            // replacement DMA counts, ready waits and the distinct final iteration.
#pragma unroll 1
            for (int base = 1; base < KT - 1; base += S)
                static_for<0, S>([&](auto J) {
                    constexpr int j = decltype(J)::value;
                    if (base + j < KT - 1)
                        step(base + j, std::false_type{}, std::false_type{},
                             std::integral_constant<int, (1 + j) % S>{});
                });
        }
        step(KT - 1, std::false_type{}, std::true_type{}, std::integral_constant<int, (KT - 1) % S>{});
    }
    asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)\n\ts_barrier" ::: "memory");

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
