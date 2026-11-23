#pragma once
#include <deep_gemm/impls/cdna4/device_common.h>
#include <deep_gemm/impls/cdna4/gemm_args.h>

// A block covers a complete 128-channel quantization word. Four waves compute
// independent row/column quadrants, sharing both operands through two LDS stages.
template <int BM, int BN, int OutputSplits = 0, int Stages = 2, int K = 4096, int N = 1024>
__device__ __forceinline__ void packed_wv_tile(deep_gemm::cdna4::GemmArgs p) {
    constexpr int MF = BM / 32, NF = BN / 32;
    static_assert(K % 128 == 0 && N % BN == 0);
#ifdef CDNA4_DENSE_GROUP_SCALE
    constexpr int ScaleLoads = 1;
#else
    constexpr int ScaleLoads = 2;
#endif
    constexpr int SA = (BM + BN) * 128, SB = SA + 1024, Step = SA + ScaleLoads * 1024;
    extern __shared__ __attribute__((aligned(1024))) uint8_t staging[];
    const int tid = threadIdx.x, lane = tid % 64, wave = __builtin_amdgcn_readfirstlane(tid / 64);
    const int r = lane % 16, g = lane / 16, wm = wave / 2, wn = wave % 2;
    const int n0 = blockIdx.x * BN, m0 = blockIdx.y * BM, group = blockIdx.z;
    const int end = OutputSplits ? p.M : min(p.M, p.grouped_layout[group]);
    if (m0 >= end)
        return;
    const int rows = min(BM, end - m0);
    const auto ra = make_rsrc(p.A + group * p.a_group_stride + int64_t(m0) * p.lda, (rows - 1) * p.lda + K);
    const auto rb = make_rsrc(p.B + group * p.b_group_stride + int64_t(n0) * p.ldb, (BN - 1) * p.ldb + K);
    const auto rsa = make_rsrc(p.sfa + group * p.sfa_group_stride + m0, ((K / 128 - 1) * p.lda_sf + rows) * 4);
#ifndef CDNA4_DENSE_GROUP_SCALE
    const auto rsb = make_rsrc(p.sfb + group * p.sfb_group_stride + n0, ((K / 128 - 1) * p.ldb_sf + BN) * 4);
#endif
    const int dma_row = tid / 8, chunk = ((tid % 8) ^ ((dma_row >> 1) & 7)) * 16;
    auto dma = [&](int slot, int k) {
        uint8_t *base = staging + slot * Step;
        static_for<0, BM / 32>([&](auto J) {
            constexpr int j = decltype(J)::value;
            __builtin_amdgcn_raw_ptr_buffer_load_lds(
                ra, (lds_ptr_t)(base + j * 4096 + wave * 1024), 16, dma_row * p.lda + chunk, j * 32 * p.lda + k * 128, 0, 0);
        });
        static_for<0, BN / 32>([&](auto J) {
            constexpr int j = decltype(J)::value;
            __builtin_amdgcn_raw_ptr_buffer_load_lds(rb, (lds_ptr_t)(base + BM * 128 + j * 4096 + wave * 1024), 16,
#ifdef CDNA4_DENSE_PANELS
                dma_row * 128 + chunk, j * 32 * K + k * 4096, 0, 0);
#else
                dma_row * p.ldb + chunk, j * 32 * p.ldb + k * 128, 0, 0);
#endif
        });
        __builtin_amdgcn_raw_ptr_buffer_load_lds(rsa, (lds_ptr_t)(base + SA + wave * 256), 4, tid * 4, k * p.lda_sf * 4, 0, 0);
#ifndef CDNA4_DENSE_GROUP_SCALE
        __builtin_amdgcn_raw_ptr_buffer_load_lds(rsb, (lds_ptr_t)(base + SB + wave * 256), 4, tid * 4, k * p.ldb_sf * 4, 0, 0);
#endif
    };
    auto fragment = [&](uint8_t *base, int row) {
        const int bank = (row >> 1) & 7;
        const v4i a = *reinterpret_cast<const v4i *>(base + row * 128 + (g ^ bank) * 16);
        const v4i b = *reinterpret_cast<const v4i *>(base + row * 128 + ((g + 4) ^ bank) * 16);
        return v8i{a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]};
    };
    v4f acc[MF][NF]{}, sum[MF][NF]{};
    auto finish_split = [&](int k) {
        if constexpr (OutputSplits > 1) {
            constexpr int span = (K / 128 + OutputSplits - 1) / OutputSplits;
            if ((k + 1) % span == 0 && k + 1 < K / 128) {
#pragma unroll
                for (int i = 0; i < MF; ++i)
#pragma unroll
                    for (int j = 0; j < NF; ++j) {
                        sum[i][j] = (k + 1 == span) ? acc[i][j] : sum[i][j] + acc[i][j];
                        acc[i][j] = v4f{};
                    }
            }
        }
    };
    static_for<0, Stages - 1>([&](auto Slot) {
        constexpr int slot = decltype(Slot)::value;
        dma(slot, slot);
        __builtin_amdgcn_sched_barrier(0);
    });
#ifdef CDNA4_DENSE_FRAGMENT_PIPE
    v8i af[MF], bf[NF];
    uint32_t ac[MF], bc[NF];
    auto read_a = [&](int slot, int i, v8i &data, uint32_t &code) {
        const int row = wm * (BM / 2) + i * 16 + r;
        auto *base = staging + slot * Step;
        data = fragment(base, row);
        code = base[SA + row * 4 + g];
    };
    auto read_b = [&](int slot, int k, int j, v8i &data, uint32_t &code) {
        const int col = wn * (BN / 2) + j * 16 + r;
        auto *base = staging + slot * Step;
        data = fragment(base + BM * 128, col);
#ifdef CDNA4_DENSE_GROUP_SCALE
        code = uint32_t(p.sfb[group * p.sfb_group_stride + ((n0 + wn * (BN / 2) + j * 16) / 32) * (K / 128) + k]) >> (g * 8);
#else
        code = base[SB + col * 4 + g];
#endif
    };
    constexpr int Newer = (Stages - 2) * (BM / 32 + BN / 32 + ScaleLoads), KT = K / 128;
    __builtin_amdgcn_s_waitcnt(waitcnt_vm(Newer));
    asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
    static_for<0, MF>([&](auto I) { read_a(0, I, af[I], ac[I]); });
    static_for<0, NF>([&](auto J) { read_b(0, 0, J, bf[J], bc[J]); });
    auto compute = [&](auto Slot, int k) {
        constexpr int slot = decltype(Slot)::value, next = (slot + 1) % Stages, refill = (slot + Stages - 1) % Stages;
        if (k + Stages - 1 < KT)
            dma(refill, k + Stages - 1);
        __builtin_amdgcn_sched_barrier(0);
        constexpr int Ops = MF * NF, Reads = MF + NF;
#if CDNA4_DENSE_FRAGMENT_PIPE == 1
        constexpr int Cut = Ops > Reads ? Ops - Reads : 0;
        v8i future_a[MF], future_b[NF];
        uint32_t future_ac[MF], future_bc[NF];
#else
        constexpr int Cut = NF - 1 < (MF - 1) * NF ? NF - 1 : (MF - 1) * NF;
#endif
        static_for<0, Ops>([&](auto Op) {
            constexpr int t = decltype(Op)::value, i = t / NF, j = t % NF;
            acc[i][j] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(bf[j], af[i], acc[i][j], 0, 0, 0, bc[j], 0, ac[i]);
            if (k + 1 < KT) {
                if constexpr (t == Cut) {
                    if (k + Stages - 1 < KT)
                        __builtin_amdgcn_s_waitcnt(waitcnt_vm(Newer));
                    else
                        __builtin_amdgcn_s_waitcnt(waitcnt_vm(0));
                    asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
                }
#if CDNA4_DENSE_FRAGMENT_PIPE == 1
                static_for<0, Reads>([&](auto F) {
                    constexpr int f = decltype(F)::value;
                    if constexpr (t == Cut + f * (Ops - Cut) / Reads) {
                        if constexpr (f < MF)
                            read_a(next, f, future_a[f], future_ac[f]);
                        else
                            read_b(next, k + 1, f - MF, future_b[f - MF], future_bc[f - MF]);
                    }
                });
#else
                // Replace each register fragment after its last current use.
                if constexpr (j == NF - 1)
                    read_a(next, i, af[i], ac[i]);
                if constexpr (i == MF - 1)
                    read_b(next, k + 1, j, bf[j], bc[j]);
#endif
            }
            __builtin_amdgcn_sched_barrier(0);
        });
#if CDNA4_DENSE_FRAGMENT_PIPE == 1
        if (k + 1 < KT) {
            static_for<0, MF>([&](auto I) {
                af[I] = future_a[I];
                ac[I] = future_ac[I];
            });
            static_for<0, NF>([&](auto J) {
                bf[J] = future_b[J];
                bc[J] = future_bc[J];
            });
        }
#endif
        finish_split(k);
    };
#else
    auto compute = [&](auto Slot, int k) {
        constexpr int slot = decltype(Slot)::value, refill = (slot + Stages - 1) % Stages;
        constexpr int newer = (Stages - 2) * (BM / 32 + BN / 32 + ScaleLoads);
        if (k + Stages - 2 < K / 128)
            __builtin_amdgcn_s_waitcnt(waitcnt_vm(newer));
        else
            __builtin_amdgcn_s_waitcnt(waitcnt_vm(0));
        asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
        uint8_t *base = staging + slot * Step;
        if (k + Stages - 1 < K / 128)
            dma(refill, k + Stages - 1);
        __builtin_amdgcn_sched_barrier(0);
        v8i af[MF], bf[NF];
        uint32_t ac[MF], bc[NF];
#pragma unroll
        for (int i = 0; i < MF; ++i) {
            const int row = wm * (BM / 2) + i * 16 + r;
            af[i] = fragment(base, row);
            ac[i] = base[SA + row * 4 + g];
        }
#pragma unroll
        for (int j = 0; j < NF; ++j) {
            const int col = wn * (BN / 2) + j * 16 + r;
            bf[j] = fragment(base + BM * 128, col);
#ifdef CDNA4_DENSE_GROUP_SCALE
            const int block = (n0 + wn * (BN / 2) + j * 16) / 32;
            const uint32_t word = p.sfb[group * p.sfb_group_stride + block * (K / 128) + k];
            bc[j] = word >> (g * 8);
#else
            bc[j] = base[SB + col * 4 + g];
#endif
        }
#pragma unroll
        for (int i = 0; i < MF; ++i)
#pragma unroll
            for (int j = 0; j < NF; ++j)
                acc[i][j] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(bf[j], af[i], acc[i][j], 0, 0, 0, bc[j], 0, ac[i]);
        finish_split(k);
    };
#endif
#pragma unroll 1
    for (int base = 0; base < K / 128; base += Stages) {
        static_for<0, Stages>([&](auto Slot) {
            constexpr int slot = decltype(Slot)::value;
            if (base + slot < K / 128)
                compute(Slot, base + slot);
        });
    }
    if constexpr (OutputSplits) {
        // Match the existing workspace reduction in split order, locally.
#pragma unroll
        for (int i = 0; i < MF; ++i) {
            const int row = m0 + wm * (BM / 2) + i * 16 + r;
#pragma unroll
            for (int j = 0; j < NF; ++j) {
                const auto v = OutputSplits > 1 ? sum[i][j] + acc[i][j] : acc[i][j];
                const v2u pack{pack_bf16x2(v[0], v[1]), pack_bf16x2(v[2], v[3])};
                const int col = n0 + wn * (BN / 2) + 16 * j + 4 * g;
                if (row < end)
                    *reinterpret_cast<v2u *>(static_cast<uint16_t *>(p.D) + int64_t(row) * p.ldd + col) = pack;
            }
        }
        return;
    }
#pragma unroll
    for (int i = 0; i < MF; ++i) {
        const int row = m0 + wm * (BM / 2) + i * 16 + r;
        uint32_t word = 0;
#pragma unroll
        for (int pair = 0; pair < NF / 2; ++pair) {
            float v[8], peak = 1.e-4f;
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const v4f z = acc[i][pair * 2 + j];
                unpack_bf16x2(pack_bf16x2(z[0], z[1]), v[j * 4], v[j * 4 + 1]);
                unpack_bf16x2(pack_bf16x2(z[2], z[3]), v[j * 4 + 2], v[j * 4 + 3]);
            }
#pragma unroll
            for (int j = 0; j < 8; ++j)
                peak = fmaxf(peak, fabsf(v[j]));
            peak = fmaxf(peak, __shfl_xor(peak, 16));
            peak = fmaxf(peak, __shfl_xor(peak, 32));
            const uint32_t exponent = ((__float_as_uint(peak * (1.f / 448.f)) - 1) >> 23) + 1;
            word |= exponent << (8 * pair);
            const float inv = __uint_as_float((254u - exponent) << 23);
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                float q[4];
#pragma unroll
                for (int z = 0; z < 4; ++z)
                    q[z] = fminf(fmaxf(v[4 * j + z] * inv, -448.f), 448.f);
                const int lo = __builtin_amdgcn_cvt_pk_fp8_f32(q[0], q[1], 0, false);
                const int data = __builtin_amdgcn_cvt_pk_fp8_f32(q[2], q[3], lo, true);
                const int col = n0 + wn * (BN / 2) + pair * 32 + j * 16 + 4 * g;
                if (row < end)
                    *reinterpret_cast<uint32_t *>(static_cast<uint8_t *>(p.D) + group * p.d_group_stride + int64_t(row) * p.ldd + col) =
                        data;
            }
        }
        if (g == 0 && row < end) {
            const int col = n0 + wn * (BN / 2);
            int32_t *dst = p.moe_sf + int64_t((group * N + col) / 128) * p.moe_sf_pitch + row;
            if constexpr (BN == 64)
                reinterpret_cast<uint8_t *>(dst)[(col / 32) % 4] = uint8_t(word);
            else if constexpr (BN == 128)
                reinterpret_cast<uint16_t *>(dst)[(col / 64) % 2] = uint16_t(word);
            else
                *dst = word;
        }
    }
}
