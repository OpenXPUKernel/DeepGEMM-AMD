#pragma once
#include <deep_gemm/impls/cdna4/device_common.h>
#include <deep_gemm/impls/cdna4/gemm_args.h>

// Four waves compute each original split. Split groups share only the final
// FP32 exchange plane, which is read in split order by group zero.
template <int BM, int BN, int OriginalSplits, int G, int Stages, int K, int N>
__device__ __forceinline__ void cooperative_dense_splits(deep_gemm::cdna4::GemmArgs p) {
    static_assert(G >= 2 && G <= 4 && (Stages == 1 || Stages == 2));
    constexpr int MF = BM / 32, NF = BN / 32, KT = K / 128, Span = (KT + OriginalSplits - 1) / OriginalSplits;
    constexpr int Splits = (KT + Span - 1) / Span;
    constexpr int SA = (BM + BN) * 128, SB = SA + ((BM + 63) / 64) * 256;
#ifdef CDNA4_DENSE_GROUP_SCALE
    constexpr int Used = SB;
#else
    constexpr int Used = SB + ((BN + 63) / 64) * 256;
#endif
    constexpr int Step = (Used + 1023) / 1024 * 1024;
    static_assert(G * Stages * Step <= 163840 && (G - 1) * BM * BN * 4 <= 163840);
    extern __shared__ __attribute__((aligned(1024))) uint8_t sm[];
    const int group = __builtin_amdgcn_readfirstlane(int(threadIdx.x) / 256), tid = threadIdx.x % 256;
    const int lane = tid % 64, wave = __builtin_amdgcn_readfirstlane(tid / 64), g = lane / 16, r = lane % 16;
    const int wm = wave / 2, wn = wave % 2, n0 = blockIdx.x * BN, m0 = blockIdx.y * BM;
    const int rows = min(BM, p.M - m0);
    if (rows <= 0)
        return;
    auto ra = make_rsrc(p.A + int64_t(m0) * p.lda, (rows - 1) * p.lda + K);
    auto rb = make_rsrc(p.B + int64_t(n0) * p.ldb, (BN - 1) * p.ldb + K);
    auto rsa = make_rsrc(p.sfa + m0, ((KT - 1) * p.lda_sf + rows) * 4);
#ifndef CDNA4_DENSE_GROUP_SCALE
    auto rsb = make_rsrc(p.sfb + n0, ((KT - 1) * p.ldb_sf + BN) * 4);
#endif
    const int dr = tid / 8, chunk = ((tid % 8) ^ ((dr >> 1) & 7)) * 16;
    auto dma = [&](int slot, int k) {
        uint8_t *base = sm + (group * Stages + slot) * Step;
        static_for<0, BM / 32>([&](auto J) {
            constexpr int j = decltype(J)::value;
            __builtin_amdgcn_raw_ptr_buffer_load_lds(
                ra, (lds_ptr_t)(base + j * 4096 + wave * 1024), 16, dr * p.lda + chunk, j * 32 * p.lda + k * 128, 0, 0);
        });
        static_for<0, BN / 32>([&](auto J) {
            constexpr int j = decltype(J)::value;
            __builtin_amdgcn_raw_ptr_buffer_load_lds(rb, (lds_ptr_t)(base + BM * 128 + j * 4096 + wave * 1024), 16,
#ifdef CDNA4_DENSE_PANELS
                dr * 128 + chunk, j * 32 * K + k * 4096, 0, 0);
#else
                dr * p.ldb + chunk, j * 32 * p.ldb + k * 128, 0, 0);
#endif
        });
        if (wave < (BM + 63) / 64)
            __builtin_amdgcn_raw_ptr_buffer_load_lds(rsa, (lds_ptr_t)(base + SA + wave * 256), 4, tid * 4, k * p.lda_sf * 4, 0, 0);
#ifndef CDNA4_DENSE_GROUP_SCALE
        if (wave < (BN + 63) / 64)
            __builtin_amdgcn_raw_ptr_buffer_load_lds(rsb, (lds_ptr_t)(base + SB + wave * 256), 4, tid * 4, k * p.ldb_sf * 4, 0, 0);
#endif
    };
    auto fragment = [&](const uint8_t *base, int row) {
        const int bank = (row >> 1) & 7;
        const v4i lo = *reinterpret_cast<const v4i *>(base + row * 128 + (g ^ bank) * 16);
        const v4i hi = *reinterpret_cast<const v4i *>(base + row * 128 + ((g + 4) ^ bank) * 16);
        return v8i{lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]};
    };
    v4f sum[MF][NF]{};
#pragma unroll 1
    for (int batch = 0; batch < Splits; batch += G) {
        const int split = batch + group, k0 = split * Span;
        v4f partial[MF][NF]{};
        if constexpr (Stages == 2)
            if (k0 < KT)
                dma(0, k0);
#pragma unroll 1
        for (int s = 0; s < min(Span, KT - batch * Span); ++s) {
            const int k = k0 + s, slot = s % Stages;
            if constexpr (Stages == 1)
                if (k < KT)
                    dma(0, k);
            __builtin_amdgcn_s_waitcnt(waitcnt_vm(0));
            asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
            if constexpr (Stages == 2)
                if (s + 1 < Span && k + 1 < KT)
                    dma(1 - slot, k + 1);
            if (k < KT) {
                const uint8_t *base = sm + (group * Stages + slot) * Step;
                v8i a[MF], b[NF];
                uint32_t ac[MF], bc[NF];
#pragma unroll
                for (int i = 0; i < MF; ++i) {
                    int row = wm * (BM / 2) + 16 * i + r;
                    a[i] = fragment(base, row);
                    ac[i] = base[SA + row * 4 + g];
                }
#pragma unroll
                for (int j = 0; j < NF; ++j) {
                    int col = wn * (BN / 2) + 16 * j + r;
                    b[j] = fragment(base + BM * 128, col);
#ifdef CDNA4_DENSE_GROUP_SCALE
                    bc[j] = uint32_t(p.sfb[((n0 + wn * (BN / 2) + j * 16) / 32) * KT + k]) >> (g * 8);
#else
                    bc[j] = base[SB + col * 4 + g];
#endif
                }
#pragma unroll
                for (int i = 0; i < MF; ++i)
#pragma unroll
                    for (int j = 0; j < NF; ++j)
                        partial[i][j] =
                            __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(b[j], a[i], partial[i][j], 0, 0, 0, bc[j], 0, ac[i]);
            }
            if constexpr (Stages == 1)
                asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
        }
        asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)\n\ts_barrier" ::: "memory");
        auto *exchange = reinterpret_cast<v4f *>(sm);
        if (group && split < Splits) {
#pragma unroll
            for (int i = 0; i < MF; ++i)
#pragma unroll
                for (int j = 0; j < NF; ++j)
                    exchange[(((group - 1) * MF + i) * NF + j) * 256 + tid] = partial[i][j];
        }
        asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
        if (!group) {
#pragma unroll
            for (int i = 0; i < MF; ++i)
#pragma unroll
                for (int j = 0; j < NF; ++j) {
                    auto v = batch == 0 ? partial[i][j] : sum[i][j] + partial[i][j];
#pragma unroll
                    for (int h = 1; h < G; ++h)
                        if (batch + h < Splits)
                            v += exchange[(((h - 1) * MF + i) * NF + j) * 256 + tid];
                    sum[i][j] = v;
                }
        }
        asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
    }
    if (!group) {
#pragma unroll
        for (int i = 0; i < MF; ++i) {
            const int row = m0 + wm * (BM / 2) + 16 * i + r;
#pragma unroll
            for (int j = 0; j < NF; ++j) {
                const auto v = sum[i][j];
                const int col = n0 + wn * (BN / 2) + 16 * j + 4 * g;
                if (row < p.M)
                    *reinterpret_cast<v2u *>(static_cast<uint16_t *>(p.D) + int64_t(row) * p.ldd + col) =
                        v2u{pack_bf16x2(v[0], v[1]), pack_bf16x2(v[2], v[3])};
            }
        }
    }
}
