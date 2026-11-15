// AMD GPU projection GEMM prototype: spatial wave tiles, no global split-K.
// A/B are K-major. FP8 scales are packed UE8M0, one byte per 32 K,
// with row-contiguous, column-strided int32 words. No external kernel library.
#pragma once
#include <deep_gemm/impls/cdna4/device_common.h>

namespace deep_gemm::cdna4::projection {
template <bool BF16, int NF, int FM = 1, int SPLITS = 1, bool DIRECT_LDS = false, int KP = 1>
__device__ __forceinline__ void run_lds(const void *ap, const void *bp, const int *sa, const int *sb, void *dp, int m,
                                        int n, int k, int sap, int sbp, int ldd) {
#if defined(__HIP_DEVICE_COMPILE__)
    constexpr int BM = 64 * FM, BN = 16 * NF, ES = BF16 ? 2 : 1, BK = 128 * KP / ES, CHUNKS = 8 * KP, ROWS = BM + BN,
                  COPIES = ROWS * CHUNKS / 256;
    static_assert(ROWS <= 256);
    static_assert(KP == 1 || (DIRECT_LDS && (KP == 2 || KP == 4)));
    __shared__ __attribute__((aligned(16))) v4i tiles[2][ROWS * CHUNKS];
    __shared__ unsigned scales[2][ROWS * KP];
    int tid = threadIdx.x, lane = tid % 64, g = lane / 16, r = lane % 16, wave = tid / 64;
    int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    auto ar = make_rsrc(ap, uint32_t(size_t(m) * k * ES));
    auto br = make_rsrc(bp, uint32_t(size_t(n) * k * ES));
    v4f acc[FM][NF] = {};
    v4i pending[COPIES];
    unsigned scale = 127;
    auto load = [&](int step) {
#pragma unroll
        for (int i = 0; i < COPIES; ++i) {
            int row = (tid + i * 256) / 8, chunk = tid % 8;
            const unsigned offset = ((row < BM ? m0 + row : n0 + row - BM) * k * ES) + step * 128 + chunk * 16;
            pending[i] = (v4i)__builtin_amdgcn_raw_buffer_load_b128(row < BM ? ar : br, offset, 0, 0);
        }
        if constexpr (!BF16)
            if (tid < ROWS) {
                int row = tid < BM ? m0 + tid : n0 + tid - BM;
                scale = row < (tid < BM ? m : n) ? unsigned((tid < BM ? sa : sb)[step * (tid < BM ? sap : sbp) + row])
                                                 : 0x7f7f7f7fu;
            }
    };
    auto stage = [&](int phase) {
#pragma unroll
        for (int i = 0; i < COPIES; ++i) {
            int row = (tid + i * 256) / 8, chunk = tid % 8;
            tiles[phase][row * 8 + (chunk ^ (row & 7))] = pending[i];
        }
        if constexpr (!BF16)
            if (tid < ROWS)
                scales[phase][tid] = scale;
    };
    auto dma = [&](int step, int phase) {
#pragma unroll
        for (int i = 0; i < COPIES; ++i) {
            const int row = (tid + i * 256) / CHUNKS, chunk = (tid % CHUNKS) ^ (row & 7);
            const unsigned offset = ((row < BM ? m0 + row : n0 + row - BM) * k * ES) + step * (128 * KP) + chunk * 16;
            const bool is_a = __builtin_amdgcn_readfirstlane(row < BM);
            __builtin_amdgcn_raw_ptr_buffer_load_lds(is_a ? ar : br, (lds_ptr_t)(&tiles[phase][i * 256 + wave * 64]),
                                                     16, offset, 0, 0, 0);
        }
        if constexpr (!BF16) {
#pragma unroll
            for (int index = tid; index < ROWS * KP; index += 256) {
                int local = index % ROWS, sub = index / ROWS;
                int row = local < BM ? m0 + local : n0 + local - BM;
                scales[phase][index] =
                    row < (local < BM ? m : n)
                        ? unsigned((local < BM ? sa : sb)[(step * KP + sub) * (local < BM ? sap : sbp) + row])
                        : 0x7f7f7f7fu;
            }
        }
    };
    const int begin = (k / BK) * blockIdx.z / SPLITS, end = (k / BK) * (blockIdx.z + 1) / SPLITS;
    if constexpr (DIRECT_LDS) {
        dma(begin, 0);
        asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)\n\ts_barrier" ::: "memory");
    } else {
        load(begin);
        stage(0);
        __syncthreads();
    }
    for (int step = begin; step < end; ++step) {
        const int phase = (step - begin) & 1;
        if (step + 1 < end) {
            if constexpr (DIRECT_LDS)
                dma(step + 1, phase ^ 1);
            else
                load(step + 1);
        }
#pragma unroll
        for (int part = 0; part < KP; ++part) {
#pragma unroll
            for (int i = 0; i < FM; ++i) {
                int arow = wave * 16 * FM + i * 16 + r;
                v4i a0 = tiles[phase][arow * CHUNKS + ((part * 8 + g) ^ (arow & 7))],
                    a1 = tiles[phase][arow * CHUNKS + ((part * 8 + g + 4) ^ (arow & 7))];
                unsigned ac = 127;
                if constexpr (!BF16)
                    ac = scales[phase][part * ROWS + arow] >> (8 * g);
#pragma unroll
                for (int j = 0; j < NF; ++j) {
                    int brow = BM + 16 * j + r;
                    v4i b0 = tiles[phase][brow * CHUNKS + ((part * 8 + g) ^ (brow & 7))],
                        b1 = tiles[phase][brow * CHUNKS + ((part * 8 + g + 4) ^ (brow & 7))];
                    if constexpr (BF16) {
                        acc[i][j] = mfma_bf16_16x16x32(b0, a0, acc[i][j]);
                        acc[i][j] = mfma_bf16_16x16x32(b1, a1, acc[i][j]);
                    } else
                        acc[i][j] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
                            cat8(b0, b1), cat8(a0, a1), acc[i][j], 0, 0, 0,
                            scales[phase][part * ROWS + brow] >> (8 * g), 0, ac);
                }
            }
        }
        if (step + 1 < end) {
            if constexpr (DIRECT_LDS)
                asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)\n\ts_barrier" ::: "memory");
            else {
                stage(phase ^ 1);
                __syncthreads();
            }
        }
    }
#pragma unroll
    for (int i = 0; i < FM; ++i) {
        if (m0 + wave * 16 * FM + 16 * i + r < m) {
#pragma unroll
            for (int j = 0; j < NF; ++j) {
                const int col0 = n0 + 16 * j + 4 * g;
                const size_t base = size_t(m0 + wave * 16 * FM + 16 * i + r) * ldd + col0;
                if (col0 + 3 < n && ldd % 4 == 0 &&
                    (reinterpret_cast<uintptr_t>(dp) % (BF16 || SPLITS > 1 ? 16 : 8)) == 0) {
                    if constexpr (SPLITS > 1)
                        *(v4f *)(static_cast<float *>(dp) + size_t(blockIdx.z) * m * ldd + base) = acc[i][j];
                    else if constexpr (BF16)
                        *(v4f *)(static_cast<float *>(dp) + base) = acc[i][j];
                    else {
                        const v2u packed{pack_bf16x2(acc[i][j][0], acc[i][j][1]),
                                         pack_bf16x2(acc[i][j][2], acc[i][j][3])};
                        *(v2u *)(static_cast<uint16_t *>(dp) + base) = packed;
                    }
                } else {
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        int col = n0 + 16 * j + 4 * g + e;
                        if (col < n) {
                            size_t offset = size_t(m0 + wave * 16 * FM + 16 * i + r) * ldd + col;
                            if constexpr (SPLITS > 1)
                                static_cast<float *>(dp)[size_t(blockIdx.z) * m * ldd + offset] = acc[i][j][e];
                            else if constexpr (BF16)
                                static_cast<float *>(dp)[offset] = acc[i][j][e];
                            else
                                static_cast<uint16_t *>(dp)[offset] = f32_to_bf16(acc[i][j][e]);
                        }
                    }
                }
            }
        }
    }
#endif
}
template <bool BF16, int SPLITS>
__device__ __forceinline__ void reduce(const float *partials, void *out, int elements) {
    int index = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (index >= elements)
        return;
    v4f value = *(const v4f *)(partials + index);
#pragma unroll
    for (int s = 1; s < SPLITS; ++s)
        value += *(const v4f *)(partials + size_t(s) * elements + index);
    if constexpr (BF16)
        *(v4f *)((float *)out + index) = value;
    else {
        v2u packed{pack_bf16x2(value[0], value[1]), pack_bf16x2(value[2], value[3])};
        *(v2u *)((uint16_t *)out + index) = packed;
    }
}
template <bool BF16, int NF, int DEPTH = 2>
__device__ __forceinline__ void run(const void *ap, const void *bp, const int *sa, const int *sb, void *dp, int m,
                                    int n, int k, int sap, int sbp, int ldd) {
#if defined(__HIP_DEVICE_COMPILE__)
    constexpr int ES = BF16 ? 2 : 1, BK = BF16 ? 32 : 128, HALVES = BF16 ? 1 : 2;
    int lane = threadIdx.x % 64, g = lane / 16, r = lane % 16, wave = threadIdx.x / 64;
    int m0 = blockIdx.y * 64 + wave * 16, n0 = blockIdx.x * (16 * NF);
    auto ar = make_rsrc(ap, uint32_t(size_t(m) * k * ES));
    auto br = make_rsrc(bp, uint32_t(size_t(n) * k * ES));
    v4i aa[DEPTH][HALVES], bb[DEPTH][NF][HALVES];
    unsigned as[DEPTH], bs[DEPTH][NF];
    v4f acc[NF] = {};
    auto load = [&](auto D, int s) {
        constexpr int d = decltype(D)::value;
        const bool valid = s < k / BK;
#pragma unroll
        for (int h = 0; h < HALVES; ++h) {
            const unsigned off = s * BK * ES + g * 16 + h * 64;
            aa[d][h] = valid ? (v4i)__builtin_amdgcn_raw_buffer_load_b128(ar, ((m0 + r) * k * ES) + off, 0, 0) : v4i{};
#pragma unroll
            for (int j = 0; j < NF; ++j)
                bb[d][j][h] =
                    valid ? (v4i)__builtin_amdgcn_raw_buffer_load_b128(br, ((n0 + 16 * j + r) * k * ES) + off, 0, 0)
                          : v4i{};
        }
        if constexpr (!BF16) {
            as[d] = valid && m0 + r < m ? unsigned(sa[s * sap + m0 + r]) >> (8 * g) : 127;
#pragma unroll
            for (int j = 0; j < NF; ++j)
                bs[d][j] = valid && n0 + 16 * j + r < n ? unsigned(sb[s * sbp + n0 + 16 * j + r]) >> (8 * g) : 127;
        }
    };
    auto compute = [&](auto D) {
        constexpr int d = decltype(D)::value;
#pragma unroll
        for (int j = 0; j < NF; ++j) {
            if constexpr (BF16)
                acc[j] = mfma_bf16_16x16x32(bb[d][j][0], aa[d][0], acc[j]);
            else
                acc[j] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
                    cat8(bb[d][j][0], bb[d][j][1]), cat8(aa[d][0], aa[d][1]), acc[j], 0, 0, 0, bs[d][j], 0, as[d]);
        }
    };
    static_for<0, DEPTH>([&](auto D) { load(D, decltype(D)::value); });
    for (int base = 0; base < k / BK; base += DEPTH) {
        static_for<0, DEPTH>([&](auto D) {
            constexpr int d = decltype(D)::value;
            if (base + d < k / BK) {
                compute(D);
                load(D, base + d + DEPTH);
            }
        });
    }
    if (m0 + r < m) {
#pragma unroll
        for (int j = 0; j < NF; ++j) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                int col = n0 + 16 * j + 4 * g + e;
                if (col < n) {
                    if constexpr (BF16)
                        static_cast<float *>(dp)[size_t(m0 + r) * ldd + col] = acc[j][e];
                    else
                        static_cast<uint16_t *>(dp)[size_t(m0 + r) * ldd + col] = f32_to_bf16(acc[j][e]);
                }
            }
        }
    }
#endif
}
} // namespace deep_gemm::cdna4::projection
