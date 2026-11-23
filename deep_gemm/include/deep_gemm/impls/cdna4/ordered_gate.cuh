#pragma once
#include <deep_gemm/impls/cdna4/device_common.h>
#include <deep_gemm/impls/cdna4/gemm_args.h>

// Compute the original split-K partials independently, then fold them in
// their original order. The partials are read by this block from LDS.
template <int FM, int NF, int Batch, int Splits, int Span> __device__ __forceinline__ void ordered_gate_tile(deep_gemm::cdna4::GemmArgs p) {
    constexpr int K = 5120, BM = 16 * FM, BN = 16 * NF;
    const int tid = threadIdx.x, lane = tid % 64, wave = tid / 64, r = lane % 16, g = lane / 16;
    const int r0 = blockIdx.y * BM, c0 = blockIdx.x * BN;
    extern __shared__ float partial[];
    const auto *a = reinterpret_cast<const uint16_t *>(p.A);
    const auto *b = reinterpret_cast<const uint16_t *>(p.B);
    float totals[FM * NF]{};
#pragma unroll 1
    for (int first = 0; first < Splits; first += Batch) {
        for (int piece = wave; piece < Batch && first + piece < Splits; piece += 4) {
            const int begin = (first + piece) * Span;
            v4f acc[FM][NF]{};
#pragma unroll
            for (int kk = 0; kk < Span; kk += 32) {
                if (begin + kk >= K)
                    break;
                v4i af[FM], bf[NF];
#pragma unroll
                for (int i = 0; i < FM; ++i) {
                    const int row = r0 + i * 16 + r;
                    af[i] = row < p.M ? *reinterpret_cast<const v4i *>(a + int64_t(row) * K + begin + kk + g * 8) : v4i{};
                }
#pragma unroll
                for (int j = 0; j < NF; ++j)
#if CDNA4_GATE_PACKED_B == 1
                    bf[j] = *reinterpret_cast<const v4i *>(b + int64_t(c0 + j * 16) * K + (begin + kk) * 16 + lane * 8);
#elif CDNA4_GATE_PACKED_B == 2
                    bf[j] = *reinterpret_cast<const v4i *>(b + int64_t(begin + kk) * p.N + (c0 + j * 16) * 32 + lane * 8);
#else
                    bf[j] = *reinterpret_cast<const v4i *>(b + int64_t(c0 + j * 16 + r) * K + begin + kk + g * 8);
#endif
#pragma unroll
                for (int i = 0; i < FM; ++i)
#pragma unroll
                    for (int j = 0; j < NF; ++j)
                        acc[i][j] = mfma_bf16_16x16x32(bf[j], af[i], acc[i][j]);
            }
#pragma unroll
            for (int i = 0; i < FM; ++i)
#pragma unroll
                for (int j = 0; j < NF; ++j)
                    *reinterpret_cast<v4f *>(partial + (piece * BM + i * 16 + r) * BN + j * 16 + g * 4) = acc[i][j];
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < FM * NF; ++i) {
            const int el = tid + i * 256;
#pragma unroll
            for (int piece = 0; piece < Batch; ++piece) {
                if (first + piece < Splits) {
                    const float x = partial[piece * BM * BN + el];
                    totals[i] = first + piece == 0 ? x : totals[i] + x;
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < FM * NF; ++i) {
        const int el = tid + i * 256, row = r0 + el / BN;
        if (row < p.M)
            static_cast<float *>(p.D)[int64_t(row) * p.ldd + c0 + el % BN] = totals[i];
    }
}
