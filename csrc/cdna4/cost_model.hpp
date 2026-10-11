#pragma once

#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdint>

#include "deep_gemm/impls/cdna4/gemm_args.h"

constexpr int MXFP8_OUT_BF16 = 0;
constexpr int MXFP8_OUT_FP32 = 1;

namespace deep_gemm::cdna4 {

enum class KernelVariant {
    mfma_16x16x128_data_parallel,
    mfma_16x16x128_persistent,
    mfma_16x16x128_split_k,
};

struct KernelConfig {
    KernelVariant variant;
    int block_m = 256;
    int block_n = 256;
    int block_k = 128;
    int split_k = 1;
    // Bit 0 describes A, bit 1 describes B.
    int mn_major = 0;
    int scale_gran_128 = 0;
    GemmType type = GemmType::Normal;
};

inline int ceil_div(const int x, const int y) { return x / y + (x % y != 0); }

inline int num_compute_units() {
    int device = 0;
    hipDeviceProp_t properties{};
    if (hipGetDevice(&device) != hipSuccess || hipGetDeviceProperties(&properties, device) != hipSuccess)
        return 1;
    return std::max(properties.multiProcessorCount, 1);
}

// Balance additional parallelism against partial-output traffic and reduction
// overhead. These empirical coefficients match the production tutorial planner.
inline int plan_splits(const int m, const int n, const int k, const int compute_units) {
    if (m <= 0 || n <= 0 || k <= 0)
        return 0;
    const auto tiles = int64_t(ceil_div(m, 256)) * ceil_div(n, 256);
    const int k_tiles = ceil_div(k, 128);
    int splits = 2 * tiles > compute_units ? 1 :
        std::min(static_cast<int>(compute_units / tiles), k_tiles / 2);
    while (splits > 1 && splits * double(m) * n * 8 / 5e12 + 5e-6 >
                         10.5e-9 * k * (1 - 1.0 / splits))
        --splits;
    splits = std::clamp(splits, 1, k_tiles);
    // Equal-sized K ranges must never create an empty final workgroup.
    return ceil_div(k_tiles, ceil_div(k_tiles, splits));
}

// The persistent kernel pipelines across tiles: every tile needs at least two full K steps, and
// its LDS epilogue stores aligned BF16 rows without accumulation.
inline bool persistent_eligible(const int k, const int out_dtype, const bool has_c,
                                const int ldd, const bool d_aligned16) {
    return out_dtype == MXFP8_OUT_BF16 && !has_c && k >= 256 && k % 128 == 0 &&
           d_aligned16 && ldd % 8 == 0 && ldd < (1 << 23);
}

inline KernelConfig get_best_config(const int m, const int n, const int k,
                                    const int out_dtype, const bool has_c,
                                    const int ldd, const bool d_aligned16,
                                    const int mn_major = 0, const int scale_gran_128 = 0) {
    KernelConfig config{KernelVariant::mfma_16x16x128_data_parallel};
    config.mn_major = mn_major;
    config.scale_gran_128 = scale_gran_128;
    config.split_k = plan_splits(m, n, k, num_compute_units());
    if (config.split_k > 1) {
        config.variant = KernelVariant::mfma_16x16x128_split_k;
    } else if (persistent_eligible(k, out_dtype, has_c, ldd, d_aligned16)) {
        config.variant = KernelVariant::mfma_16x16x128_persistent;
    }
    return config;
}

// m-grouped GEMMs (contiguous, PSUM and masked layouts): BF16 output, no accumulation.
inline KernelConfig get_m_grouped_config(const GemmType type, const int k, const int ldd,
                                         const bool d_aligned16, const int mn_major,
                                         const int scale_gran_128) {
    KernelConfig config{KernelVariant::mfma_16x16x128_data_parallel};
    config.type = type;
    config.mn_major = mn_major;
    config.scale_gran_128 = scale_gran_128;
    if (persistent_eligible(k, MXFP8_OUT_BF16, false, ldd, d_aligned16))
        config.variant = KernelVariant::mfma_16x16x128_persistent;
    return config;
}

// K-grouped GEMMs: one data-parallel tile per workgroup, K range from the group.
inline KernelConfig get_k_grouped_config(const GemmType type, const int mn_major,
                                         const int scale_gran_128) {
    KernelConfig config{KernelVariant::mfma_16x16x128_data_parallel};
    config.type = type;
    config.mn_major = mn_major;
    config.scale_gran_128 = scale_gran_128;
    return config;
}

}  // namespace deep_gemm::cdna4
