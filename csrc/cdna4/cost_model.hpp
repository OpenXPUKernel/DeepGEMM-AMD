#pragma once

#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "deep_gemm/impls/cdna4/gemm_args.h"

constexpr int MXFP8_OUT_BF16 = 0;
constexpr int MXFP8_OUT_FP32 = 1;

namespace deep_gemm::cdna4 {

enum class KernelVariant {
    mfma_16x16x128_data_parallel,
    mfma_16x16x128_persistent,
    mfma_16x16x128_split_k,
    // B streamed into registers per wave (fp8_gemm_mfma_16x16x128_small_m.cuh)
    mfma_16x16x128_small_m,
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
    // Small-M kernel: K steps in flight per wave, waves per workgroup, and whether K has a
    // partial last step
    int depth = 0;
    int waves = 4;
    bool k_tail = false;
    OperandType a_type = OperandType::Fp8, b_type = OperandType::Fp8;
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

// Small-M kernel: tile rows 16 FM >= rows (the rows of a GEMM, or the expected rows of a group),
// 16 NF columns per workgroup and four waves splitting K. Wider column blocks cut the A reads per B
// byte (up to NF = max(FM, 2)) while at least 192 workgroups remain for the compute units.
// "DG_CDNA4_SMALL_M=NF,DEPTH,WAVES" overrides the plan (0 keeps a value), "off" disables the kernel.
inline KernelConfig get_small_m_config(const int rows, const int n, const int k, const int scale_gran_128,
                                       const GemmType type = GemmType::Normal, const int groups = 1) {
    KernelConfig config{KernelVariant::mfma_16x16x128_small_m};
    config.type = type;
    config.scale_gran_128 = scale_gran_128;
    const int fm = rows <= 16 ? 1 : rows <= 32 ? 2 : rows <= 64 ? 4 : 8;
    int nf = 1;
    for (int c = std::max(fm, 2); c > 1 && nf == 1; c /= 2)
        if (c <= 4 && int64_t(groups) * ceil_div(n, 16 * c) >= 192)
            nf = c;
    int depth = nf >= 4 ? 2 : 3, waves = 4;
    if (const char* env = std::getenv("DG_CDNA4_SMALL_M")) {
        int e_nf = 0, e_depth = 0, e_waves = 0;
        if (std::sscanf(env, "%d,%d,%d", &e_nf, &e_depth, &e_waves) >= 1) {
            nf = e_nf > 0 ? e_nf : nf;
            depth = e_depth > 0 ? e_depth : depth;
            waves = e_waves > 0 ? e_waves : waves;
        }
    }
    config.block_m = 16 * fm;
    config.block_n = 16 * nf;
    config.depth = depth;
    config.waves = waves;
    config.k_tail = k % 128 != 0;
    return config;
}

// Rows up to 32 always stream B faster than the 256x256 tiles; up to 64 rows, the larger weight
// matrices still go to the 256x256 tiles (their A reads per B byte grow with the rows).
inline bool small_m_eligible(const int rows, const int n, const int k, const int mn_major) {
    int max_rows = int64_t(n) * k > (int64_t(64) << 20) ? 32 : 64;
    if (const char* env = std::getenv("DG_CDNA4_SMALL_M")) {
        if (std::string(env) == "off")
            return false;
        max_rows = 128; // an explicit configuration covers 8 A fragments too
    }
    return mn_major == 0 && rows > 0 && rows <= max_rows;
}

// The planners take k = 128 x the K steps: K, or 2 K with BF16 operands (64 K values per step).
inline int planner_k(const int k, const OperandType a_type) { return a_type == OperandType::Bf16 ? 2 * k : k; }

inline KernelConfig get_best_config(const int m, const int n, const int k,
                                    const int out_dtype, const bool has_c,
                                    const int ldd, const bool d_aligned16,
                                    const int mn_major = 0, const int scale_gran_128 = 0,
                                    const OperandType a_type = OperandType::Fp8,
                                    const OperandType b_type = OperandType::Fp8) {
    if (small_m_eligible(m, n, k, mn_major)) {
        auto config = get_small_m_config(m, n, k, scale_gran_128);
        config.a_type = a_type;
        config.b_type = b_type;
        return config;
    }
    KernelConfig config{KernelVariant::mfma_16x16x128_data_parallel};
    config.a_type = a_type;
    config.b_type = b_type;
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

// m-grouped GEMMs (contiguous, PSUM and masked layouts; masked layouts may accumulate into FP32)
inline KernelConfig get_m_grouped_config(const GemmType type, const int k, const int ldd,
                                         const bool d_aligned16, const int mn_major,
                                         const int scale_gran_128, const int out_dtype = MXFP8_OUT_BF16,
                                         const bool has_c = false) {
    KernelConfig config{KernelVariant::mfma_16x16x128_data_parallel};
    config.type = type;
    config.mn_major = mn_major;
    config.scale_gran_128 = scale_gran_128;
    if (persistent_eligible(k, out_dtype, has_c, ldd, d_aligned16))
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
