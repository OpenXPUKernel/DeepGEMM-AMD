#pragma once

#include <algorithm>
#include <cstdlib>

#include <cstdint>

constexpr int MXFP8_OUT_BF16 = 0;
constexpr int MXFP8_OUT_FP32 = 1;

namespace deep_gemm::cdna4 {

enum class KernelVariant { tiled_128, tiled_256, data_parallel, persistent, split_k };

struct KernelConfig {
    KernelVariant variant;
    int block_m;
    int block_n;
    int block_k;
    int split_k;
};

inline int ceil_div(const int x, const int y) { return (x + y - 1) / y; }

inline int num_compute_units() {
    int device = 0;
    if (hipGetDevice(&device) != hipSuccess)
        return 256;
    hipDeviceProp_t properties;
    return hipGetDeviceProperties(&properties, device) == hipSuccess ? properties.multiProcessorCount : 256;
}

// Shape cost model. The candidate set is deliberately small because each entry is
// a compile-time kernel specialization in the single HIP extension.
inline KernelConfig get_best_config(const int m, const int n, const int k,
                                  const int out_dtype, const bool has_c) {
    const int cus = num_compute_units();
    const int tiles_128 = ceil_div(m, 128) * ceil_div(n, 128);
    const int tiles_256 = ceil_div(m, 256) * ceil_div(n, 256);
    const int k_tiles = ceil_div(k, 128);
    const bool full_k = k % 128 == 0;
    const bool persistent = out_dtype == MXFP8_OUT_BF16 && !has_c && full_k && k >= 256;

    if (full_k && m <= 128 && n <= 128)
        return {KernelVariant::tiled_128, 128, 128, 128, 1};
    if (full_k && (m < 256 || n < 256) && tiles_128 >= 2)
        return {KernelVariant::tiled_256, 256, 256, 128, 1};
    if (persistent && tiles_256 >= cus)
        return {KernelVariant::persistent, 256, 256, 128, 1};

    const int suggested_splits = std::max(1, std::min(cus / std::max(tiles_256, 1), k_tiles / 2));
    if (suggested_splits > 1 && tiles_256 * 2 < cus)
        return {KernelVariant::split_k, 256, 256, 128, suggested_splits};
    return {KernelVariant::data_parallel, 256, 256, 128, 1};
}

}  // namespace deep_gemm::cdna4
