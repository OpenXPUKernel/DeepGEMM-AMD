// Lossless static weight layout conversion for AMD GPU projection tiles.
// The caller must retain matching K-stage metadata and zero-padded row count.
#pragma once
#include <deep_gemm/impls/cdna4/device_common.h>
namespace deep_gemm::cdna4::projection {
template <bool BF16, int KP>
__device__ __forceinline__ void pack_weights(const void *source, void *destination, int n, int k) {
#if defined(__HIP_DEVICE_COMPILE__)
    constexpr int ES = BF16 ? 2 : 1, CHUNKS = KP * 8, BK = KP * 128 / ES;
    const int steps = k / BK, total = ((n + 63) / 64) * steps * 64 * CHUNKS;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total)
        return;
    int chunk = index % CHUNKS, row = (index / CHUNKS) % 64;
    int step = (index / (CHUNKS * 64)) % steps, tile = index / (CHUNKS * 64 * steps);
    int global_row = tile * 64 + row;
    v4i value{};
    if (global_row < n)
        value = *(const v4i *)((const uint8_t *)source + size_t(global_row) * k * ES + step * KP * 128 +
                               (chunk ^ ((row >> 1) & 7)) * 16);
    ((v4i *)destination)[index] = value;
#endif
}
} // namespace deep_gemm::cdna4::projection
