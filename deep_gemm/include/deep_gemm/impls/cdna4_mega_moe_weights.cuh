#pragma once
#include <deep_gemm/impls/cdna4/device_common.h>

// Initialization-only byte permutation. No dequantization or scale changes.
// Logical [E,N,K/2] becomes physical [E,N/256,K/128,256,64].
// The public prepared-weight tag, rather than an environment flag, owns this
// format contract. It must never be inferred from an ordinary tensor's shape.
struct MoePanelPermutationArgs {
    const uint8_t *source;
    uint8_t *destination;
    int64_t chunks;
    int n, k_bytes;
};

__device__ __forceinline__ void moe_permute_weight_panels(MoePanelPermutationArgs p) {
    for (int64_t chunk = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; chunk < p.chunks;
         chunk += int64_t(gridDim.x) * blockDim.x) {
        const int64_t panel = chunk / 1024;
        const int row = (chunk / 4) % 256, column = (chunk % 4) * 16;
        const int k_panel = panel % (p.k_bytes / 64);
        const int n_panel = (panel / (p.k_bytes / 64)) % (p.n / 256);
        const int64_t expert = panel / (int64_t(p.k_bytes / 64) * (p.n / 256));
        const int64_t source = ((expert * p.n + n_panel * 256 + row) * p.k_bytes) + k_panel * 64 + column;
        *reinterpret_cast<v4u *>(p.destination + chunk * 16) = *reinterpret_cast<const v4u *>(p.source + source);
    }
}
