// Shape-guarded cache-domain mHC. Never use its publication protocol across domains.
#include <deep_gemm/impls/cdna4/mega_mhc_cache_local.cuh>

namespace {
using deep_gemm::cdna4::MegaMhcArgs;
constexpr int kControl = 1597696;
constexpr int kSharedBytes = 65536;
struct Aux {
    int *counters;
    const uint16_t *fn;
    unsigned long long *clocks;
    int domains;
    int grid_multiplier;
    int split_override;
};

__global__ __launch_bounds__(256) void mhc(MegaMhcArgs args, int *counters, const uint16_t *fn, int domains,
                                           unsigned long long *clocks) {
    deep_gemm::cdna4::mhc_cache_local::run<5120, false, false, false, false, false, false, true, true, false, 64, false,
                                           -1, true, 10, false, false, false, true, true>(args, counters, fn, domains,
                                                                                          clocks);
}

__global__ __launch_bounds__(256) void domain_probe(int *output) {
    if (threadIdx.x == 0) {
        output[blockIdx.x * 2] = __builtin_amdgcn_s_getreg(20 | (31 << 11));
        output[blockIdx.x * 2 + 1] = __builtin_amdgcn_s_getreg(4 | (31 << 11));
    }
}
} // namespace

extern "C" size_t args_size() { return sizeof(MegaMhcArgs); }

extern "C" int cache_domains(int *output, void *stream, int blocks) {
    if (!output || blocks < 1)
        return hipErrorInvalidValue;
    hipLaunchKernelGGL(domain_probe, dim3(blocks), dim3(256), kSharedBytes, static_cast<hipStream_t>(stream), output);
    return hipGetLastError();
}

extern "C" int cache_occupancy(int control, int shared_bytes, int *active) {
    if (control != kControl || shared_bytes != kSharedBytes || !active)
        return hipErrorInvalidValue;
    return hipOccupancyMaxActiveBlocksPerMultiprocessor(active, mhc, 256, shared_bytes);
}

extern "C" int launch(const MegaMhcArgs *args, void *stream, int block_mode, int control, const uint16_t *opaque_aux) {
    if (!args || block_mode != 32768 || control != kControl || args->hidden != 5120 || !args->shifted_prev_mix ||
        args->splits != 10 || args->num_tokens < 0)
        return hipErrorInvalidValue;
    if (!args->num_tokens)
        return hipSuccess;
    auto aux = reinterpret_cast<const Aux *>(opaque_aux);
    if (!aux || !aux->counters || !aux->fn || aux->domains < 1 || aux->domains > 32 || aux->grid_multiplier != 1 ||
        aux->split_override != 10)
        return hipErrorInvalidValue;
    int device = 0, active = 0;
    auto status = hipGetDevice(&device);
    if (status != hipSuccess)
        return status;
    hipDeviceProp_t properties{};
    status = hipGetDeviceProperties(&properties, device);
    if (status != hipSuccess)
        return status;
    status = hipOccupancyMaxActiveBlocksPerMultiprocessor(&active, mhc, 256, kSharedBytes);
    if (status != hipSuccess || active < 1)
        return hipErrorLaunchOutOfResources;
    hipLaunchKernelGGL(mhc, dim3(properties.multiProcessorCount), dim3(256), kSharedBytes,
                       static_cast<hipStream_t>(stream), *args, aux->counters, aux->fn, aux->domains, aux->clocks);
    return hipGetLastError();
}
