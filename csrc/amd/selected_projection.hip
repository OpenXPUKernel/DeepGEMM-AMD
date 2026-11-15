// Native projection entrypoints for the shape-guarded model adapter.
#include <deep_gemm/impls/cdna4/projection_gemm_m32.cuh>
#include <deep_gemm/impls/cdna4/projection_pack.cuh>

namespace {
template <bool BF16, int STRIPES>
__global__ __launch_bounds__(256) void projection(const void *a, const void *b, const int *sa, const int *sb, void *d,
                                                  int m, int n, int k, int sap, int sbp, int ldd) {
    deep_gemm::cdna4::projection::run_m32<BF16, BF16 ? 2 : 1, 1, 4, true, 1, false, 2, STRIPES, 2, BF16 ? 0 : 256, 0,
                                          !BF16, 1, true, !BF16, false, 4, false, BF16 ? 40 : 1, false, true, false, 1,
                                          false, false, BF16 ? 2 : 0, BF16 ? 4 : 0, 0, -1, !BF16, !BF16, false, true,
                                          false, false, false, false, true>(a, b, sa, sb, d, m, n, k, sap, sbp, ldd);
}

__global__ void pack_fp8(const void *source, void *destination, int n, int k) {
    deep_gemm::cdna4::projection::pack_weights<false, 1>(source, destination, n, k);
}
} // namespace

extern "C" int launch(int bf16, int mode, const void *a, const void *b, const int *sa, const int *sb, void *d, int m,
                      int n, int k, int sap, int sbp, int ldd, void *stream) {
    if (!a || !b || !d || m != 1536 || k != 5120 || ldd != n ||
        (bf16 ? (bf16 != 1 || mode != 104 || n != 1056)
              : (mode != 57 || n != 1792 || !sa || !sb || sap < m || sbp < n)))
        return hipErrorInvalidValue;
    const void *kernel = bf16 ? reinterpret_cast<const void *>(projection<true, 4>)
                              : reinterpret_cast<const void *>(projection<false, 1>);
    void *args[] = {&a, &b, &sa, &sb, &d, &m, &n, &k, &sap, &sbp, &ldd};
    return hipLaunchKernel(kernel, dim3((n + 63) / 64, 24), dim3(256), args, 0, static_cast<hipStream_t>(stream));
}

extern "C" int pack(int bf16, const void *source, void *destination, int n, int k, void *stream) {
    if (bf16 || !source || !destination || n != 1792 || k != 5120 || reinterpret_cast<uintptr_t>(source) % 16 ||
        reinterpret_cast<uintptr_t>(destination) % 16)
        return hipErrorInvalidValue;
    hipLaunchKernelGGL(pack_fp8, dim3((n * k / 16 + 255) / 256), dim3(256), 0, static_cast<hipStream_t>(stream), source,
                       destination, n, k);
    return hipGetLastError();
}
