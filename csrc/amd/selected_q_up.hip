// Fixed-contract Q-up specialization with buffer-bounded row tails.
// Full N tiles and immutable weight-scale blocks retain the original contract.
#include <deep_gemm/impls/cdna4/selected_q_up_args.h>
#include <hip/hip_runtime.h>
#if defined(__HIP_DEVICE_COMPILE__)
#include <deep_gemm/impls/cdna4/selected_q_up_core.cuh>
#endif

namespace {
using namespace deep_gemm::cdna4;
__global__ __launch_bounds__(256) void q_up(SelectedQUpArgs p) {
#if defined(__HIP_DEVICE_COMPILE__)
    p.K = p.lda = p.ldb = 1280;
    using Config = Mfma16x16x128GemmConfig<Mfma16x16x128GemmMode::Persistent, 0, 0, GemmType::Normal, OperandType::Fp8,
                                           OperandType::Fp8, 0, 0, 8, 16, 0, 0, 1, false, true>;
    fp8_gemm_mfma_16x16x128_impl<Config, std::uint16_t, false, 1280, false, true>(p);
#endif
}
} // namespace

extern "C" int qu_launch(int mode, const void *a, const void *b, const int *sa, const int *sb, void *d, int m, int n,
                         int k, int sap, int sbp, int ldd, void *stream) {
    if (mode != 61 || !a || !b || !sa || !sb || !d || m < 1 || m > 1536 || (n != 32768 && n != 36864) || k != 1280 ||
        ldd != n || sap < m || sbp < n || (uint64_t(9) * sap + m) * 4 >= (uint64_t(1) << 32) ||
        (uint64_t(9) * sbp + n) * 4 >= (uint64_t(1) << 32) || reinterpret_cast<uintptr_t>(a) % 16 ||
        reinterpret_cast<uintptr_t>(b) % 16 || reinterpret_cast<uintptr_t>(sa) % 4 ||
        reinterpret_cast<uintptr_t>(sb) % 4 || reinterpret_cast<uintptr_t>(d) % 16)
        return hipErrorInvalidValue;
    int device = 0;
    auto status = hipGetDevice(&device);
    if (status != hipSuccess)
        return status;
    hipDeviceProp_t properties{};
    status = hipGetDeviceProperties(&properties, device);
    if (status != hipSuccess)
        return status;
    SelectedQUpArgs args{};
    args.A = static_cast<const uint8_t *>(a);
    args.B = static_cast<const uint8_t *>(b);
    args.sfa = sa;
    args.sfb = sb;
    args.D = d;
    args.M = m;
    args.N = n;
    args.K = args.lda = args.ldb = k;
    args.lda_sf = sap;
    args.ldb_sf = sbp;
    args.ldd = ldd;
    args.alpha = 1;
    args.splits = 1;
    hipLaunchKernelGGL(q_up, dim3(properties.multiProcessorCount), dim3(256), 152064, static_cast<hipStream_t>(stream),
                       args);
    return hipGetLastError();
}

extern "C" int qu_row_tails_supported() { return 1; }
