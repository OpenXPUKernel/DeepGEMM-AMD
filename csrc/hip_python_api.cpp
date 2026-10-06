#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <c10/hip/HIPGuard.h>
#include <c10/hip/HIPStream.h>

#include "cdna4/jit_runtime.hpp"
#include "deep_gemm/impls/cdna4/gemm_args.h"

#include <limits>
#include <optional>
#include <tuple>

namespace py = pybind11;

namespace {
using deep_gemm::cdna4::GemmArgs;
using deep_gemm::cdna4::ceil_div;
using deep_gemm::cdna4::get_best_config;
using deep_gemm::cdna4::jit_runtime;

void check_tensor(const torch::Tensor& t, const torch::Device& device, const char* name) {
    TORCH_CHECK(t.defined() && t.is_cuda() && t.device() == device,
                name, " must be on the output accelerator device");
    TORCH_CHECK(t.dim() == 2, name, " must be 2-D");
}

int stride_as_int(const torch::Tensor& t, int dim, const char* name) {
    const auto value = t.stride(dim);
    TORCH_CHECK(value >= 0 && value <= std::numeric_limits<int>::max(), name, " stride is out of range");
    return static_cast<int>(value);
}

struct OperandLayout { bool mn_major; int ld; };

OperandLayout operand_layout(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.scalar_type() == torch::kFloat8_e4m3fn, name, " must be FP8 E4M3");
    const auto mn = t.size(0), k = t.size(1);
    TORCH_CHECK(t.stride(1) == 1 || t.stride(0) == 1 || std::min(mn, k) <= 1,
                name, " must be K-major or MN-major");
    bool mn_major;
    int64_t ld;
    if (mn == 1 && (t.stride(1) == 1 || k == 1)) {
        mn_major = false;
        ld = (k + 3) / 4 * 4;
    } else if (k == 1 && t.stride(0) == 1) {
        mn_major = true;
        ld = (mn + 3) / 4 * 4;
    } else {
        mn_major = mn == 1 || (k != 1 && t.stride(1) != 1);
        ld = t.stride(mn_major ? 1 : 0);
    }
    const auto row = mn_major ? mn : k;
    TORCH_CHECK(ld >= row && ld % 4 == 0 && ld < (1 << 24),
                name, " physical row stride must cover its row, be divisible by four and below 1 << 24");
    TORCH_CHECK(row % 4 == 0 || reinterpret_cast<uintptr_t>(t.data_ptr()) % 4 == 0,
                name, " ragged rows require a four-byte aligned base");
    return {mn_major, static_cast<int>(ld)};
}

int scale_stride(const torch::Tensor& t, int rows, int k, int gran_k, const char* name) {
    TORCH_CHECK(gran_k == 32 || gran_k == 128, "scale granularity must be 32 or 128");
    TORCH_CHECK(t.scalar_type() == torch::kInt32 && t.size(0) == rows &&
                t.size(1) == ceil_div(k, gran_k * 4), name, " has an invalid packed-scale dtype or shape");
    const int pitch = stride_as_int(t, 1, name);
    TORCH_CHECK((t.stride(0) == 1 || rows == 1) && pitch >= (int64_t(rows) + 3) / 4 * 4 && pitch % 4 == 0,
                name, " must be MN-major with an aligned column stride covering all rows");
    return pitch;
}

void cdna4_init(const std::string& root) { jit_runtime.init(root); }

void cdna4_fp8_gemm_nt(const torch::Tensor& a, const torch::Tensor& sfa,
                       const torch::Tensor& b, const torch::Tensor& sfb,
                       const torch::Tensor& d, const std::optional<torch::Tensor>& c,
                       double alpha, int gran_k_a, int gran_k_b) {
    const auto device = d.device();
    check_tensor(d, device, "d");
    check_tensor(a, device, "a");
    check_tensor(b, device, "b");
    check_tensor(sfa, device, "sfa");
    check_tensor(sfb, device, "sfb");
    TORCH_CHECK(a.size(0) <= std::numeric_limits<int>::max() &&
                b.size(0) <= std::numeric_limits<int>::max() &&
                a.size(1) <= std::numeric_limits<int>::max(), "GEMM dimensions are out of range");
    const int m = a.size(0), n = b.size(0), k = a.size(1);
    TORCH_CHECK(b.size(1) == k && d.size(0) == m && d.size(1) == n, "GEMM shape mismatch");
    TORCH_CHECK(d.scalar_type() == torch::kBFloat16 || d.scalar_type() == torch::kFloat,
                "d must be BF16 or FP32");
    TORCH_CHECK(n <= 1 || d.stride(1) == 1, "d must be N-major");
    const int ldd = m > 1 ? stride_as_int(d, 0, "d") : n;
    TORCH_CHECK(m <= 1 || ldd >= n, "d rows must not overlap");
    if (c.has_value()) {
        check_tensor(*c, device, "c");
        TORCH_CHECK(c->sizes() == d.sizes() && c->scalar_type() == d.scalar_type(), "c must match d");
        TORCH_CHECK(n <= 1 || c->stride(1) == 1, "c must be N-major");
        TORCH_CHECK(m <= 1 || c->stride(0) >= n, "c rows must not overlap");
        TORCH_CHECK(c->data_ptr() != d.data_ptr() || c->strides() == d.strides(),
                    "c aliases d with different strides");
    }
    const c10::hip::HIPGuard device_guard(d.get_device());
    if (m == 0 || n == 0)
        return;
    if (k == 0) {
        if (c.has_value()) d.copy_(*c);
        else d.zero_();
        return;
    }
    const auto lhs = operand_layout(a, "a"), rhs = operand_layout(b, "b");
    const int lda_sf = scale_stride(sfa, m, k, gran_k_a, "sfa");
    const int ldb_sf = scale_stride(sfb, n, k, gran_k_b, "sfb");
    const int out_dtype = d.scalar_type() == torch::kFloat ? MXFP8_OUT_FP32 : MXFP8_OUT_BF16;
    const auto config = get_best_config(m, n, k, out_dtype, c.has_value(), ldd,
        reinterpret_cast<uintptr_t>(d.data_ptr()) % 16 == 0,
        int(lhs.mn_major) | (int(rhs.mn_major) << 1), int(gran_k_a == 128) | (int(gran_k_b == 128) << 1));

    // Each split writes FP32 [M, align(N,4)]; workspace_size is measured in bytes.
    const auto workspace_size = jit_runtime.workspace_size(config, m, n);
    auto workspace = workspace_size == 0 ? torch::Tensor() :
        torch::empty({static_cast<int64_t>(workspace_size / sizeof(float))}, d.options().dtype(torch::kFloat));
    const void* c_ptr = c.has_value() ? c->data_ptr() : nullptr;
    if (c.has_value() && c_ptr != d.data_ptr() && c->strides() != d.strides()) {
        d.copy_(*c);
        c_ptr = d.data_ptr();
    }
    GemmArgs args{
        reinterpret_cast<const std::uint8_t*>(a.data_ptr()), reinterpret_cast<const std::uint8_t*>(b.data_ptr()),
        sfa.data_ptr<std::int32_t>(), sfb.data_ptr<std::int32_t>(),
        d.data_ptr(), c_ptr, m, n, k, lhs.ld, rhs.ld, lda_sf, ldb_sf, ldd,
        static_cast<float>(alpha), config.split_k,
        workspace.defined() ? workspace.data_ptr<float>() : nullptr, ceil_div(n, 4) * 4};
    const auto stream = c10::hip::getCurrentHIPStream(d.get_device()).stream();
    const int error = jit_runtime.launch(args, config, out_dtype, stream, deep_gemm::cdna4::num_compute_units());
    TORCH_CHECK(error == 0, "CDNA4 FP8 GEMM JIT launch failed with code ", error);
}

std::tuple<int64_t, int> cdna4_fp8_gemm_plan(int m, int n, int k) {
    TORCH_CHECK(m >= 0 && n >= 0 && k >= 0, "GEMM dimensions must be nonnegative");
    if (m == 0 || n == 0 || k == 0) return {0, 0};
    const auto tiles = int64_t(ceil_div(m, 256)) * ceil_div(n, 256);
    return {tiles, deep_gemm::cdna4::plan_splits(m, n, k, deep_gemm::cdna4::num_compute_units())};
}

}  // namespace

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME _C
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "DeepGEMM HIP extension";
    m.def("cdna4_init", &cdna4_init, py::arg("root"));
    m.def("cdna4_fp8_gemm_nt", &cdna4_fp8_gemm_nt,
          py::arg("a"), py::arg("sfa"), py::arg("b"), py::arg("sfb"), py::arg("d"),
          py::arg("c") = std::nullopt, py::arg("alpha") = 1.0,
          py::arg("gran_k_a") = 32, py::arg("gran_k_b") = 32);
    m.def("cdna4_fp8_gemm_plan", &cdna4_fp8_gemm_plan, py::arg("m"), py::arg("n"), py::arg("k"));
}
