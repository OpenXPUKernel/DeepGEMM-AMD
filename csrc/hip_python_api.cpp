#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "cdna4/jit_runtime.hpp"
#include "deep_gemm/impls/cdna4/gemm_args.h"

#include <limits>
#include <optional>
#include <tuple>

namespace py = pybind11;

namespace {

using deep_gemm::cdna4::GemmArgs;
using deep_gemm::cdna4::KernelConfig;
using deep_gemm::cdna4::get_best_config;
using deep_gemm::cdna4::jit_runtime;

void check_tensor(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.defined(), name, " must be defined");
    TORCH_CHECK(t.is_cuda(), name, " must be on the active accelerator");
}

int stride_as_int(const torch::Tensor& t, int dim, const char* name) {
    const auto value = t.stride(dim);
    TORCH_CHECK(value >= 0 && value <= std::numeric_limits<int>::max(), name, " stride is out of range");
    return static_cast<int>(value);
}

int current_num_compute_units() {
    int device = 0;
    hipGetDevice(&device);
    hipDeviceProp_t properties{};
    return hipGetDeviceProperties(&properties, device) == hipSuccess ? properties.multiProcessorCount : 256;
}

void cdna4_init(const std::string& root) {
    jit_runtime.init(root);
}

void cdna4_fp8_gemm_nt(const torch::Tensor& a, const torch::Tensor& sfa,
                       const torch::Tensor& b, const torch::Tensor& sfb,
                       const torch::Tensor& d, const std::optional<torch::Tensor>& c,
                       double alpha) {
    check_tensor(a, "a");
    check_tensor(sfa, "sfa");
    check_tensor(b, "b");
    check_tensor(sfb, "sfb");
    check_tensor(d, "d");
    TORCH_CHECK(a.scalar_type() == torch::kFloat8_e4m3fn, "a must be torch.float8_e4m3fn");
    TORCH_CHECK(b.scalar_type() == torch::kFloat8_e4m3fn, "b must be torch.float8_e4m3fn");
    TORCH_CHECK(sfa.scalar_type() == torch::kInt32 && sfb.scalar_type() == torch::kInt32,
                "scale tensors must be torch.int32");
    TORCH_CHECK(d.scalar_type() == torch::kBFloat16 || d.scalar_type() == torch::kFloat,
                "d must be torch.bfloat16 or torch.float32");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && d.dim() == 2, "GEMM tensors must be 2-D");
    TORCH_CHECK(a.size(1) == b.size(1), "a and b must have the same K dimension");
    TORCH_CHECK(d.size(0) == a.size(0) && d.size(1) == b.size(0), "d has an invalid shape");
    TORCH_CHECK(!c.has_value() || (c->device() == d.device() && c->sizes() == d.sizes() &&
                                   c->scalar_type() == d.scalar_type()),
                "c must match d");
    TORCH_CHECK(a.device() == d.device() && b.device() == d.device() &&
                sfa.device() == d.device() && sfb.device() == d.device(),
                "all tensors must be on the same device");
    TORCH_CHECK(a.size(0) <= std::numeric_limits<int>::max() &&
                b.size(0) <= std::numeric_limits<int>::max() &&
                a.size(1) <= std::numeric_limits<int>::max(), "GEMM dimensions are out of range");

    const int m = static_cast<int>(a.size(0));
    const int n = static_cast<int>(b.size(0));
    const int k = static_cast<int>(a.size(1));
    const int out_dtype = d.scalar_type() == torch::kFloat ? MXFP8_OUT_FP32 : MXFP8_OUT_BF16;
    const auto config = get_best_config(m, n, k, out_dtype, c.has_value());

    const int lda = m > 1 ? stride_as_int(a, 0, "a") : (k + 15) / 16 * 16;
    const int ldb = n > 1 ? stride_as_int(b, 0, "b") : (k + 15) / 16 * 16;
    const int lda_sf = stride_as_int(sfa, 1, "sfa");
    const int ldb_sf = stride_as_int(sfb, 1, "sfb");
    const int ldd = d.size(0) > 1 ? stride_as_int(d, 0, "d") : n;

    const auto workspace_size = jit_runtime.workspace_size(config, m, n);
    auto workspace = workspace_size == 0
        ? torch::Tensor()
        : torch::empty({static_cast<long long>(workspace_size)}, d.options().dtype(torch::kFloat));
    GemmArgs args{
        reinterpret_cast<const std::uint8_t*>(a.data_ptr()),
        reinterpret_cast<const std::uint8_t*>(b.data_ptr()),
        reinterpret_cast<const std::int32_t*>(sfa.data_ptr()),
        reinterpret_cast<const std::int32_t*>(sfb.data_ptr()),
        d.data_ptr(), c.has_value() ? c->data_ptr() : nullptr,
        m, n, k, lda, ldb, lda_sf, ldb_sf, ldd,
        static_cast<float>(alpha), config.split_k,
        workspace.defined() ? workspace.data_ptr<float>() : nullptr,
        n > 0 ? (n + 3) / 4 * 4 : 0, nullptr};
    const auto stream = reinterpret_cast<hipStream_t>(
        at::cuda::getCurrentCUDAStream(d.device().index()).stream());
    const int error = jit_runtime.launch(args, config, out_dtype, stream, current_num_compute_units());
    TORCH_CHECK(error == 0, "CDNA4 FP8 GEMM JIT launch failed with code ", error);
}

std::tuple<int, int> cdna4_fp8_gemm_plan(int m, int n, int k) {
    const auto config = get_best_config(m, n, k, MXFP8_OUT_BF16, false);
    const int tiles = (m + config.block_m - 1) / config.block_m * (n + config.block_n - 1) / config.block_n;
    return {tiles, config.split_k};
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
          py::arg("c") = std::nullopt, py::arg("alpha") = 1.0);
    m.def("cdna4_fp8_gemm_plan", &cdna4_fp8_gemm_plan,
          py::arg("m"), py::arg("n"), py::arg("k"));
}
