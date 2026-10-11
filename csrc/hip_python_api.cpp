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

// B of an m-grouped GEMM, [G, N, K]: K-major or MN-major within each group.
struct GroupedOperandLayout { bool mn_major; int ld; int64_t group_stride; };

GroupedOperandLayout grouped_operand_layout(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.scalar_type() == torch::kFloat8_e4m3fn && t.dim() == 3, name, " must be a 3-D FP8 E4M3 tensor");
    const auto mn = t.size(1), k = t.size(2);
    const bool mn_major = mn > 1 && k > 1 ? t.stride(2) != 1 : (mn == 1 ? false : t.stride(1) == 1 && k > 1);
    TORCH_CHECK(mn_major ? t.stride(1) == 1 || mn == 1 : t.stride(2) == 1 || k == 1,
                name, " must be K-major or MN-major within each group");
    int64_t ld = mn_major ? (k > 1 ? t.stride(2) : (mn + 3) / 4 * 4) : (mn > 1 ? t.stride(1) : (k + 3) / 4 * 4);
    const auto row = mn_major ? mn : k;
    TORCH_CHECK(ld >= row && ld % 4 == 0 && ld < (1 << 24),
                name, " physical row stride must cover its row, be divisible by four and below 1 << 24");
    const auto group_stride = t.stride(0);
    TORCH_CHECK(row % 4 == 0 || (reinterpret_cast<uintptr_t>(t.data_ptr()) % 4 == 0 && group_stride % 4 == 0),
                name, " ragged rows require four-byte aligned group bases");
    return {mn_major, static_cast<int>(ld), group_stride};
}

// Packed scales of a 3-D operand: [G, rows, ceil(K / (4 gran_k))], MN-major within each group.
std::pair<int, int64_t> grouped_scale_stride(const torch::Tensor& t, int groups, int rows, int k, int gran_k,
                                             const char* name) {
    TORCH_CHECK(gran_k == 32 || gran_k == 128, "scale granularity must be 32 or 128");
    TORCH_CHECK(t.scalar_type() == torch::kInt32 && t.dim() == 3 && t.size(0) == groups && t.size(1) == rows &&
                t.size(2) == ceil_div(k, gran_k * 4), name, " has an invalid packed-scale dtype or shape");
    const auto pitch = t.size(2) > 1 ? t.stride(2) : (int64_t(rows) + 3) / 4 * 4;
    TORCH_CHECK((t.stride(1) == 1 || rows == 1) && pitch >= (int64_t(rows) + 3) / 4 * 4 && pitch % 4 == 0 &&
                pitch <= std::numeric_limits<int>::max(),
                name, " must be MN-major with an aligned column stride covering all rows");
    return {static_cast<int>(pitch), t.stride(0)};
}

// Packed scales of a K-grouped operand: [sum of per-group words, rows], rows contiguous.
int k_grouped_scale_stride(const torch::Tensor& t, int rows, const char* name) {
    TORCH_CHECK(t.scalar_type() == torch::kInt32 && t.dim() == 2 && t.size(1) == rows,
                name, " has an invalid packed-scale dtype or shape");
    const auto pitch = t.size(0) > 1 ? t.stride(0) : (int64_t(rows) + 3) / 4 * 4;
    TORCH_CHECK((t.stride(1) == 1 || rows == 1) && pitch >= rows && pitch % 4 == 0 &&
                pitch <= std::numeric_limits<int>::max(), name, " must have contiguous rows and an aligned pitch");
    return static_cast<int>(pitch);
}

void check_grouped_layout(const torch::Tensor& layout, int64_t numel, const torch::Device& device) {
    TORCH_CHECK(layout.is_cuda() && layout.device() == device, "grouped layout must be on the output device");
    TORCH_CHECK(layout.scalar_type() == torch::kInt32 && layout.is_contiguous() && layout.numel() == numel,
                "grouped layout must be a contiguous int32 tensor with ", numel, " elements");
}

GemmArgs make_args(const torch::Tensor& a, const torch::Tensor& sfa, const torch::Tensor& b,
                   const torch::Tensor& sfb, const torch::Tensor& d, const void* c,
                   int m, int n, int k, int lda, int ldb, int lda_sf, int ldb_sf, int ldd) {
    GemmArgs args{};
    args.A = reinterpret_cast<const std::uint8_t*>(a.data_ptr());
    args.B = reinterpret_cast<const std::uint8_t*>(b.data_ptr());
    args.sfa = sfa.data_ptr<std::int32_t>();
    args.sfb = sfb.data_ptr<std::int32_t>();
    args.D = d.data_ptr();
    args.C = c;
    args.M = m, args.N = n, args.K = k;
    args.lda = lda, args.ldb = ldb, args.lda_sf = lda_sf, args.ldb_sf = ldb_sf, args.ldd = ldd;
    args.alpha = 1.f;
    args.splits = 1;
    return args;
}

void launch_or_throw(GemmArgs& args, const deep_gemm::cdna4::KernelConfig& config, int out_dtype,
                     const torch::Tensor& d, int tiles) {
    const auto stream = c10::hip::getCurrentHIPStream(d.get_device()).stream();
    const int error = jit_runtime.launch(args, config, out_dtype, stream, deep_gemm::cdna4::num_compute_units(), tiles);
    TORCH_CHECK(error == 0, "CDNA4 FP8 GEMM JIT launch failed with code ", error);
}

constexpr int kBlockMN = 256;

void check_m_alignment(int alignment) {
    TORCH_CHECK(alignment > 0 && alignment % kBlockMN == 0,
                "CDNA4 grouped GEMMs use ", kBlockMN, "-row tiles: the M alignment of the contiguous layout must be "
                "a multiple of ", kBlockMN, " (set_mk_alignment_for_contiguous_layout)");
}

// D[M, N] = A[M, K] @ B[group(m), N, K]^T with per-row groups or PSUM group ends.
void cdna4_m_grouped_fp8_gemm_contiguous(const torch::Tensor& a, const torch::Tensor& sfa,
                                         const torch::Tensor& b, const torch::Tensor& sfb,
                                         const torch::Tensor& d, const torch::Tensor& grouped_layout,
                                         int gran_k_a, int gran_k_b, bool use_psum_layout,
                                         bool ensure_zero_padding, int m_alignment) {
    const auto device = d.device();
    check_tensor(d, device, "d");
    check_tensor(a, device, "a");
    check_tensor(sfa, device, "sfa");
    TORCH_CHECK(b.is_cuda() && b.device() == device && sfb.is_cuda() && sfb.device() == device,
                "b and sfb must be on the output accelerator device");
    const int m = a.size(0), k = a.size(1), groups = b.size(0), n = b.size(1);
    TORCH_CHECK(b.size(2) == k && d.size(0) == m && d.size(1) == n, "grouped GEMM shape mismatch");
    TORCH_CHECK(d.scalar_type() == torch::kBFloat16, "m-grouped GEMMs write BF16 outputs");
    TORCH_CHECK(n <= 1 || d.stride(1) == 1, "d must be N-major");
    check_grouped_layout(grouped_layout, use_psum_layout ? groups : m, device);
    check_m_alignment(m_alignment);
    const c10::hip::HIPGuard device_guard(d.get_device());
    if (m == 0 || n == 0)
        return;
    TORCH_CHECK(k > 0 && groups > 0, "grouped GEMMs need K > 0 and at least one group");
    const auto lhs = operand_layout(a, "a");
    TORCH_CHECK(!lhs.mn_major || m == 1, "a must be K-major");
    const auto rhs = grouped_operand_layout(b, "b");
    const int lda_sf = scale_stride(sfa, m, k, gran_k_a, "sfa");
    const auto [ldb_sf, sfb_group_stride] = grouped_scale_stride(sfb, groups, n, k, gran_k_b, "sfb");
    const int ldd = m > 1 ? stride_as_int(d, 0, "d") : n;
    const auto type = use_psum_layout ? deep_gemm::cdna4::GemmType::MGroupedContiguousPsum
                                      : deep_gemm::cdna4::GemmType::MGroupedContiguous;
    const auto config = deep_gemm::cdna4::get_m_grouped_config(
        type, k, ldd, reinterpret_cast<uintptr_t>(d.data_ptr()) % 16 == 0,
        int(rhs.mn_major) << 1, int(gran_k_a == 128) | (int(gran_k_b == 128) << 1));
    GemmArgs args = make_args(a, sfa, b, sfb, d, nullptr, m, n, k, lhs.ld, rhs.ld, lda_sf, ldb_sf, ldd);
    args.grouped_layout = grouped_layout.data_ptr<std::int32_t>();
    args.num_groups = groups;
    args.zero_padding = ensure_zero_padding;
    args.b_group_stride = rhs.group_stride;
    args.sfb_group_stride = sfb_group_stride;
    launch_or_throw(args, config, MXFP8_OUT_BF16, d, ceil_div(m, kBlockMN) * ceil_div(n, kBlockMN));
}

// D[g, :masked_m[g]] = A[g, :masked_m[g]] @ B[g]^T
void cdna4_m_grouped_fp8_gemm_masked(const torch::Tensor& a, const torch::Tensor& sfa,
                                     const torch::Tensor& b, const torch::Tensor& sfb,
                                     const torch::Tensor& d, const torch::Tensor& masked_m,
                                     int expected_m, int gran_k_a, int gran_k_b) {
    const auto device = d.device();
    TORCH_CHECK(a.is_cuda() && a.device() == device && b.is_cuda() && b.device() == device &&
                sfa.is_cuda() && sfa.device() == device && sfb.is_cuda() && sfb.device() == device &&
                d.is_cuda(), "all tensors must be on the output accelerator device");
    TORCH_CHECK(a.dim() == 3 && d.dim() == 3, "masked GEMMs take 3-D a and d");
    const int groups = a.size(0), max_m = a.size(1), k = a.size(2), n = b.size(1);
    TORCH_CHECK(b.size(0) == groups && b.size(2) == k && d.size(0) == groups && d.size(1) == max_m &&
                d.size(2) == n, "masked GEMM shape mismatch");
    TORCH_CHECK(d.scalar_type() == torch::kBFloat16, "masked GEMMs write BF16 outputs");
    TORCH_CHECK(n <= 1 || d.stride(2) == 1, "d must be N-major");
    TORCH_CHECK(expected_m > 0, "expected_m must be positive");
    check_grouped_layout(masked_m, groups, device);
    const c10::hip::HIPGuard device_guard(d.get_device());
    if (groups == 0 || max_m == 0 || n == 0)
        return;
    TORCH_CHECK(k > 0, "masked GEMMs need K > 0");
    const auto lhs = grouped_operand_layout(a, "a");
    TORCH_CHECK(!lhs.mn_major, "a must be K-major");
    const auto rhs = grouped_operand_layout(b, "b");
    TORCH_CHECK(!rhs.mn_major, "b must be K-major");
    const auto [lda_sf, sfa_group_stride] = grouped_scale_stride(sfa, groups, max_m, k, gran_k_a, "sfa");
    const auto [ldb_sf, sfb_group_stride] = grouped_scale_stride(sfb, groups, n, k, gran_k_b, "sfb");
    const int ldd = max_m > 1 ? stride_as_int(d, 1, "d") : n;
    const auto config = deep_gemm::cdna4::get_m_grouped_config(
        deep_gemm::cdna4::GemmType::MGroupedMasked, k, ldd,
        reinterpret_cast<uintptr_t>(d.data_ptr()) % 16 == 0 && (d.stride(0) * 2) % 16 == 0, 0,
        int(gran_k_a == 128) | (int(gran_k_b == 128) << 1));
    GemmArgs args = make_args(a, sfa, b, sfb, d, nullptr, max_m, n, k, lhs.ld, rhs.ld, lda_sf, ldb_sf, ldd);
    args.grouped_layout = masked_m.data_ptr<std::int32_t>();
    args.num_groups = groups;
    args.a_group_stride = lhs.group_stride;
    args.b_group_stride = rhs.group_stride;
    args.sfa_group_stride = sfa_group_stride;
    args.sfb_group_stride = sfb_group_stride;
    args.d_group_stride = d.stride(0);
    launch_or_throw(args, config, MXFP8_OUT_BF16, d, groups * ceil_div(max_m, kBlockMN) * ceil_div(n, kBlockMN));
}

// D[g] (+)= A_g^T B_g over the K range of group g. MN-major (TN): A [sum_k, M], B [sum_k, N].
// K-major (NT): per-group [M, k_g] / [N, k_g] blocks stored back to back.
void cdna4_k_grouped_fp8_gemm(const torch::Tensor& a, const torch::Tensor& sfa,
                              const torch::Tensor& b, const torch::Tensor& sfb,
                              const torch::Tensor& d, const std::optional<torch::Tensor>& c,
                              const torch::Tensor& grouped_layout, bool use_psum_layout,
                              int k_alignment, int gran_k, bool k_major) {
    const auto device = d.device();
    TORCH_CHECK(a.is_cuda() && a.device() == device && b.is_cuda() && b.device() == device &&
                sfa.is_cuda() && sfa.device() == device && sfb.is_cuda() && sfb.device() == device &&
                d.is_cuda(), "all tensors must be on the output accelerator device");
    TORCH_CHECK(a.scalar_type() == torch::kFloat8_e4m3fn && b.scalar_type() == torch::kFloat8_e4m3fn,
                "a and b must be FP8 E4M3");
    TORCH_CHECK(d.dim() == 3 && d.is_contiguous(), "d must be a contiguous [G, M, N] tensor");
    TORCH_CHECK(d.scalar_type() == torch::kBFloat16 || d.scalar_type() == torch::kFloat, "d must be BF16 or FP32");
    TORCH_CHECK(gran_k == 32 || gran_k == 128, "scale granularity must be 32 or 128");
    TORCH_CHECK(k_alignment > 0 && k_alignment % 128 == 0, "the K alignment must be a multiple of 128");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "a and b must be contiguous");
    const int groups = d.size(0), m = d.size(1), n = d.size(2);
    check_grouped_layout(grouped_layout, groups, device);
    if (c.has_value()) {
        TORCH_CHECK(c->sizes() == d.sizes() && c->scalar_type() == d.scalar_type() && c->is_contiguous(),
                    "c must match d");
    }
    const c10::hip::HIPGuard device_guard(d.get_device());
    if (groups == 0 || m == 0 || n == 0)
        return;
    int64_t sum_k;
    if (k_major) {
        TORCH_CHECK(a.numel() % m == 0 && b.numel() % n == 0 && a.numel() / m == b.numel() / n,
                    "K-major operands must hold sum_k x M and sum_k x N values");
        sum_k = a.numel() / m;
    } else {
        TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && a.size(1) == m && b.size(1) == n && a.size(0) == b.size(0),
                    "MN-major operands must be [sum_k, M] and [sum_k, N]");
        TORCH_CHECK(m % 4 == 0 && n % 4 == 0, "MN-major K-grouped operands need M and N divisible by four");
        sum_k = a.size(0);
    }
    if (c.has_value() && c->data_ptr() != d.data_ptr())
        d.copy_(*c);
    if (sum_k == 0) {
        if (!c.has_value())
            d.zero_();
        return;
    }
    const int lda_sf = k_grouped_scale_stride(sfa, m, "sfa");
    const int ldb_sf = k_grouped_scale_stride(sfb, n, "sfb");
    const int out_dtype = d.scalar_type() == torch::kFloat ? MXFP8_OUT_FP32 : MXFP8_OUT_BF16;
    const auto type = use_psum_layout ? deep_gemm::cdna4::GemmType::KGroupedContiguousPsum
                                      : deep_gemm::cdna4::GemmType::KGroupedContiguous;
    const auto config = deep_gemm::cdna4::get_k_grouped_config(type, k_major ? 0 : 3, gran_k == 128 ? 3 : 0);
    GemmArgs args = make_args(a, sfa, b, sfb, d, c.has_value() ? d.data_ptr() : nullptr, m, n, 0,
                              k_major ? 0 : m, k_major ? 0 : n, lda_sf, ldb_sf, n);
    args.grouped_layout = grouped_layout.data_ptr<std::int32_t>();
    args.num_groups = groups;
    args.k_alignment = k_alignment;
    args.d_group_stride = int64_t(m) * n;
    launch_or_throw(args, config, out_dtype, d, groups * ceil_div(m, kBlockMN) * ceil_div(n, kBlockMN));
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
    m.def("cdna4_m_grouped_fp8_gemm_contiguous", &cdna4_m_grouped_fp8_gemm_contiguous,
          py::arg("a"), py::arg("sfa"), py::arg("b"), py::arg("sfb"), py::arg("d"), py::arg("grouped_layout"),
          py::arg("gran_k_a"), py::arg("gran_k_b"), py::arg("use_psum_layout"), py::arg("ensure_zero_padding"),
          py::arg("m_alignment"));
    m.def("cdna4_m_grouped_fp8_gemm_masked", &cdna4_m_grouped_fp8_gemm_masked,
          py::arg("a"), py::arg("sfa"), py::arg("b"), py::arg("sfb"), py::arg("d"), py::arg("masked_m"),
          py::arg("expected_m"), py::arg("gran_k_a"), py::arg("gran_k_b"));
    m.def("cdna4_k_grouped_fp8_gemm", &cdna4_k_grouped_fp8_gemm,
          py::arg("a"), py::arg("sfa"), py::arg("b"), py::arg("sfb"), py::arg("d"), py::arg("c"),
          py::arg("grouped_layout"), py::arg("use_psum_layout"), py::arg("k_alignment"), py::arg("gran_k"),
          py::arg("k_major"));
}
