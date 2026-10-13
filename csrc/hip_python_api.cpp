#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <c10/hip/HIPGuard.h>
#include <c10/hip/HIPStream.h>

#include "cdna4/jit_runtime.hpp"
#include "cdna4/mqa_logits.hpp"
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

using deep_gemm::cdna4::OperandType;

// mn, k: the logical sizes (an FP4 tensor packs two values per byte along its contiguous dimension)
struct OperandLayout { bool mn_major; int ld; int64_t mn, k; };

const char* operand_type_name(OperandType type) {
    return type == OperandType::Bf16 ? "BF16" : type == OperandType::Fp8 ? "FP8 E4M3" : "packed FP4 (int8)";
}

bool is_packed_fp4(const torch::Tensor& t) {
    return t.scalar_type() == torch::kInt8 || t.scalar_type() == torch::kByte;
}

// Operands with scales are FP8 or packed FP4, operands without scales BF16
OperandType operand_type(const torch::Tensor& t, bool scaled, const char* name) {
    if (!scaled) {
        TORCH_CHECK(t.scalar_type() == torch::kBFloat16, name, " must be BF16 without scales");
        return OperandType::Bf16;
    }
    TORCH_CHECK(t.scalar_type() == torch::kFloat8_e4m3fn || is_packed_fp4(t),
                name, " must be FP8 E4M3 or packed FP4 (int8)");
    return is_packed_fp4(t) ? OperandType::Fp4 : OperandType::Fp8;
}

bool matches(const torch::Tensor& t, OperandType type) {
    return type == OperandType::Fp4 ? is_packed_fp4(t)
         : t.scalar_type() == (type == OperandType::Bf16 ? torch::kBFloat16 : torch::kFloat8_e4m3fn);
}

// Row strides are bytes: the kernels address operands bytewise.
OperandLayout operand_layout(const torch::Tensor& t, const char* name, OperandType type) {
    TORCH_CHECK(matches(t, type), name, " must be ", operand_type_name(type));
    const int64_t es = t.element_size();
    const auto mn = t.size(0), k = t.size(1);
    TORCH_CHECK(t.stride(1) == 1 || t.stride(0) == 1 || std::min(mn, k) <= 1,
                name, " must be K-major or MN-major");
    bool mn_major;
    int64_t ld;
    if (mn == 1 && (t.stride(1) == 1 || k == 1)) {
        mn_major = false;
        ld = (k * es + 3) / 4 * 4;
    } else if (k == 1 && t.stride(0) == 1) {
        mn_major = true;
        ld = (mn * es + 3) / 4 * 4;
    } else {
        mn_major = mn == 1 || (k != 1 && t.stride(1) != 1);
        ld = t.stride(mn_major ? 1 : 0) * es;
    }
    const auto row = (mn_major ? mn : k) * es;
    TORCH_CHECK(ld >= row && ld % 4 == 0 && ld < (1 << 24),
                name, " physical row stride must cover its row, be divisible by four and below 1 << 24");
    TORCH_CHECK(row % 4 == 0 || reinterpret_cast<uintptr_t>(t.data_ptr()) % 4 == 0,
                name, " ragged rows require a four-byte aligned base");
    const int64_t pack = type == OperandType::Fp4 ? 2 : 1;
    return {mn_major, static_cast<int>(ld), mn_major ? mn * pack : mn, mn_major ? k : k * pack};
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

using OptTensor = std::optional<torch::Tensor>;

GemmArgs make_args(const torch::Tensor& a, const OptTensor& sfa, const torch::Tensor& b,
                   const OptTensor& sfb, const torch::Tensor& d, const void* c,
                   int m, int n, int k, int lda, int ldb, int lda_sf, int ldb_sf, int ldd) {
    GemmArgs args{};
    args.A = reinterpret_cast<const std::uint8_t*>(a.data_ptr());
    args.B = reinterpret_cast<const std::uint8_t*>(b.data_ptr());
    args.sfa = sfa.has_value() ? sfa->data_ptr<std::int32_t>() : nullptr;
    args.sfb = sfb.has_value() ? sfb->data_ptr<std::int32_t>() : nullptr;
    args.D = d.data_ptr();
    args.C = c;
    args.M = m, args.N = n, args.K = k;
    args.lda = lda, args.ldb = ldb, args.lda_sf = lda_sf, args.ldb_sf = ldb_sf, args.ldd = ldd;
    args.alpha = 1.f;
    args.splits = 1;
    return args;
}

void cdna4_init(const std::string& root) { jit_runtime.init(root); }

// Logical [MN, K] of an operand [..., MN, K]: FP4 packs two values per byte along the contiguous
// dimension (K, or MN for an MN-major tensor)
std::pair<int64_t, int64_t> logical_mn_k(const torch::Tensor& t, OperandType type) {
    if (type != OperandType::Fp4)
        return {t.size(-2), t.size(-1)};
    const bool mn_major = t.size(-1) > 1 && t.stride(-1) != 1;
    return mn_major ? std::pair<int64_t, int64_t>{2 * t.size(-2), t.size(-1)}
                    : std::pair<int64_t, int64_t>{t.size(-2), 2 * t.size(-1)};
}

// FP8 / packed FP4 operands with packed scales, or (no scales) BF16 operands
void cdna4_fp8_gemm_nt(const torch::Tensor& a, const std::optional<torch::Tensor>& sfa,
                       const torch::Tensor& b, const std::optional<torch::Tensor>& sfb,
                       const torch::Tensor& d, const std::optional<torch::Tensor>& c,
                       double alpha, int gran_k_a, int gran_k_b) {
    const auto device = d.device();
    const bool bf16 = !sfa.has_value();
    TORCH_CHECK(sfa.has_value() == sfb.has_value(), "pass scales for both operands or for none");
    check_tensor(d, device, "d");
    check_tensor(a, device, "a");
    check_tensor(b, device, "b");
    if (!bf16) {
        check_tensor(*sfa, device, "sfa");
        check_tensor(*sfb, device, "sfb");
    }
    const auto a_type = operand_type(a, !bf16, "a"), b_type = operand_type(b, !bf16, "b");
    const auto [m64, k64] = logical_mn_k(a, a_type);
    const auto [n64, kb64] = logical_mn_k(b, b_type);
    TORCH_CHECK(m64 <= std::numeric_limits<int>::max() && n64 <= std::numeric_limits<int>::max() &&
                k64 <= std::numeric_limits<int>::max(), "GEMM dimensions are out of range");
    const int m = m64, n = n64, k = k64;
    TORCH_CHECK(kb64 == k && d.size(0) == m && d.size(1) == n, "GEMM shape mismatch");
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
    const auto lhs = operand_layout(a, "a", a_type), rhs = operand_layout(b, "b", b_type);
    const int lda_sf = bf16 ? 0 : scale_stride(*sfa, m, k, gran_k_a, "sfa");
    const int ldb_sf = bf16 ? 0 : scale_stride(*sfb, n, k, gran_k_b, "sfb");
    const int out_dtype = d.scalar_type() == torch::kFloat ? MXFP8_OUT_FP32 : MXFP8_OUT_BF16;
    const auto config = get_best_config(m, n, deep_gemm::cdna4::planner_k(k, a_type), out_dtype, c.has_value(), ldd,
        reinterpret_cast<uintptr_t>(d.data_ptr()) % 16 == 0,
        int(lhs.mn_major) | (int(rhs.mn_major) << 1),
        bf16 ? 0 : int(gran_k_a == 128) | (int(gran_k_b == 128) << 1), a_type, b_type);

    // Each split writes FP32 [M, align(N,4)]; workspace_size is measured in bytes.
    const auto workspace_size = jit_runtime.workspace_size(config, m, n);
    auto workspace = workspace_size == 0 ? torch::Tensor() :
        torch::empty({static_cast<int64_t>(workspace_size / sizeof(float))}, d.options().dtype(torch::kFloat));
    const void* c_ptr = c.has_value() ? c->data_ptr() : nullptr;
    if (c.has_value() && c_ptr != d.data_ptr() && c->strides() != d.strides()) {
        d.copy_(*c);
        c_ptr = d.data_ptr();
    }
    GemmArgs args = make_args(a, sfa, b, sfb, d, c_ptr, m, n, k, lhs.ld, rhs.ld, lda_sf, ldb_sf, ldd);
    args.alpha = static_cast<float>(alpha);
    args.splits = config.split_k;
    args.ws = workspace.defined() ? workspace.data_ptr<float>() : nullptr;
    args.ldw = ceil_div(n, 4) * 4;
    const auto stream = c10::hip::getCurrentHIPStream(d.get_device()).stream();
    const int error = jit_runtime.launch(args, config, out_dtype, stream, deep_gemm::cdna4::num_compute_units());
    TORCH_CHECK(error == 0, "CDNA4 FP8 GEMM JIT launch failed with code ", error);
}

// B of an m-grouped GEMM, [G, N, K]: K-major or MN-major within each group.
struct GroupedOperandLayout { bool mn_major; int ld; int64_t group_stride; };

GroupedOperandLayout grouped_operand_layout(const torch::Tensor& t, const char* name, OperandType type) {
    TORCH_CHECK(matches(t, type) && t.dim() == 3, name, " must be a 3-D ", operand_type_name(type), " tensor");
    const int64_t es = t.element_size();
    const auto mn = t.size(1), k = t.size(2);
    const bool mn_major = mn > 1 && k > 1 ? t.stride(2) != 1 : (mn == 1 ? false : t.stride(1) == 1 && k > 1);
    TORCH_CHECK(mn_major ? t.stride(1) == 1 || mn == 1 : t.stride(2) == 1 || k == 1,
                name, " must be K-major or MN-major within each group");
    int64_t ld = mn_major ? (k > 1 ? t.stride(2) * es : (mn * es + 3) / 4 * 4)
                          : (mn > 1 ? t.stride(1) * es : (k * es + 3) / 4 * 4);
    const auto row = (mn_major ? mn : k) * es;
    TORCH_CHECK(ld >= row && ld % 4 == 0 && ld < (1 << 24),
                name, " physical row stride must cover its row, be divisible by four and below 1 << 24");
    const auto group_stride = t.stride(0) * es;
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

void launch_or_throw(GemmArgs& args, const deep_gemm::cdna4::KernelConfig& config, int out_dtype,
                     const torch::Tensor& d, int tiles) {
    const auto stream = c10::hip::getCurrentHIPStream(d.get_device()).stream();
    const int error = jit_runtime.launch(args, config, out_dtype, stream, deep_gemm::cdna4::num_compute_units(), tiles);
    TORCH_CHECK(error == 0, "CDNA4 FP8 GEMM JIT launch failed with code ", error);
}

constexpr int kBlockMN = 256;

void check_m_alignment(int alignment, int block_m) {
    TORCH_CHECK(alignment > 0 && alignment % block_m == 0,
                "this CDNA4 grouped GEMM uses ", block_m, "-row tiles: the M alignment of the contiguous layout "
                "must be a multiple of ", block_m, " (set_mk_alignment_for_contiguous_layout)");
}

// Small-M kernel for a grouped GEMM: tiles that fit `rows` rows, at most as tall as the alignment
deep_gemm::cdna4::KernelConfig grouped_small_m_config(int rows, int n, int k, int alignment, int scale_gran_128,
                                                      deep_gemm::cdna4::GemmType type, int groups) {
    auto config = deep_gemm::cdna4::get_small_m_config(rows, n, k, scale_gran_128, type, groups);
    while (config.block_m > 16 && alignment % config.block_m != 0)
        config = deep_gemm::cdna4::get_small_m_config(config.block_m / 2, n, k, scale_gran_128, type, groups);
    return config;
}

// Small-M kernel on a masked / PSUM layout: workgroups per group and column block, enough for the
// expected rows (more rows loop inside the workgroups); returns the launch's workgroups
int set_group_m_slots(GemmArgs& args, const deep_gemm::cdna4::KernelConfig& config, int expected_rows,
                      int max_rows, int groups, int n) {
    args.group_m_slots = std::clamp(ceil_div(std::max(expected_rows, 1), config.block_m), 1,
                                    std::max(1, ceil_div(max_rows, config.block_m)));
    return groups * args.group_m_slots * ceil_div(n, config.block_n);
}

// D[M, N] = A[M, K] @ B[group(m), N, K]^T with per-row groups or PSUM group ends.
void cdna4_m_grouped_fp8_gemm_contiguous(const torch::Tensor& a, const OptTensor& sfa,
                                         const torch::Tensor& b, const OptTensor& sfb,
                                         const torch::Tensor& d, const torch::Tensor& grouped_layout,
                                         int gran_k_a, int gran_k_b, bool use_psum_layout,
                                         bool ensure_zero_padding, int m_alignment, int expected_m) {
    const auto device = d.device();
    const bool bf16 = !sfa.has_value();
    TORCH_CHECK(sfa.has_value() == sfb.has_value(), "pass scales for both operands or for none");
    check_tensor(d, device, "d");
    check_tensor(a, device, "a");
    TORCH_CHECK(b.is_cuda() && b.device() == device && b.dim() == 3, "b must be a 3-D tensor on the output device");
    if (!bf16) {
        check_tensor(*sfa, device, "sfa");
        TORCH_CHECK(sfb->is_cuda() && sfb->device() == device, "sfb must be on the output accelerator device");
    }
    const auto a_type = operand_type(a, !bf16, "a"), b_type = operand_type(b, !bf16, "b");
    const int m = a.size(0), k = logical_mn_k(a, a_type).second, groups = b.size(0);
    const int n = logical_mn_k(b, b_type).first;
    TORCH_CHECK(logical_mn_k(b, b_type).second == k && d.size(0) == m && d.size(1) == n,
                "grouped GEMM shape mismatch");
    TORCH_CHECK(d.scalar_type() == torch::kBFloat16, "m-grouped GEMMs write BF16 outputs");
    TORCH_CHECK(n <= 1 || d.stride(1) == 1, "d must be N-major");
    check_grouped_layout(grouped_layout, use_psum_layout ? groups : m, device);
    TORCH_CHECK(m_alignment > 0, "the M alignment must be positive");
    const c10::hip::HIPGuard device_guard(d.get_device());
    if (m == 0 || n == 0)
        return;
    TORCH_CHECK(k > 0 && groups > 0, "grouped GEMMs need K > 0 and at least one group");
    const auto lhs = operand_layout(a, "a", a_type);
    TORCH_CHECK(!lhs.mn_major || m == 1, "a must be K-major");
    const auto rhs = grouped_operand_layout(b, "b", b_type);
    const int lda_sf = bf16 ? 0 : scale_stride(*sfa, m, k, gran_k_a, "sfa");
    const auto [ldb_sf, sfb_group_stride] =
        bf16 ? std::pair<int, int64_t>{0, 0} : grouped_scale_stride(*sfb, groups, n, k, gran_k_b, "sfb");
    const int kb = deep_gemm::cdna4::planner_k(k, a_type);
    const int ldd = m > 1 ? stride_as_int(d, 0, "d") : n;
    const auto type = use_psum_layout ? deep_gemm::cdna4::GemmType::MGroupedContiguousPsum
                                      : deep_gemm::cdna4::GemmType::MGroupedContiguous;
    const int sf_gran = bf16 ? 0 : int(gran_k_a == 128) | (int(gran_k_b == 128) << 1);
    // Small groups (a PSUM layout with few expected rows), or an alignment the 256-row tiles do not
    // divide, take the small-M kernel; it needs K-major B.
    const bool small_m = !rhs.mn_major &&
                         ((use_psum_layout && deep_gemm::cdna4::small_m_eligible(expected_m, n, kb, 0)) ||
                          m_alignment % kBlockMN != 0);
    auto config = small_m
        ? grouped_small_m_config(expected_m > 0 ? expected_m : 64, n, kb, m_alignment, sf_gran, type, groups)
        : deep_gemm::cdna4::get_m_grouped_config(type, kb, ldd, reinterpret_cast<uintptr_t>(d.data_ptr()) % 16 == 0,
                                                 int(rhs.mn_major) << 1, sf_gran);
    config.a_type = a_type;
    config.b_type = b_type;
    check_m_alignment(m_alignment, config.block_m);
    GemmArgs args = make_args(a, sfa, b, sfb, d, nullptr, m, n, k, lhs.ld, rhs.ld, lda_sf, ldb_sf, ldd);
    args.grouped_layout = grouped_layout.data_ptr<std::int32_t>();
    args.num_groups = groups;
    args.m_alignment = m_alignment;
    args.zero_padding = ensure_zero_padding;
    args.b_group_stride = rhs.group_stride;
    args.sfb_group_stride = sfb_group_stride;
    int tiles = ceil_div(m, config.block_m) * ceil_div(n, config.block_n);
    if (small_m && use_psum_layout)
        tiles = set_group_m_slots(args, config, expected_m > 0 ? expected_m : m / groups, m, groups, n);
    launch_or_throw(args, config, MXFP8_OUT_BF16, d, tiles);
}

// D[g, :masked_m[g]] = A[g, :masked_m[g]] @ B[g]^T
void cdna4_m_grouped_fp8_gemm_masked(const torch::Tensor& a, const OptTensor& sfa,
                                     const torch::Tensor& b, const OptTensor& sfb,
                                     const torch::Tensor& d, const torch::Tensor& masked_m,
                                     int expected_m, int gran_k_a, int gran_k_b) {
    const auto device = d.device();
    const bool bf16 = !sfa.has_value();
    TORCH_CHECK(sfa.has_value() == sfb.has_value(), "pass scales for both operands or for none");
    TORCH_CHECK(a.is_cuda() && a.device() == device && b.is_cuda() && b.device() == device && d.is_cuda() &&
                (bf16 || (sfa->is_cuda() && sfa->device() == device && sfb->is_cuda() && sfb->device() == device)),
                "all tensors must be on the output accelerator device");
    TORCH_CHECK(a.dim() == 3 && b.dim() == 3 && d.dim() == 3, "masked GEMMs take 3-D a, b and d");
    const auto a_type = operand_type(a, !bf16, "a"), b_type = operand_type(b, !bf16, "b");
    const int groups = a.size(0), max_m = a.size(1), k = logical_mn_k(a, a_type).second;
    const int n = logical_mn_k(b, b_type).first;
    TORCH_CHECK(b.size(0) == groups && logical_mn_k(b, b_type).second == k && d.size(0) == groups &&
                d.size(1) == max_m && d.size(2) == n, "masked GEMM shape mismatch");
    TORCH_CHECK(d.scalar_type() == torch::kBFloat16, "masked GEMMs write BF16 outputs");
    TORCH_CHECK(n <= 1 || d.stride(2) == 1, "d must be N-major");
    TORCH_CHECK(expected_m > 0, "expected_m must be positive");
    check_grouped_layout(masked_m, groups, device);
    const c10::hip::HIPGuard device_guard(d.get_device());
    if (groups == 0 || max_m == 0 || n == 0)
        return;
    TORCH_CHECK(k > 0, "masked GEMMs need K > 0");
    const auto lhs = grouped_operand_layout(a, "a", a_type);
    TORCH_CHECK(!lhs.mn_major, "a must be K-major");
    const auto rhs = grouped_operand_layout(b, "b", b_type);
    TORCH_CHECK(!rhs.mn_major, "b must be K-major");
    const auto [lda_sf, sfa_group_stride] =
        bf16 ? std::pair<int, int64_t>{0, 0} : grouped_scale_stride(*sfa, groups, max_m, k, gran_k_a, "sfa");
    const auto [ldb_sf, sfb_group_stride] =
        bf16 ? std::pair<int, int64_t>{0, 0} : grouped_scale_stride(*sfb, groups, n, k, gran_k_b, "sfb");
    const int ldd = max_m > 1 ? stride_as_int(d, 1, "d") : n;
    const int kb = deep_gemm::cdna4::planner_k(k, a_type);
    const int sf_gran = bf16 ? 0 : int(gran_k_a == 128) | (int(gran_k_b == 128) << 1);
    auto config = deep_gemm::cdna4::small_m_eligible(expected_m, n, kb, 0)
        ? deep_gemm::cdna4::get_small_m_config(expected_m, n, kb, sf_gran, deep_gemm::cdna4::GemmType::MGroupedMasked,
                                               groups)
        : deep_gemm::cdna4::get_m_grouped_config(
              deep_gemm::cdna4::GemmType::MGroupedMasked, kb, ldd,
              reinterpret_cast<uintptr_t>(d.data_ptr()) % 16 == 0 && (d.stride(0) * 2) % 16 == 0, 0, sf_gran);
    config.a_type = a_type;
    config.b_type = b_type;
    GemmArgs args = make_args(a, sfa, b, sfb, d, nullptr, max_m, n, k, lhs.ld, rhs.ld, lda_sf, ldb_sf, ldd);
    args.grouped_layout = masked_m.data_ptr<std::int32_t>();
    args.num_groups = groups;
    args.a_group_stride = lhs.group_stride;
    args.b_group_stride = rhs.group_stride;
    args.sfa_group_stride = sfa_group_stride;
    args.sfb_group_stride = sfb_group_stride;
    args.d_group_stride = d.stride(0);
    const int tiles = config.variant == deep_gemm::cdna4::KernelVariant::mfma_16x16x128_small_m
        ? set_group_m_slots(args, config, expected_m, max_m, groups, n)
        : groups * ceil_div(max_m, config.block_m) * ceil_div(n, config.block_n);
    launch_or_throw(args, config, MXFP8_OUT_BF16, d, tiles);
}

// D[g] (+)= A_g^T B_g over the K range of group g. MN-major (TN): A [sum_k, M], B [sum_k, N].
// K-major (NT): per-group [M, k_g] / [N, k_g] blocks stored back to back; FP4: [M, sum_k / 2] /
// [N, sum_k / 2], a group's K range is a column range of every row.
void cdna4_k_grouped_fp8_gemm(const torch::Tensor& a, const OptTensor& sfa,
                              const torch::Tensor& b, const OptTensor& sfb,
                              const torch::Tensor& d, const std::optional<torch::Tensor>& c,
                              const torch::Tensor& grouped_layout, bool use_psum_layout,
                              int k_alignment, int gran_k, bool k_major) {
    const auto device = d.device();
    const bool bf16 = !sfa.has_value();
    TORCH_CHECK(sfa.has_value() == sfb.has_value(), "pass scales for both operands or for none");
    TORCH_CHECK(a.is_cuda() && a.device() == device && b.is_cuda() && b.device() == device && d.is_cuda() &&
                (bf16 || (sfa->is_cuda() && sfa->device() == device && sfb->is_cuda() && sfb->device() == device)),
                "all tensors must be on the output accelerator device");
    const auto a_type = operand_type(a, !bf16, "a"), b_type = operand_type(b, !bf16, "b");
    TORCH_CHECK(a_type == b_type, "K-grouped GEMMs take two operands of one type");
    const bool fp4 = a_type == OperandType::Fp4;
    TORCH_CHECK(!fp4 || (k_major && gran_k == 32), "FP4 K-grouped GEMMs are K-major with 32-wide scale blocks");
    TORCH_CHECK(d.dim() == 3 && d.is_contiguous(), "d must be a contiguous [G, M, N] tensor");
    TORCH_CHECK(d.scalar_type() == torch::kBFloat16 || d.scalar_type() == torch::kFloat, "d must be BF16 or FP32");
    TORCH_CHECK(bf16 || gran_k == 32 || gran_k == 128, "scale granularity must be 32 or 128");
    TORCH_CHECK(k_alignment > 0 && (bf16 || k_alignment % 128 == 0), "the K alignment must be a multiple of 128");
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
    if (fp4) {
        TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && a.size(0) == m && b.size(0) == n && a.size(1) == b.size(1),
                    "FP4 operands must be [M, sum_k / 2] and [N, sum_k / 2]");
        TORCH_CHECK(a.size(1) % 4 == 0, "FP4 K-grouped operands need sum_k divisible by eight");
        sum_k = 2 * a.size(1);
    } else if (k_major) {
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
    const int lda_sf = bf16 ? 0 : k_grouped_scale_stride(*sfa, m, "sfa");
    const int ldb_sf = bf16 ? 0 : k_grouped_scale_stride(*sfb, n, "sfb");
    const int es = int(a.element_size());
    const int out_dtype = d.scalar_type() == torch::kFloat ? MXFP8_OUT_FP32 : MXFP8_OUT_BF16;
    const auto type = use_psum_layout ? deep_gemm::cdna4::GemmType::KGroupedContiguousPsum
                                      : deep_gemm::cdna4::GemmType::KGroupedContiguous;
    auto config = deep_gemm::cdna4::get_k_grouped_config(type, k_major ? 0 : 3, !bf16 && gran_k == 128 ? 3 : 0);
    config.a_type = a_type;
    config.b_type = b_type;
    // (K-major FP8 / BF16 row strides are the group's K, set on the device)
    const int lda = fp4 ? int(a.size(1)) : k_major ? 0 : m * es, ldb = fp4 ? int(b.size(1)) : k_major ? 0 : n * es;
    GemmArgs args = make_args(a, sfa, b, sfb, d, c.has_value() ? d.data_ptr() : nullptr, m, n, 0,
                              lda, ldb, lda_sf, ldb_sf, n);
    args.grouped_layout = grouped_layout.data_ptr<std::int32_t>();
    args.num_groups = groups;
    args.k_alignment = k_alignment;
    args.d_group_stride = int64_t(m) * n;
    launch_or_throw(args, config, out_dtype, d, groups * ceil_div(m, kBlockMN) * ceil_div(n, kBlockMN));
}

// MQA logits of q [num_q, H, D] (packed FP4: D / 2 bytes) against KV rows from `source`
// (MqaLogitsSource) into the preallocated logits. The Python wrappers validate shapes and build the
// per-source tensors (metadata).
void cdna4_mqa_logits(const torch::Tensor& q, const OptTensor& q_sf, const torch::Tensor& kv, const OptTensor& kv_sf,
                      const torch::Tensor& weights, const torch::Tensor& logits, int source,
                      const OptTensor& ks, const OptTensor& ke, const OptTensor& group_ranges,
                      const OptTensor& context_lens, const OptTensor& block_table, const OptTensor& run_starts,
                      int next_n, int page_kv, const OptTensor& sparse_starts, int num_max_sparse_blocks,
                      int sparse_block_kv) {
    using deep_gemm::cdna4::MqaLogitsFormat;
    using deep_gemm::cdna4::MqaLogitsSource;
    const auto src = static_cast<MqaLogitsSource>(source);
    const bool paged = src == MqaLogitsSource::Paged || src == MqaLogitsSource::PagedSparse;
    const bool fp4 = is_packed_fp4(q);
    const auto format = !q_sf.has_value() ? MqaLogitsFormat::Fp8 : fp4 ? MqaLogitsFormat::MxFp4 : MqaLogitsFormat::MxFp8;
    TORCH_CHECK(q.dim() == 3 && q.is_contiguous() && (fp4 || q.scalar_type() == torch::kFloat8_e4m3fn),
                "q must be a contiguous [num_q, H, D] FP8 or packed FP4 tensor");
    TORCH_CHECK(format != MqaLogitsFormat::Fp8 || !fp4, "FP4 Q needs scales");
    const int num_q = q.size(0), num_heads = q.size(1), head_dim = q.size(2) * (fp4 ? 2 : 1);
    TORCH_CHECK(head_dim == 32 || head_dim == 64 || head_dim == 128, "head dims of 32, 64 or 128 values");
    TORCH_CHECK(!fp4 || head_dim >= 64, "FP4 heads hold 64 or 128 values");
    if (q_sf.has_value())
        TORCH_CHECK(q_sf->scalar_type() == torch::kInt32 && q_sf->is_contiguous() && q_sf->numel() == int64_t(num_q) * num_heads,
                    "q_sf must hold one packed int32 per token and head");
    TORCH_CHECK(weights.dim() == 2 && weights.size(0) == num_q && weights.size(1) == num_heads && weights.stride(1) == 1 &&
                weights.scalar_type() == (format == MqaLogitsFormat::Fp8 ? torch::kFloat : torch::kBFloat16),
                "weights must be [num_q, H] with contiguous rows, float with FP8 Q, BF16 with MX Q");
    TORCH_CHECK(logits.dim() == 2 && logits.size(0) == num_q && logits.stride(1) == 1 &&
                logits.scalar_type() == (format == MqaLogitsFormat::Fp8 ? torch::kFloat : torch::kBFloat16),
                "logits must be [num_q, columns] with contiguous rows");
    const c10::hip::HIPGuard device_guard(q.get_device());
    if (num_q == 0 || logits.size(1) == 0)
        return;

    deep_gemm::cdna4::MqaLogitsArgs args{};
    args.q = reinterpret_cast<const std::uint8_t*>(q.data_ptr());
    args.q_sf = q_sf.has_value() ? q_sf->data_ptr<std::int32_t>() : nullptr;
    args.kv = reinterpret_cast<const std::uint8_t*>(kv.data_ptr());
    args.kv_sf = kv_sf.has_value() ? kv_sf->data_ptr() : nullptr;
    args.weights = weights.data_ptr();
    args.ld_weights = stride_as_int(weights, 0, "weights");
    args.logits = logits.data_ptr();
    args.ld_logits = logits.stride(0);
    args.num_q = num_q;
    const auto ptr = [](const OptTensor& t) { return t.has_value() ? t->data_ptr<std::int32_t>() : nullptr; };
    args.ks = ptr(ks), args.ke = ptr(ke), args.group_ranges = ptr(group_ranges);
    args.context_lens = ptr(context_lens), args.block_table = ptr(block_table), args.run_starts = ptr(run_starts);
    args.sparse_starts = ptr(sparse_starts);
    args.next_n = std::max(next_n, 1), args.page_kv = page_kv;
    args.num_max_sparse_blocks = num_max_sparse_blocks, args.sparse_block_kv = sparse_block_kv;
    if (paged) {
        args.page_stride = kv.stride(0) * kv.element_size();
        args.block_table_stride = block_table.has_value() ? stride_as_int(*block_table, 0, "block_table") : 0;
        TORCH_CHECK(page_kv > 0, "pages need rows");
    } else {
        args.num_kv = kv.size(0);
        TORCH_CHECK(args.num_kv > 0 && kv_sf.has_value(), "contiguous KV needs rows and scales");
    }
    const auto plan = deep_gemm::cdna4::plan_mqa_logits(format, src, head_dim, num_heads,
                                                       run_starts.has_value() ? 4 : args.next_n);
    // Paged: a workgroup per token, the first of each group of a request's tokens works
    const int waves = src == MqaLogitsSource::Paged ? num_q : ceil_div(num_q, plan.tokens_per_wave());
    TORCH_CHECK(!group_ranges.has_value() || group_ranges->numel() == 2 * int64_t(waves),
                "the MQA logits metadata does not match the number of heads");
    const auto stream = c10::hip::getCurrentHIPStream(q.get_device()).stream();
    const int error = deep_gemm::cdna4::launch_mqa_logits(args, plan, waves, logits.size(1), stream);
    TORCH_CHECK(error == 0, "CDNA4 MQA logits launch failed with code ", error);
}

// Tokens per wave of the contiguous MQA logits kernel (the group size of its metadata)
int cdna4_mqa_logits_tokens_per_wave(int num_heads) {
    return deep_gemm::cdna4::plan_mqa_logits(deep_gemm::cdna4::MqaLogitsFormat::MxFp8,
                                             deep_gemm::cdna4::MqaLogitsSource::Contiguous, 128, num_heads)
        .tokens_per_wave();
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
          py::arg("m_alignment"), py::arg("expected_m") = 0);
    m.def("cdna4_m_grouped_fp8_gemm_masked", &cdna4_m_grouped_fp8_gemm_masked,
          py::arg("a"), py::arg("sfa"), py::arg("b"), py::arg("sfb"), py::arg("d"), py::arg("masked_m"),
          py::arg("expected_m"), py::arg("gran_k_a"), py::arg("gran_k_b"));
    m.def("cdna4_k_grouped_fp8_gemm", &cdna4_k_grouped_fp8_gemm,
          py::arg("a"), py::arg("sfa"), py::arg("b"), py::arg("sfb"), py::arg("d"), py::arg("c"),
          py::arg("grouped_layout"), py::arg("use_psum_layout"), py::arg("k_alignment"), py::arg("gran_k"),
          py::arg("k_major"));
    m.def("cdna4_mqa_logits", &cdna4_mqa_logits,
          py::arg("q"), py::arg("q_sf"), py::arg("kv"), py::arg("kv_sf"), py::arg("weights"), py::arg("logits"),
          py::arg("source"), py::arg("ks") = std::nullopt, py::arg("ke") = std::nullopt,
          py::arg("group_ranges") = std::nullopt, py::arg("context_lens") = std::nullopt,
          py::arg("block_table") = std::nullopt, py::arg("run_starts") = std::nullopt, py::arg("next_n") = 1,
          py::arg("page_kv") = 0, py::arg("sparse_starts") = std::nullopt, py::arg("num_max_sparse_blocks") = 0,
          py::arg("sparse_block_kv") = 0);
    m.def("cdna4_mqa_logits_tokens_per_wave", &cdna4_mqa_logits_tokens_per_wave, py::arg("num_heads"));
}
