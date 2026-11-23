#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#include <c10/hip/HIPGuard.h>
#include <c10/hip/HIPStream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <uni_symm/backend/hip/backend.hpp>
#include <uni_symm/python_api.hpp>

#include "cdna4/decode_batch.hpp"
#include "cdna4/jit_runtime.hpp"
#include "cdna4/mqa_logits.hpp"
#include "deep_gemm/impls/cdna4/gemm_args.h"
#include "deep_gemm/impls/cdna4/mega_gate_args.h"
#include "deep_gemm/layout/cdna4_mega_moe.cuh"

#include <array>
#include <limits>
#include <map>
#include <optional>
#include <tuple>

namespace py = pybind11;

namespace {
using deep_gemm::cdna4::ceil_div;
using deep_gemm::cdna4::GemmArgs;
using deep_gemm::cdna4::get_best_config;
using deep_gemm::cdna4::jit_runtime;

void check_tensor(const torch::Tensor &t, const torch::Device &device, const char *name) {
    TORCH_CHECK(t.defined() && t.is_cuda() && t.device() == device, name, " must be on the output accelerator device");
    TORCH_CHECK(t.dim() == 2, name, " must be 2-D");
}

int stride_as_int(const torch::Tensor &t, int dim, const char *name) {
    const auto value = t.stride(dim);
    TORCH_CHECK(value >= 0 && value <= std::numeric_limits<int>::max(), name, " stride is out of range");
    return static_cast<int>(value);
}

using deep_gemm::cdna4::OperandType;

// mn, k: the logical sizes (an FP4 tensor packs two values per byte along its contiguous dimension)
struct OperandLayout {
    bool mn_major;
    int ld;
    int64_t mn, k;
};

const char *operand_type_name(OperandType type) {
    return type == OperandType::Bf16 ? "BF16" : type == OperandType::Fp8 ? "FP8 E4M3" : "packed FP4 (int8)";
}

bool is_packed_fp4(const torch::Tensor &t) {
    return t.scalar_type() == torch::kInt8 || t.scalar_type() == torch::kByte;
}

// Operands with scales are FP8 or packed FP4, operands without scales BF16
OperandType operand_type(const torch::Tensor &t, bool scaled, const char *name) {
    if (!scaled) {
        TORCH_CHECK(t.scalar_type() == torch::kBFloat16, name, " must be BF16 without scales");
        return OperandType::Bf16;
    }
    TORCH_CHECK(t.scalar_type() == torch::kFloat8_e4m3fn || is_packed_fp4(t), name,
                " must be FP8 E4M3 or packed FP4 (int8)");
    return is_packed_fp4(t) ? OperandType::Fp4 : OperandType::Fp8;
}

bool matches(const torch::Tensor &t, OperandType type) {
    return type == OperandType::Fp4
               ? is_packed_fp4(t)
               : t.scalar_type() == (type == OperandType::Bf16 ? torch::kBFloat16 : torch::kFloat8_e4m3fn);
}

// Row strides are bytes: the kernels address operands bytewise.
OperandLayout operand_layout(const torch::Tensor &t, const char *name, OperandType type) {
    TORCH_CHECK(matches(t, type), name, " must be ", operand_type_name(type));
    const int64_t es = t.element_size();
    const auto mn = t.size(0), k = t.size(1);
    TORCH_CHECK(t.stride(1) == 1 || t.stride(0) == 1 || std::min(mn, k) <= 1, name, " must be K-major or MN-major");
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
    TORCH_CHECK(ld >= row && ld % 4 == 0 && ld < (1 << 24), name,
                " physical row stride must cover its row, be divisible by four and below 1 << 24");
    TORCH_CHECK(row % 4 == 0 || reinterpret_cast<uintptr_t>(t.data_ptr()) % 4 == 0, name,
                " ragged rows require a four-byte aligned base");
    const int64_t pack = type == OperandType::Fp4 ? 2 : 1;
    return {mn_major, static_cast<int>(ld), mn_major ? mn * pack : mn, mn_major ? k : k * pack};
}

int scale_stride(const torch::Tensor &t, int rows, int k, int gran_k, const char *name) {
    TORCH_CHECK(gran_k == 32 || gran_k == 128, "scale granularity must be 32 or 128");
    TORCH_CHECK(t.scalar_type() == torch::kInt32 && t.size(0) == rows && t.size(1) == ceil_div(k, gran_k * 4), name,
                " has an invalid packed-scale dtype or shape");
    const int pitch = stride_as_int(t, 1, name);
    TORCH_CHECK((t.stride(0) == 1 || rows == 1) && pitch >= (int64_t(rows) + 3) / 4 * 4 && pitch % 4 == 0, name,
                " must be MN-major with an aligned column stride covering all rows");
    return pitch;
}

using OptTensor = std::optional<torch::Tensor>;

GemmArgs make_args(const torch::Tensor &a, const OptTensor &sfa, const torch::Tensor &b, const OptTensor &sfb,
                   const torch::Tensor &d, const void *c, int m, int n, int k, int lda, int ldb, int lda_sf, int ldb_sf,
                   int ldd) {
    GemmArgs args{};
    args.A = reinterpret_cast<const std::uint8_t *>(a.data_ptr());
    args.B = reinterpret_cast<const std::uint8_t *>(b.data_ptr());
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

void cdna4_init(const std::string &root) { jit_runtime.init(root); }

// Logical [MN, K] of an operand [..., MN, K]: FP4 packs two values per byte along the contiguous
// dimension (K, or MN for an MN-major tensor)
std::pair<int64_t, int64_t> logical_mn_k(const torch::Tensor &t, OperandType type) {
    if (type != OperandType::Fp4)
        return {t.size(-2), t.size(-1)};
    const bool mn_major = t.size(-1) > 1 && t.stride(-1) != 1;
    return mn_major ? std::pair<int64_t, int64_t>{2 * t.size(-2), t.size(-1)}
                    : std::pair<int64_t, int64_t>{t.size(-2), 2 * t.size(-1)};
}

// FP8 / packed FP4 operands with packed scales, or (no scales) BF16 operands
void launch_compact_wv(GemmArgs &args, const char *shape, int groups, int output_splits, hipStream_t stream, const torch::Tensor &original,
    const OptTensor &panels, const OptTensor &panel_sf) {
    const std::string tile(shape);
    TORCH_CHECK(tile == "64x128" || tile == "128x128" || tile == "128x256" || tile == "32x128" || tile == "64x64" || tile == "64x256",
        "unsupported packed WV tile");
    const int bm = std::stoi(tile.substr(0, tile.find('x')));
    const int bn = std::stoi(tile.substr(tile.find('x') + 1));
    const char *value = output_splits && args.K == 5120 ? std::getenv("DG_CDNA4_QSWA_STAGES") : nullptr;
    if (!value && !output_splits)
        value = std::getenv("DG_CDNA4_WV_PANEL_STAGES");
    if (!value && output_splits && args.K == 8192)
        value = std::getenv("DG_CDNA4_OUT_PANEL_STAGES");
    if (!value)
        value = std::getenv("DG_CDNA4_COMPACT_STAGES");
    TORCH_CHECK(!value || std::string(value) == "2" || std::string(value) == "3", "compact stages must be 2 or 3");
    const int stages = value && std::string(value) == "3" ? 3 : 2;
    std::ostringstream source;
    const char *flow = std::getenv(!output_splits   ? "DG_CDNA4_WV_FRAGMENT_PIPE"
                                   : args.K == 5120 ? "DG_CDNA4_QSWA_FRAGMENT_PIPE"
                                                    : "DG_CDNA4_OUT_FRAGMENT_PIPE");
    if (flow && (std::string(flow) == "1" || std::string(flow) == "2"))
        source << "#define CDNA4_DENSE_FRAGMENT_PIPE " << flow << "\n";
    TORCH_CHECK(!panel_sf || panels, "compact scales require an explicitly packed weight");
    if (panels) {
        TORCH_CHECK(panels->device() == original.device() && panels->scalar_type() == torch::kFloat8_e4m3fn && panels->is_contiguous() &&
                        panels->dim() == 5 && panels->size(0) == groups && panels->size(1) == args.N / 32 &&
                        panels->size(2) == args.K / 128 && panels->size(3) == 32 && panels->size(4) == 128,
            "dense panels must be contiguous FP8 [groups,N/32,K/128,32,128]");
        args.B = reinterpret_cast<const uint8_t *>(panels->data_ptr());
        args.b_group_stride = int64_t(args.N) * args.K;
        args.ldb = args.K;
        source << "#define CDNA4_DENSE_PANELS 1\n";
        if (panel_sf) {
            TORCH_CHECK(panel_sf->device() == original.device() && panel_sf->scalar_type() == torch::kInt32 && panel_sf->is_contiguous() &&
                            panel_sf->dim() == 3 && panel_sf->size(0) == groups && panel_sf->size(1) == args.N / 32 &&
                            panel_sf->size(2) == args.K / 128,
                "invalid compact panel scales");
            args.sfb = panel_sf->data_ptr<int32_t>();
            args.sfb_group_stride = int64_t(args.N / 32) * (args.K / 128);
            args.ldb_sf = args.K / 128;
            source << "#define CDNA4_DENSE_GROUP_SCALE 1\n";
        }
    }
    int coop = 0, coop_stages = 0;
    unsigned lds = stages * ((bm + bn) * 128 + (panel_sf ? 1024 : 2048));
    if (const char *requested = std::getenv(args.K == 5120 ? "DG_CDNA4_QSWA_SPLIT_WAVES" : "DG_CDNA4_OUT_SPLIT_WAVES");
        output_splits > 1 && groups == 1 && requested && std::string(requested) != "0") {
        char tail = 0;
        TORCH_CHECK(std::sscanf(requested, "%d,%d%c", &coop, &coop_stages, &tail) == 2 && coop >= 2 && coop <= 4 &&
                        (coop_stages == 1 || coop_stages == 2),
            "invalid cooperative split shape");
        const int kt = args.K / 128, span = ceil_div(kt, output_splits);
        coop = std::min(coop, ceil_div(kt, span));
        int bytes = (bm + bn) * 128 + ceil_div(bm, 64) * 256 + (panel_sf ? 0 : ceil_div(bn, 64) * 256);
        bytes = ceil_div(bytes, 1024) * 1024;
        const int required = std::max(coop * coop_stages * bytes, (coop - 1) * bm * bn * 4);
        if (coop >= 2 && required <= 163840)
            lds = required;
        else
            coop = 0;
    }
    if (coop) {
        source << "#include <deep_gemm/impls/cdna4/cooperative_dense_splits.cuh>\n"
                  "extern \"C\" __global__ __launch_bounds__("
               << coop * 256
               << ") void dg_packed_wv(deep_gemm::cdna4::GemmArgs p){"
                  "cooperative_dense_splits<"
               << bm << "," << bn << "," << output_splits << "," << coop << "," << coop_stages << "," << args.K << "," << args.N
               << ">(p);}";
    } else
        source << "#include <deep_gemm/impls/cdna4/packed_wv.cuh>\n"
                  "extern \"C\" __global__ __launch_bounds__(256) void dg_packed_wv(deep_gemm::cdna4::GemmArgs p){"
                  "packed_wv_tile<"
               << bm << "," << bn << "," << output_splits << "," << stages << "," << args.K << "," << args.N << ">(p);}";
    const auto text = source.str();
    const auto module = jit_runtime.get_module("compact_wv:" + text, [&] { return text; }, {"dg_packed_wv"});

    const int error = deep_gemm::cdna4::JitRuntime::launch_kernel(
        module->kernels[0], args, dim3(args.N / bn, ceil_div(args.M, bm), groups), dim3(256 * std::max(1, coop)), lds, stream);
    TORCH_CHECK(error == 0, "packed WV tile launch failed: ", error);
}

void cdna4_fp8_gemm_nt(const torch::Tensor &a, const std::optional<torch::Tensor> &sfa, const torch::Tensor &b,
    const std::optional<torch::Tensor> &sfb, const torch::Tensor &d, const std::optional<torch::Tensor> &c, double alpha, int gran_k_a,
    int gran_k_b, bool reference_plan = false, const OptTensor &panels = std::nullopt, const OptTensor &panel_sf = std::nullopt) {
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
                    k64 <= std::numeric_limits<int>::max(),
                "GEMM dimensions are out of range");
    const int m = m64, n = n64, k = k64;
    TORCH_CHECK(kb64 == k && d.size(0) == m && d.size(1) == n, "GEMM shape mismatch");
    TORCH_CHECK(d.scalar_type() == torch::kBFloat16 || d.scalar_type() == torch::kFloat, "d must be BF16 or FP32");
    TORCH_CHECK(n <= 1 || d.stride(1) == 1, "d must be N-major");
    const int ldd = m > 1 ? stride_as_int(d, 0, "d") : n;
    TORCH_CHECK(m <= 1 || ldd >= n, "d rows must not overlap");
    if (c.has_value()) {
        check_tensor(*c, device, "c");
        TORCH_CHECK(c->sizes() == d.sizes() && c->scalar_type() == d.scalar_type(), "c must match d");
        TORCH_CHECK(n <= 1 || c->stride(1) == 1, "c must be N-major");
        TORCH_CHECK(m <= 1 || c->stride(0) >= n, "c rows must not overlap");
        TORCH_CHECK(c->data_ptr() != d.data_ptr() || c->strides() == d.strides(), "c aliases d with different strides");
    }
    const c10::hip::HIPGuardMasqueradingAsCUDA device_guard(d.get_device());
    if (m == 0 || n == 0)
        return;
    if (k == 0) {
        if (c.has_value())
            d.copy_(*c);
        else
            d.zero_();
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

    const bool decode_out = n == 5120 && k == 8192;
    const bool decode_qswa = n == 1792 && k == 5120;
    if (!reference_plan && (decode_out || decode_qswa) && (deep_gemm::cdna4::decode_batch_rows(m, 4) || deep_gemm::cdna4::decode_batch_rows(m, 5)) && a_type == OperandType::Fp8 &&
        b_type == OperandType::Fp8 && !lhs.mn_major && !rhs.mn_major && !c && alpha == 1.0 && gran_k_a == 32 && gran_k_b == 32 &&
        out_dtype == MXFP8_OUT_BF16) {
        const char *tile = std::getenv(decode_out ? "DG_CDNA4_LOCAL_OUT_TILE" : "DG_CDNA4_LOCAL_QSWA_TILE");
        if (decode_out && deep_gemm::cdna4::decode_batch_rows(m, 4)) {
            if (const char *draft_tile = std::getenv("DG_CDNA4_DRAFT_OUT_TILE"))
                tile = draft_tile;
        }
        if (tile && std::string(tile) != "off") {
            int splits = 1;
            if (config.variant == deep_gemm::cdna4::KernelVariant::mfma_16x16x128_split_k)
                splits = config.split_k;
            else if (config.variant == deep_gemm::cdna4::KernelVariant::mfma_16x16x128_small_m)
                splits = config.waves;
            GemmArgs args = make_args(a, sfa, b, sfb, d, nullptr, m, n, k, lhs.ld, rhs.ld, lda_sf, ldb_sf, ldd);
            launch_compact_wv(args, tile, 1, splits, c10::hip::getCurrentHIPStream(d.get_device()).stream(), b, panels, panel_sf);
            return;
        }
    }
    // Each split writes FP32 [M, align(N,4)]; workspace_size is measured in bytes.
    const auto workspace_size = jit_runtime.workspace_size(config, m, n);
    auto workspace = workspace_size == 0 ? torch::Tensor()
                                         : torch::empty({static_cast<int64_t>(workspace_size / sizeof(float))},
                                                        d.options().dtype(torch::kFloat));
    const void *c_ptr = c.has_value() ? c->data_ptr() : nullptr;
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
struct GroupedOperandLayout {
    bool mn_major;
    int ld;
    int64_t group_stride;
};

GroupedOperandLayout grouped_operand_layout(const torch::Tensor &t, const char *name, OperandType type) {
    TORCH_CHECK(matches(t, type) && t.dim() == 3, name, " must be a 3-D ", operand_type_name(type), " tensor");
    const int64_t es = t.element_size();
    const auto mn = t.size(1), k = t.size(2);
    const bool mn_major = mn > 1 && k > 1 ? t.stride(2) != 1 : (mn == 1 ? false : t.stride(1) == 1 && k > 1);
    TORCH_CHECK(mn_major ? t.stride(1) == 1 || mn == 1 : t.stride(2) == 1 || k == 1, name,
                " must be K-major or MN-major within each group");
    int64_t ld = mn_major ? (k > 1 ? t.stride(2) * es : (mn * es + 3) / 4 * 4)
                          : (mn > 1 ? t.stride(1) * es : (k * es + 3) / 4 * 4);
    const auto row = (mn_major ? mn : k) * es;
    TORCH_CHECK(ld >= row && ld % 4 == 0 && ld < (1 << 24), name,
                " physical row stride must cover its row, be divisible by four and below 1 << 24");
    const auto group_stride = t.stride(0) * es;
    TORCH_CHECK(row % 4 == 0 || (reinterpret_cast<uintptr_t>(t.data_ptr()) % 4 == 0 && group_stride % 4 == 0), name,
                " ragged rows require four-byte aligned group bases");
    return {mn_major, static_cast<int>(ld), group_stride};
}

// Packed scales of a 3-D operand: [G, rows, ceil(K / (4 gran_k))], MN-major within each group.
std::pair<int, int64_t> grouped_scale_stride(const torch::Tensor &t, int groups, int rows, int k, int gran_k,
                                             const char *name) {
    TORCH_CHECK(gran_k == 32 || gran_k == 128, "scale granularity must be 32 or 128");
    TORCH_CHECK(t.scalar_type() == torch::kInt32 && t.dim() == 3 && t.size(0) == groups && t.size(1) == rows &&
                    t.size(2) == ceil_div(k, gran_k * 4),
                name, " has an invalid packed-scale dtype or shape");
    const auto pitch = t.size(2) > 1 ? t.stride(2) : (int64_t(rows) + 3) / 4 * 4;
    TORCH_CHECK((t.stride(1) == 1 || rows == 1) && pitch >= (int64_t(rows) + 3) / 4 * 4 && pitch % 4 == 0 &&
                    pitch <= std::numeric_limits<int>::max(),
                name, " must be MN-major with an aligned column stride covering all rows");
    return {static_cast<int>(pitch), t.stride(0)};
}

// Packed scales of a K-grouped operand: [sum of per-group words, rows], rows contiguous.
int k_grouped_scale_stride(const torch::Tensor &t, int rows, const char *name) {
    TORCH_CHECK(t.scalar_type() == torch::kInt32 && t.dim() == 2 && t.size(1) == rows, name,
                " has an invalid packed-scale dtype or shape");
    const auto pitch = t.size(0) > 1 ? t.stride(0) : (int64_t(rows) + 3) / 4 * 4;
    TORCH_CHECK((t.stride(1) == 1 || rows == 1) && pitch >= rows && pitch % 4 == 0 &&
                    pitch <= std::numeric_limits<int>::max(),
                name, " must have contiguous rows and an aligned pitch");
    return static_cast<int>(pitch);
}

void check_grouped_layout(const torch::Tensor &layout, int64_t numel, const torch::Device &device) {
    TORCH_CHECK(layout.is_cuda() && layout.device() == device, "grouped layout must be on the output device");
    TORCH_CHECK(layout.scalar_type() == torch::kInt32 && layout.is_contiguous() && layout.numel() == numel,
                "grouped layout must be a contiguous int32 tensor with ", numel, " elements");
}

void launch_or_throw(GemmArgs &args, const deep_gemm::cdna4::KernelConfig &config, int out_dtype,
                     const torch::Tensor &d, int tiles) {
    const auto stream = c10::hip::getCurrentHIPStream(d.get_device()).stream();
    const int error = jit_runtime.launch(args, config, out_dtype, stream, deep_gemm::cdna4::num_compute_units(), tiles);
    TORCH_CHECK(error == 0, "CDNA4 FP8 GEMM JIT launch failed with code ", error);
}

constexpr int kBlockMN = 256;

void check_m_alignment(int alignment, int block_m) {
    TORCH_CHECK(alignment > 0 && alignment % block_m == 0, "this CDNA4 grouped GEMM uses ", block_m,
                "-row tiles: the M alignment of the contiguous layout "
                "must be a multiple of ",
                block_m, " (set_mk_alignment_for_contiguous_layout)");
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
int set_group_m_slots(GemmArgs &args, const deep_gemm::cdna4::KernelConfig &config, int expected_rows, int max_rows,
                      int groups, int n) {
    args.group_m_slots = std::clamp(ceil_div(std::max(expected_rows, 1), config.block_m), 1,
                                    std::max(1, ceil_div(max_rows, config.block_m)));
    return groups * args.group_m_slots * ceil_div(n, config.block_n);
}

// D[M, N] = A[M, K] @ B[group(m), N, K]^T with per-row groups or PSUM group ends.
void cdna4_m_grouped_fp8_gemm_contiguous_impl(const torch::Tensor &a, const OptTensor &sfa, const torch::Tensor &b,
                                              const OptTensor &sfb, const torch::Tensor &d,
                                              const torch::Tensor &grouped_layout, int gran_k_a, int gran_k_b,
                                              bool use_psum_layout, bool ensure_zero_padding, int m_alignment,
                                              int expected_m) {
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
    TORCH_CHECK(logical_mn_k(b, b_type).second == k && d.size(0) == m && d.size(1) == n, "grouped GEMM shape mismatch");
    TORCH_CHECK(d.scalar_type() == torch::kBFloat16, "m-grouped GEMMs write BF16 outputs");
    TORCH_CHECK(n <= 1 || d.stride(1) == 1, "d must be N-major");
    check_grouped_layout(grouped_layout, use_psum_layout ? groups : m, device);
    TORCH_CHECK(m_alignment > 0, "the M alignment must be positive");
    const c10::hip::HIPGuardMasqueradingAsCUDA device_guard(d.get_device());
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
    const bool small_m =
        !rhs.mn_major &&
        ((use_psum_layout && deep_gemm::cdna4::small_m_eligible(expected_m, n, kb, 0)) || m_alignment % kBlockMN != 0);
    auto config =
        small_m
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
    const auto stream = c10::hip::getCurrentHIPStream(d.get_device()).stream();
    const int error =
        jit_runtime.launch(args, config, MXFP8_OUT_BF16, stream, deep_gemm::cdna4::num_compute_units(), tiles);
    TORCH_CHECK(error == 0, "CDNA4 grouped GEMM launch failed: ", error);
}

void cdna4_m_grouped_fp8_gemm_contiguous(const torch::Tensor &a, const OptTensor &sfa, const torch::Tensor &b,
                                         const OptTensor &sfb, const torch::Tensor &d, const torch::Tensor &layout,
                                         int ga, int gb, bool psum, bool zero, int alignment, int expected) {
    cdna4_m_grouped_fp8_gemm_contiguous_impl(a, sfa, b, sfb, d, layout, ga, gb, psum, zero, alignment, expected);
}

// D[g, :masked_m[g]] = A[g, :masked_m[g]] @ B[g]^T
// D[g, :masked_m[g]] (+)= alpha A[g, :masked_m[g]] @ B[g]^T: the masked layout, and (masked_m all
// full) batched GEMMs over strided views. A / B are K- or MN-major within each group (MN-major
// operands run the 256x256 tiles); D is N-major, C (if any) has D's strides.
void cdna4_m_grouped_fp8_gemm_masked(const torch::Tensor &a, const OptTensor &sfa, const torch::Tensor &b, const OptTensor &sfb,
    const torch::Tensor &d, const torch::Tensor &masked_m, int expected_m, int gran_k_a, int gran_k_b, const OptTensor &c, double alpha,
    const OptTensor &output_sf, const OptTensor &panels = std::nullopt, const OptTensor &panel_sf = std::nullopt) {
    const auto device = d.device();
    const bool bf16 = !sfa.has_value();
    TORCH_CHECK(sfa.has_value() == sfb.has_value(), "pass scales for both operands or for none");
    TORCH_CHECK(a.is_cuda() && a.device() == device && b.is_cuda() && b.device() == device && d.is_cuda() &&
                    (bf16 || (sfa->is_cuda() && sfa->device() == device && sfb->is_cuda() && sfb->device() == device)),
                "all tensors must be on the output accelerator device");
    TORCH_CHECK(a.dim() == 3 && b.dim() == 3 && d.dim() == 3, "masked GEMMs take 3-D a, b and d");
    const auto a_type = operand_type(a, !bf16, "a"), b_type = operand_type(b, !bf16, "b");
    const int groups = a.size(0), max_m = logical_mn_k(a, a_type).first, k = logical_mn_k(a, a_type).second;
    const int n = logical_mn_k(b, b_type).first;
    TORCH_CHECK(b.size(0) == groups && logical_mn_k(b, b_type).second == k && d.size(0) == groups &&
                    d.size(1) == max_m && d.size(2) == n,
                "masked GEMM shape mismatch");
    if (output_sf) {
        TORCH_CHECK(d.scalar_type() == at::kFloat8_e4m3fn && !c && alpha == 1.0 && !bf16 && groups == 8 && n == 1024 && k == 4096 &&
                        expected_m == max_m && (deep_gemm::cdna4::decode_batch_rows(max_m, 4) || deep_gemm::cdna4::decode_batch_rows(max_m, 5)) && d.stride(0) == n && d.stride(1) == groups * n &&
                        output_sf->device() == device && output_sf->scalar_type() == torch::kInt32 && output_sf->dim() == 2 &&
                        output_sf->size(0) == max_m && output_sf->size(1) == groups * n / 128 && output_sf->stride(0) == 1 &&
                        output_sf->stride(1) >= max_m && output_sf->stride(1) <= INT32_MAX,
            "invalid packed WV output layout");
    } else {
        TORCH_CHECK(d.scalar_type() == torch::kBFloat16 || d.scalar_type() == torch::kFloat, "d must be BF16 or FP32");
    }
    TORCH_CHECK(n <= 1 || d.stride(2) == 1, "d must be N-major");
    if (c.has_value())
        TORCH_CHECK(c->is_cuda() && c->device() == device && c->sizes() == d.sizes() && c->strides() == d.strides() &&
                        c->scalar_type() == d.scalar_type(),
                    "c must match d in shape, strides and dtype");
    TORCH_CHECK(expected_m > 0, "expected_m must be positive");
    check_grouped_layout(masked_m, groups, device);
    const c10::hip::HIPGuardMasqueradingAsCUDA device_guard(d.get_device());
    if (groups == 0 || max_m == 0 || n == 0)
        return;
    TORCH_CHECK(k > 0, "masked GEMMs need K > 0");
    const auto lhs = grouped_operand_layout(a, "a", a_type);
    const auto rhs = grouped_operand_layout(b, "b", b_type);
    const int mn_major = int(lhs.mn_major) | (int(rhs.mn_major) << 1);
    const int out_dtype = d.scalar_type() == torch::kFloat ? MXFP8_OUT_FP32 : MXFP8_OUT_BF16;
    const auto [lda_sf, sfa_group_stride] =
        bf16 ? std::pair<int, int64_t>{0, 0} : grouped_scale_stride(*sfa, groups, max_m, k, gran_k_a, "sfa");
    const auto [ldb_sf, sfb_group_stride] =
        bf16 ? std::pair<int, int64_t>{0, 0} : grouped_scale_stride(*sfb, groups, n, k, gran_k_b, "sfb");
    const int ldd = max_m > 1 ? stride_as_int(d, 1, "d") : n;
    const int kb = deep_gemm::cdna4::planner_k(k, a_type);
    const int sf_gran = bf16 ? 0 : int(gran_k_a == 128) | (int(gran_k_b == 128) << 1);
    const bool aligned =
        reinterpret_cast<uintptr_t>(d.data_ptr()) % 16 == 0 && (d.stride(0) * d.element_size()) % 16 == 0;
    auto config = deep_gemm::cdna4::small_m_eligible(expected_m, n, kb, mn_major)
                      ? deep_gemm::cdna4::get_small_m_config(expected_m, n, kb, sf_gran,
                                                             deep_gemm::cdna4::GemmType::MGroupedMasked, groups)
                      : deep_gemm::cdna4::get_m_grouped_config(deep_gemm::cdna4::GemmType::MGroupedMasked, kb, ldd,
                                                               aligned, mn_major, sf_gran, out_dtype, c.has_value());
    config.a_type = a_type;
    config.b_type = b_type;
    GemmArgs args = make_args(a, sfa, b, sfb, d, c.has_value() ? c->data_ptr() : nullptr, max_m, n, k, lhs.ld, rhs.ld,
                              lda_sf, ldb_sf, ldd);
    args.alpha = static_cast<float>(alpha);
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
    if (output_sf) {
        TORCH_CHECK(config.variant == deep_gemm::cdna4::KernelVariant::mfma_16x16x128_persistent, "packed WV requires the persistent GEMM");
        args.moe_sf = output_sf->data_ptr<int32_t>();
        args.moe_sf_pitch = output_sf->stride(1);
        if (const char *shape = std::getenv("DG_CDNA4_PACKED_WV_TILE"); shape && std::string(shape) != "off") {
            TORCH_CHECK(mn_major == 0 && a_type == OperandType::Fp8 && b_type == OperandType::Fp8 && gran_k_a == 32 && gran_k_b == 32,
                "packed WV tile requires row-wise FP8 inputs");
            launch_compact_wv(args, shape, groups, 0, c10::hip::getCurrentHIPStream(d.get_device()).stream(), b, panels, panel_sf);
            return;
        }
    }
    launch_or_throw(args, config, output_sf ? MXFP8_OUT_PACKED_WV : out_dtype, d, tiles);
}

// D[g] (+)= A_g^T B_g over the K range of group g. MN-major (TN): A [sum_k, M], B [sum_k, N].
// K-major (NT): per-group [M, k_g] / [N, k_g] blocks stored back to back; FP4: [M, sum_k / 2] /
// [N, sum_k / 2], a group's K range is a column range of every row.
void cdna4_k_grouped_fp8_gemm(const torch::Tensor &a, const OptTensor &sfa, const torch::Tensor &b,
                              const OptTensor &sfb, const torch::Tensor &d, const std::optional<torch::Tensor> &c,
                              const torch::Tensor &grouped_layout, bool use_psum_layout, int k_alignment, int gran_k,
                              bool k_major) {
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
    const c10::hip::HIPGuardMasqueradingAsCUDA device_guard(d.get_device());
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
    GemmArgs args =
        make_args(a, sfa, b, sfb, d, c.has_value() ? d.data_ptr() : nullptr, m, n, 0, lda, ldb, lda_sf, ldb_sf, n);
    args.grouped_layout = grouped_layout.data_ptr<std::int32_t>();
    args.num_groups = groups;
    args.k_alignment = k_alignment;
    args.d_group_stride = int64_t(m) * n;
    launch_or_throw(args, config, out_dtype, d, groups * ceil_div(m, kBlockMN) * ceil_div(n, kBlockMN));
}

// Return the unchanged split-K partials so the routing kernel can perform
// their original ordered reduction without materializing another logits tensor.
OptTensor cdna4_bf16_gate_partials(const torch::Tensor &x, const torch::Tensor &weight, const OptTensor &packed, int packed_layout) {
    TORCH_CHECK(x.is_cuda() && weight.is_cuda() && x.dim() == 2 && weight.dim() == 2 && x.is_contiguous() &&
                    weight.is_contiguous() && x.scalar_type() == torch::kBFloat16 &&
                    weight.scalar_type() == torch::kBFloat16 && x.device() == weight.device() &&
                    x.size(1) == weight.size(1),
                "invalid BF16 gate matrices");
    const c10::hip::HIPGuardMasqueradingAsCUDA guard(x.get_device());
    const int m = x.size(0), n = weight.size(0), k = x.size(1);
    if (!((m == 500 && n == 384) || (m == 400 && n == 128)) || k != 5120)
        return std::nullopt;
    TORCH_CHECK((packed_layout == 0) == !packed.has_value() && packed_layout >= 0 && packed_layout <= 2, "invalid prepared gate layout");
    if (packed) {
        const bool n_first = packed_layout == 1;
        TORCH_CHECK(packed->device() == weight.device() && packed->scalar_type() == torch::kBFloat16 && packed->is_contiguous() &&
                        packed->dim() == 5 && packed->size(0) == (n_first ? n / 16 : k / 32) &&
                        packed->size(1) == (n_first ? k / 32 : n / 16) && packed->size(2) == 4 && packed->size(3) == 16 &&
                        packed->size(4) == 8,
            "prepared gate operand must match the declared fragment layout");
    }
    const int cus = deep_gemm::cdna4::num_compute_units();
    const auto config = get_best_config(m, n, deep_gemm::cdna4::planner_k(k, OperandType::Bf16), MXFP8_OUT_FP32, false,
                                        n, true, 0, 0, OperandType::Bf16, OperandType::Bf16);
    if (config.variant != deep_gemm::cdna4::KernelVariant::mfma_16x16x128_split_k || config.split_k <= 1)
        return std::nullopt;
    const char *tile = m == 400 ? std::getenv("DG_GATE_DRAFT_ORDERED_TILE") : nullptr;
    if (!tile)
        tile = std::getenv("DG_GATE_ORDERED_TILE");
    if (tile && std::string(tile) != "off") {
        int fm = 0, nf = 0, batch = 0;
        char rest = 0;
        TORCH_CHECK(std::sscanf(tile, "%d,%d,%d%c", &fm, &nf, &batch, &rest) == 3 && (fm == 1 || fm == 2) &&
                        (nf == 1 || nf == 2 || nf == 4) && (batch == 4 || batch == 8 || batch == 16 || batch == 40) &&
                        int64_t(fm) * nf * 256 * batch * 4 <= 131072,
            "unsupported ordered gate tile");
        const int span = ceil_div(ceil_div(k, 64), config.split_k) * 64;
        auto logits = torch::empty({m, n}, x.options().dtype(torch::kFloat));
        auto args = make_args(x, std::nullopt, weight, std::nullopt, logits, nullptr, m, n, k, k, k, 0, 0, n);
        std::ostringstream code;
        if (packed) {
            args.B = static_cast<const uint8_t *>(packed->data_ptr());
            code << "#define CDNA4_GATE_PACKED_B " << packed_layout << "\n";
        }
        code << "#include <deep_gemm/impls/cdna4/ordered_gate.cuh>\n"
                "extern \"C\" __global__ __launch_bounds__(256) void ordered_gate(deep_gemm::cdna4::GemmArgs p){"
                "ordered_gate_tile<"
             << fm << "," << nf << "," << batch << "," << config.split_k << "," << span << ">(p);}";
        const auto text = code.str();
        const auto module = jit_runtime.get_module("ordered_gate:" + text, [&] { return text; }, {"ordered_gate"});
        const auto stream = c10::hip::getCurrentHIPStream(x.get_device()).stream();
        const int error = deep_gemm::cdna4::JitRuntime::launch_kernel(
            module->kernels[0], args, dim3(n / (16 * nf), ceil_div(m, 16 * fm)), dim3(256), fm * nf * 256 * batch * 4, stream);
        TORCH_CHECK(error == 0, "ordered gate projection launch failed: ", error);
        return logits;
    }
    const char *major = std::getenv("DG_GATE_TOKEN_MAJOR"), *small_tile = std::getenv("DG_GATE_PARTIAL_TILE");
    const bool token_major = major && std::string(major) == "1" && small_tile && std::string(small_tile) != "off";
    auto workspace = token_major ? torch::empty({m, config.split_k, n}, x.options().dtype(torch::kFloat)).transpose(0, 1)
                                 : torch::empty({config.split_k, m, n}, x.options().dtype(torch::kFloat));
    const auto lhs = operand_layout(x, "gate input", OperandType::Bf16);
    const auto rhs = operand_layout(weight, "gate weight", OperandType::Bf16);
    auto args = make_args(x, std::nullopt, weight, std::nullopt, workspace, nullptr, m, n, k, lhs.ld, rhs.ld, 0, 0, n);
    args.splits = config.split_k;
    args.ws = workspace.data_ptr<float>();
    args.ldw = n;
    const auto stream = c10::hip::getCurrentHIPStream(x.get_device()).stream();
    if (const char *selected = std::getenv("DG_GATE_PARTIAL_TILE"); selected && std::string(selected) != "off") {
        int fm = 0, nf = 0, depth = 0;
        char tail = 0;
        TORCH_CHECK(std::sscanf(selected, "%d,%d,%d%c", &fm, &nf, &depth, &tail) == 3 &&
                        (fm == 1 || fm == 2 || fm == 4 || fm == 8) && (nf == 1 || nf == 2 || nf == 4) &&
                        (depth == 1 || depth == 2),
                    "DG_GATE_PARTIAL_TILE must be FM,NF,DEPTH");
        const int span = ceil_div(ceil_div(k, 64), config.split_k) * 64;
        const int last = k - (config.split_k - 1) * span;
        TORCH_CHECK(last > 0 && last % 64 == 0, "gate partial K range must contain full BF16 steps");
        const int tiles = ceil_div(m, 16 * fm) * ceil_div(n, 16 * nf);
        const char *direct_flag = std::getenv("DG_GATE_PARTIAL_DIRECT");
        const bool direct = direct_flag && std::string(direct_flag) == "1";
        int group = 0;
        if (const char *order = std::getenv("DG_GATE_PARTIAL_ORDER"); order && std::string(order) != "0") {
            char *end = nullptr;
            long parsed = std::strtol(order, &end, 10);
            TORCH_CHECK(
                end != order && !*end && (parsed == 2 || parsed == 4 || parsed == 8 || parsed == 40) && config.split_k % parsed == 0,
                "invalid gate split launch group");
            group = int(parsed);
        }
        std::ostringstream source;
        if (packed) {
            args.B = static_cast<const uint8_t *>(packed->data_ptr());
            source << "#define CDNA4_GATE_PACKED_B " << packed_layout << "\n";
        }
        if (direct)
            source << "#define CDNA4_DIRECT_GATE_SPLITS 1\n";
        source << "#include <deep_gemm/impls/cdna4/fp8_gemm_mfma_16x16x128_small_m.cuh>\n"
                  "extern \"C\" __global__ __launch_bounds__(64) void gate_partials(GemmArgs p) {";
        if (group)
            source << "const int slice=blockIdx.x%" << group << "+blockIdx.x/" << (tiles * group) << "*" << group << ",tile=(blockIdx.x/"
                   << group << ")%" << tiles << ";";
        else
            source << "const int slice=blockIdx.y,tile=blockIdx.x;";
        source << "p.A += uint64_t(slice)*" << span * 2 << ";"
               << "p.B += uint64_t(slice)*"
               << span * 2 *
                      (packed_layout == 1      ? 16
                          : packed_layout == 2 ? n
                                               : 1)
               << ";"
               << "p.D = reinterpret_cast<float*>(p.D)+uint64_t(slice)*" << workspace.stride(0) << ";"
               << "p.M=" << m << ";p.N=" << n << ";p.K=(slice==" << config.split_k - 1 << "?" << last << ":" << span << ");"
               << "p.lda=" << lhs.ld << ";p.ldb=" << rhs.ld << ";p.ldd=" << workspace.stride(1) << ";p.C=nullptr;p.alpha=1.f;"
               << "fp8_gemm_mfma_16x16x128_small_m_impl<Mfma16x16x128SmallMConfig<" << fm << "," << nf << "," << depth
               << ",1,0,GemmType::Normal,false,OperandType::Bf16,OperandType::Bf16>,float>(p,tile);}";
        const auto text = source.str();
        const auto module = jit_runtime.get_module("gate_partials:" + text, [&] { return text; }, {"gate_partials"});
        const unsigned lds = direct ? 0 : 16 * fm * (16 * nf + 4) * sizeof(float);
        const dim3 grid = group ? dim3(tiles * config.split_k) : dim3(tiles, config.split_k);
        const int error = deep_gemm::cdna4::JitRuntime::launch_kernel(module->kernels[0], args, grid, dim3(64, 1, 1), lds, stream);
        TORCH_CHECK(error == 0, "tiled BF16 gate partial launch failed: ", error);
    } else {
        const int error = jit_runtime.launch(args, config, MXFP8_OUT_FP32, stream, cus, 0, false);
        TORCH_CHECK(error == 0, "BF16 gate partial launch failed: ", error);
    }
    return workspace;
}

// MoE gate routing (mega_gate.cuh) from FP32 logits [tokens, routed experts]; the Python wrapper
// validates the tensors and computes the logits with the BF16 GEMM
void cdna4_mega_gate_routing(const torch::Tensor &logits, const OptTensor &bias, const OptTensor &image_bias,
                             const OptTensor &image_token_mask, const OptTensor &mask,
                             const OptTensor &fix_routing_mask, const OptTensor &force_random,
                             const OptTensor &to_physical_map, const OptTensor &logical_count,
                             const torch::Tensor &topk_idx, const torch::Tensor &topk_weights,
                             const OptTensor &unmapped_topk_idx, int num_topk, int num_shared_experts, int rank_idx,
                             double routed_scaling_factor) {
    TORCH_CHECK((logits.dim() == 2 || logits.dim() == 3) && logits.scalar_type() == torch::kFloat &&
                    logits.stride(-1) == 1,
                "logits must be [tokens, experts] or [splits, tokens, experts] FP32");
    const int num_tokens = logits.size(-2), num_experts = logits.size(-1);
    const int splits = logits.dim() == 3 ? logits.size(0) : 1;
    TORCH_CHECK(splits == 1 || (num_tokens <= 512 && num_experts % 4 == 0),
                "fused gate reduction requires at most 512 tokens and complete four-expert vectors");
    const bool token_major = splits > 1 && logits.stride(0) == num_experts && logits.stride(1) == int64_t(splits) * num_experts;
    TORCH_CHECK(splits >= 1 && splits <= 80 && (logits.dim() == 2 || token_major || logits.stride(0) == num_tokens * logits.stride(1)),
        "gate split workspace must contain consecutive split matrices or per-token split rows");
    TORCH_CHECK(num_experts > 0 && num_experts <= 512 && num_topk > 0 && num_topk + num_shared_experts <= 32,
                "up to 512 routed experts and 32 slots");
    const c10::hip::HIPGuardMasqueradingAsCUDA device_guard(logits.get_device());
    if (num_tokens == 0)
        return;
    const auto ptr = [](const OptTensor &t) { return t.has_value() ? t->data_ptr() : nullptr; };
    deep_gemm::cdna4::MegaGateArgs args{};
    args.logits = logits.data_ptr<float>();
    args.ld_logits = stride_as_int(logits, logits.dim() - 2, "logits");
    args.bias = static_cast<const float *>(ptr(bias));
    args.image_bias = static_cast<const float *>(ptr(image_bias));
    args.image_token_mask = static_cast<const std::uint8_t *>(ptr(image_token_mask));
    args.mask = static_cast<const std::uint8_t *>(ptr(mask));
    args.fix_routing_mask = static_cast<const std::uint8_t *>(ptr(fix_routing_mask));
    args.force_random = static_cast<const std::uint8_t *>(ptr(force_random));
    args.to_physical_map = static_cast<const std::int32_t *>(ptr(to_physical_map));
    args.logical_count = static_cast<const std::int32_t *>(ptr(logical_count));
    args.topk_idx = topk_idx.data_ptr<std::int64_t>();
    args.topk_weights = topk_weights.data_ptr<float>();
    args.unmapped_topk_idx = unmapped_topk_idx.has_value() ? unmapped_topk_idx->data_ptr<std::int64_t>() : nullptr;
    args.unmapped_stride = unmapped_topk_idx.has_value() ? unmapped_topk_idx->stride(0) : 0;
    args.num_tokens = num_tokens, args.num_routed_experts = num_experts;
    args.num_shared_experts = num_shared_experts, args.num_topk = num_topk;
    args.num_duplicate_experts = to_physical_map.has_value() ? to_physical_map->size(1) : 0;
    args.rank_idx = static_cast<std::uint32_t>(rank_idx);
    args.routed_scaling_factor = static_cast<float>(routed_scaling_factor);
    TORCH_CHECK(args.ld_logits % 4 == 0 && args.ld_logits >= ceil_div(num_experts, 4) * 4 &&
                    reinterpret_cast<std::uintptr_t>(args.logits) % 16 == 0,
                "logits rows must be 16-byte aligned with a pitch of the experts rounded up to 4");
    const int experts_per_lane = ceil_div(num_experts, 64);
    const bool biased = bias.has_value() || image_bias.has_value();
    const auto instance = std::to_string(experts_per_lane) + (biased ? ", true" : ", false") +
                          (splits > 1 ? "," + std::to_string(splits) : "");
    std::string replica_define;
    if (const char *value = std::getenv("DG_GATE_REPLICA_BLOCK"); value && std::string(value) != "0" && to_physical_map) {
        char *end = nullptr;
        const long block = std::strtol(value, &end, 10);
        TORCH_CHECK(end != value && !*end && block >= 1 && block <= 1024, "replica token block must be in [1,1024]");
        replica_define = "#define DG_GATE_REPLICA_BLOCK " + std::to_string(block) + "\n";
    }
    if (token_major)
        replica_define += "#define DG_GATE_TOKEN_MAJOR 1\n";
    if (const char *group = std::getenv("DG_GATE_SPLIT_READS");
        splits > 1 && group &&
        (std::string(group) == "2" || std::string(group) == "4" || std::string(group) == "8" || std::string(group) == "16"))
        replica_define += "#define DG_GATE_SPLIT_READS " + std::string(group) + "\n";
    const auto module = jit_runtime.get_module("mega_gate:" + instance + replica_define,
        [&] {
            return replica_define +
                   "#include <deep_gemm/impls/cdna4/mega_gate.cuh>\n"
                   "extern \"C\" __global__ __launch_bounds__(256) void dg_cdna4_mega_gate(MegaGateArgs p) "
                   "{ " +
                   std::string(splits > 1 ? "mega_gate_split_impl<" : "mega_gate_impl<") + instance + ">(p); }\n";
        },
        {"dg_cdna4_mega_gate"}, {"-fno-honor-nans"});
    const unsigned blocks = static_cast<unsigned>(
        std::min(ceil_div(num_tokens, splits > 1 ? 2 : 4), 16 * deep_gemm::cdna4::num_compute_units()));
    const auto stream = c10::hip::getCurrentHIPStream(logits.get_device()).stream();
    const int error = deep_gemm::cdna4::JitRuntime::launch_kernel(module->kernels[0], args, dim3(blocks, 1, 1),
                                                                  dim3(256, 1, 1), 0, stream);
    TORCH_CHECK(error == 0, "CDNA4 mega gate launch failed with code ", error);
}

// Mega mHC (mega_mhc.cuh): the residual kernel over (64-token blocks, hidden splits), the mix
// kernel (a token per thread), then the norm kernel over the tokens. The Python wrapper validates
// the tensors and sizes the workspace: [token blocks, splits, 64, 25] floats of partials, then
// [tokens, 4] previous mixes; norm_bf16 is y_bf16 or (shifted, FP8 only) a scratch.
void cdna4_mega_mhc(const torch::Tensor &x, const torch::Tensor &residual, const OptTensor &shifted_prev_mix, const torch::Tensor &post_mix,
    const torch::Tensor &comb_res_mix, const torch::Tensor &fn, const torch::Tensor &mix_scales, const torch::Tensor &mix_bases,
    double hc_norm_eps, double hc_pre_eps, double hc_post_scale, double sinkhorn_eps, int num_sinkhorn_iters,
    const torch::Tensor &rmsnorm_weight, double rmsnorm_eps, double rmsnorm_scale, const torch::Tensor &new_residual,
    const OptTensor &new_prev_mix, const torch::Tensor &new_post_mix, const torch::Tensor &new_comb_res_mix, const OptTensor &norm_bf16,
    bool write_y_bf16, const OptTensor &y_fp8, const OptTensor &y_sf, const OptTensor &y_shared_sf, int shared_sf_block_m,
    const torch::Tensor &workspace, int splits, const OptTensor &fn_panels = std::nullopt) {
    const int num_tokens = x.size(0), hidden = x.size(1);
    TORCH_CHECK(hidden % 1024 == 0 && hidden / 1024 <= 16, "hidden must be a multiple of 1024 up to 16384");
    const bool shifted = shifted_prev_mix.has_value();
    TORCH_CHECK(shifted == new_prev_mix.has_value() && (!shifted || norm_bf16.has_value()) &&
                (!write_y_bf16 || norm_bf16.has_value()) && (write_y_bf16 || y_fp8.has_value()));
    const int m_blocks = ceil_div(num_tokens, 64);
    const int64_t partial_floats = int64_t(m_blocks) * splits * 64 * 25;
    TORCH_CHECK(workspace.scalar_type() == torch::kFloat &&
                workspace.numel() >= partial_floats + int64_t(num_tokens) * 4);
    const c10::hip::HIPGuardMasqueradingAsCUDA device_guard(x.get_device());
    if (num_tokens == 0)
        return;
    const auto u16 = [](const torch::Tensor &t) { return reinterpret_cast<std::uint16_t *>(t.data_ptr()); };
    deep_gemm::cdna4::MegaMhcArgs args{};
    args.x = u16(x), args.residual = u16(residual), args.rmsnorm_weight = u16(rmsnorm_weight);
    args.post_mix = post_mix.data_ptr<float>(), args.comb_res_mix = comb_res_mix.data_ptr<float>();
    args.shifted_prev_mix = shifted ? shifted_prev_mix->data_ptr<float>() : nullptr;
    args.fn = fn.data_ptr<float>(), args.mix_scales = mix_scales.data_ptr<float>();
    args.mix_bases = mix_bases.data_ptr<float>();
    args.new_residual = u16(new_residual);
    args.new_prev_mix = shifted ? new_prev_mix->data_ptr<float>() : nullptr;
    args.new_post_mix = new_post_mix.data_ptr<float>(), args.new_comb_res_mix = new_comb_res_mix.data_ptr<float>();
    args.norm_bf16 = norm_bf16.has_value() ? u16(*norm_bf16) : nullptr;
    args.write_y_bf16 = write_y_bf16;
    args.y_fp8 = y_fp8.has_value() ? reinterpret_cast<std::uint8_t *>(y_fp8->data_ptr()) : nullptr;
    if (y_sf.has_value()) {
        args.y_sf = y_sf->data_ptr<std::int32_t>();
        args.y_sf_stride_token = y_sf->stride(0), args.y_sf_stride_word = y_sf->stride(1);
    }
    if (y_shared_sf.has_value()) {
        args.y_shared_sf = y_shared_sf->data_ptr<std::int32_t>();
        args.y_shared_sf_stride_word = y_shared_sf->stride(1);
        args.shared_sf_block_m = shared_sf_block_m;
    }
    args.mix_partials = workspace.data_ptr<float>();
    args.sqr_partials = args.mix_partials + int64_t(m_blocks) * splits * 64 * 24;
    args.prev_mix = shifted ? args.new_prev_mix : args.mix_partials + partial_floats;
    args.num_tokens = num_tokens, args.hidden = hidden, args.splits = splits;
    args.hc_norm_eps = hc_norm_eps, args.hc_pre_eps = hc_pre_eps, args.hc_post_scale = hc_post_scale;
    args.sinkhorn_eps = sinkhorn_eps, args.num_sinkhorn_iters = num_sinkhorn_iters;
    args.rmsnorm_eps = rmsnorm_eps, args.rmsnorm_scale = rmsnorm_scale;

    const std::string mode = shifted ? "true" : "false", chunks = std::to_string(hidden / 1024);
    std::string mix_policy = "0";
    if (const char *value = std::getenv("DG_MHC_MATRIX_MIX"); value && (std::string(value) == "1" || std::string(value) == "2"))
        mix_policy = value;
    const int mix_width = mix_policy == "0" ? 4 : 16;
    std::string residual_policy = "0";
    if (const char *v = std::getenv("DG_MHC_RESIDUAL_SCHEDULE"); v && std::string(v).size() == 1 && *v >= '1' && *v <= '4')
        residual_policy = v;
    std::string fn_mode = "0";
    if (fn_panels) {
        TORCH_CHECK(fn_panels->device() == x.device() && fn_panels->scalar_type() == torch::kBFloat16 && fn_panels->is_contiguous() &&
                        fn_panels->sizes() == at::IntArrayRef({hidden / 64, 4, 2, 2, 4, 24, 8}),
            "mHC FN panels must supply compact high/low BF16 operand rows");
        const char *v = std::getenv("DG_MHC_FN_PANEL_PATH");
        TORCH_CHECK(v && (std::string(v) == "1" || std::string(v) == "2" || std::string(v) == "3"), "invalid mHC panel read mode");
        fn_mode = v;
        if (residual_policy < "2")
            residual_policy = "2";
        if (fn_mode == "3" && residual_policy < "3")
            residual_policy = "3";
    }
    std::string scale_policy = "0";
    if (const char *value = std::getenv("DG_MHC_SCALE_WORDS");
        value && (std::string(value) == "1" || std::string(value) == "2" || std::string(value) == "3"))
        scale_policy = value;
    const auto module = jit_runtime.get_module("mega_mhc:fused_finish_v1:" + mode + ":" + chunks + ":scales=" + scale_policy +
                                                   ":mix=" + mix_policy + ":residual=" + residual_policy + ":fn=" + fn_mode,
        [&] {
            return (fn_panels ? ("#define DG_MHC_COMPACT_FN " + fn_mode + "\n") : std::string()) + "#define DG_MHC_RESIDUAL_SCHEDULE " +
                   residual_policy + "\n#define DG_MHC_MATRIX_MIX " + mix_policy + "\n#define DG_MHC_SCALE_WORDS " + scale_policy +
                   "\n#include <deep_gemm/impls/cdna4/mega_mhc.cuh>\n"
                   "extern \"C\" __global__ __launch_bounds__(256) void "
                   "dg_cdna4_mega_mhc_residual(MegaMhcArgs p) "
                   "{ mega_mhc_residual_impl<" +
                   mode +
                   ">(p); }\n"
                   "extern \"C\" __global__ __launch_bounds__(256) void dg_cdna4_mega_mhc_mix(MegaMhcArgs p) "
                   "{ mega_mhc_mix_impl(p); }\n"
                   "extern \"C\" __global__ __launch_bounds__(256) void dg_cdna4_mega_mhc_norm(MegaMhcArgs "
                   "p) "
                   "{ mega_mhc_norm_impl<" +
                   mode + ", " + chunks +
                   ">(p); }\n"
                   "extern \"C\" __global__ __launch_bounds__(256) void dg_cdna4_mega_mhc_finish(MegaMhcArgs p) { "
                   "const int mix_blocks=(p.num_tokens+" +
                   std::to_string(128 / mix_width - 1) + ")/" + std::to_string(128 / mix_width) +
                   "; "
                   "if(int(blockIdx.x)<mix_blocks) { if(threadIdx.x<128) mega_mhc_mix_impl(p,blockIdx.x,128); } "
                   "else mega_mhc_norm_impl<true," +
                   chunks + ">(p,int(blockIdx.x)-mix_blocks,int(gridDim.x)-mix_blocks); }\n";
        },
        {"dg_cdna4_mega_mhc_residual", "dg_cdna4_mega_mhc_mix", "dg_cdna4_mega_mhc_norm", "dg_cdna4_mega_mhc_finish"});
    const auto stream = c10::hip::getCurrentHIPStream(x.get_device()).stream();
    auto residual_args = args;
    if (fn_panels)
        residual_args.fn = reinterpret_cast<const float *>(fn_panels->data_ptr());
    const unsigned residual_lds = fn_mode == "2" || fn_mode == "3" ? 0u : (residual_policy >= "2" ? 48u : 64u) * 1024;
    int error = deep_gemm::cdna4::JitRuntime::launch_kernel(
        module->kernels[0], residual_args, dim3(m_blocks, splits, 1), dim3(256), residual_lds, stream);
    TORCH_CHECK(error == 0, "CDNA4 mega mHC residual launch failed with code ", error);
    const char *residual_check = std::getenv("DG_MHC_CHECK_RESIDUAL");
    if (residual_policy != "0" && residual_check && std::string(residual_check) == "1") {
        auto reference = jit_runtime.get_module("mhc-residual-original:" + mode,
            [&] {
                return std::string(
                           "#include <deep_gemm/impls/cdna4/mega_mhc.cuh>\n"
                           "extern \"C\" __global__ __launch_bounds__(256) void ref_residual(MegaMhcArgs p){mega_mhc_residual_impl<") +
                       mode + ">(p);}";
            },
            {"ref_residual"});
        auto nr = torch::empty_like(new_residual), ws = torch::empty_like(workspace);
        auto norm = shifted ? torch::empty_like(*norm_bf16) : torch::Tensor();
        auto ref = args;
        ref.new_residual = u16(nr);
        if (shifted)
            ref.norm_bf16 = u16(norm);
        ref.mix_partials = ws.data_ptr<float>();
        ref.sqr_partials = ref.mix_partials + int64_t(m_blocks) * splits * 64 * 24;
        const int status =
            deep_gemm::cdna4::JitRuntime::launch_kernel(reference->kernels[0], ref, dim3(m_blocks, splits), dim3(256), 64 * 1024, stream);
        TORCH_CHECK(status == 0, "original residual replay failed");
        at::_assert_async((nr.view(torch::kInt16) == new_residual.view(torch::kInt16)).all(), "mHC residual differs");
        at::_assert_async((ws.reshape({-1}).narrow(0, 0, partial_floats).view(torch::kInt32) ==
                              workspace.reshape({-1}).narrow(0, 0, partial_floats).view(torch::kInt32))
                              .all(),
            "mHC ordered partials differ");
        if (shifted)
            at::_assert_async((norm.view(torch::kInt16) == norm_bf16->view(torch::kInt16)).all(), "mHC normalization input differs");
    }
    int mix_threads = 256;
    if (const char *requested = std::getenv("DG_MHC_MIX_THREADS")) {
        char *end = nullptr;
        const long parsed = std::strtol(requested, &end, 10);
        TORCH_CHECK(end != requested && *end == '\0' && (parsed == 64 || parsed == 128 || parsed == 256),
                    "invalid DG_MHC_MIX_THREADS");
        mix_threads = int(parsed);
    }
    auto check_mix = [&]() {
        const char *check = std::getenv("DG_MHC_CHECK_MATRIX_MIX");
        if (mix_policy == "0" || !check || std::string(check) != "1")
            return;
        auto ref_prev = torch::empty({num_tokens * 4}, workspace.options());
        auto ref_post = torch::empty_like(new_post_mix);
        auto ref_comb = torch::empty_like(new_comb_res_mix);
        auto ref = args;
        ref.new_prev_mix = ref.prev_mix = ref_prev.data_ptr<float>();
        ref.new_post_mix = ref_post.data_ptr<float>();
        ref.new_comb_res_mix = ref_comb.data_ptr<float>();
        auto oracle = jit_runtime.get_module("mhc-original-mix-replay",
            [] {
                return std::string(
                    "#include <deep_gemm/impls/cdna4/mega_mhc.cuh>\n"
                    "extern \"C\" __global__ __launch_bounds__(256) void original_mix(MegaMhcArgs p){mega_mhc_mix_impl(p);}");
            },
            {"original_mix"});
        const int status = deep_gemm::cdna4::JitRuntime::launch_kernel(
            oracle->kernels[0], ref, dim3(ceil_div(num_tokens, mix_threads / 4)), dim3(mix_threads), 0, stream);
        TORCH_CHECK(status == 0, "original mHC mix replay failed");
        auto actual_prev = shifted ? new_prev_mix->reshape({-1}) : workspace.reshape({-1}).narrow(0, partial_floats, num_tokens * 4);
        at::_assert_async((ref_prev.view(torch::kInt32) == actual_prev.view(torch::kInt32)).all(), "mHC previous mix differs");
        at::_assert_async((ref_post.view(torch::kInt32) == new_post_mix.view(torch::kInt32)).all(), "mHC post mix differs");
        at::_assert_async((ref_comb.view(torch::kInt32) == new_comb_res_mix.view(torch::kInt32)).all(), "mHC combination mix differs");
    };
    const char *finish_option = std::getenv("DG_MHC_FUSED_FINISH");
    if (shifted && mix_threads == 128 && finish_option && std::string(finish_option) == "1") {
        // Shifted normalization consumes the previous layer's mix, so it is
        // independent of the new mix. A single grid owns disjoint token ranges
        // for both branches and joins them before the next model operation.
        int resident = 0;
        TORCH_CHECK(hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(&resident, module->kernels[3], 256, 0) ==
                            hipSuccess &&
                        resident > 0,
                    "fused mHC finish occupancy query failed");
        const int norm_blocks = std::min(num_tokens, resident * deep_gemm::cdna4::num_compute_units());
        const int mix_blocks = ceil_div(num_tokens, 128 / mix_width);
        error = deep_gemm::cdna4::JitRuntime::launch_kernel(
            module->kernels[3], args, dim3(mix_blocks + norm_blocks, 1, 1), dim3(256, 1, 1), 0, stream);
        TORCH_CHECK(error == 0, "fused mHC finish launch failed with code ", error);
        check_mix();
        return;
    }
    error = deep_gemm::cdna4::JitRuntime::launch_kernel(
        module->kernels[1], args, dim3(ceil_div(num_tokens, mix_threads / mix_width), 1, 1), dim3(mix_threads, 1, 1), 0, stream);
    TORCH_CHECK(error == 0, "CDNA4 mega mHC mix launch failed with code ", error);
    // a token per resident workgroup at a time (each loads the next token ahead)
    int resident = 0;
    if (hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(&resident, module->kernels[2], 256, 0) != hipSuccess)
        resident = 2;
    const unsigned norm_blocks =
        static_cast<unsigned>(std::min(num_tokens, std::max(resident, 1) * deep_gemm::cdna4::num_compute_units()));
    error = deep_gemm::cdna4::JitRuntime::launch_kernel(module->kernels[2], args, dim3(norm_blocks, 1, 1),
                                                        dim3(256, 1, 1), 0, stream);
    TORCH_CHECK(error == 0, "CDNA4 mega mHC norm launch failed with code ", error);
    check_mix();
}

// D = A @ B^T and sqr_sum = sum_k A^2 for BF16 A [M, K], FP32 B [N, K] (hc_prenorm_gemm.cuh). With
// num_splits, D [splits, M, N] and sqr_sum [splits, M] receive the K splits' partials; without,
// D [M, N] and sqr_sum [M] the sums (partials of internal splits are added in split order).
void cdna4_hc_prenorm_gemm(const torch::Tensor &a, const torch::Tensor &b, const torch::Tensor &d,
                           const torch::Tensor &sqr_sum, std::optional<int> num_splits) {
    const auto device = d.device();
    TORCH_CHECK(a.is_cuda() && a.device() == device && b.is_cuda() && b.device() == device && sqr_sum.is_cuda() &&
                    sqr_sum.device() == device,
                "all tensors must be on the output accelerator device");
    TORCH_CHECK(a.dim() == 2 && a.scalar_type() == torch::kBFloat16 && a.stride(1) == 1,
                "a must be K-major BF16 [M, K]");
    TORCH_CHECK(b.dim() == 2 && b.scalar_type() == torch::kFloat && b.stride(1) == 1, "b must be K-major FP32 [N, K]");
    const int m = a.size(0), k = a.size(1), n = b.size(0);
    TORCH_CHECK(b.size(1) == k && n > 0 && k > 0, "prenorm GEMM shape mismatch");
    TORCH_CHECK(d.scalar_type() == torch::kFloat && sqr_sum.scalar_type() == torch::kFloat && d.is_contiguous() &&
                    sqr_sum.is_contiguous(),
                "d and sqr_sum must be contiguous FP32 tensors");
    if (num_splits.has_value()) {
        TORCH_CHECK(*num_splits >= 1 && d.dim() == 3 && d.size(0) == *num_splits && d.size(1) == m && d.size(2) == n &&
                        sqr_sum.dim() == 2 && sqr_sum.size(0) == *num_splits && sqr_sum.size(1) == m,
                    "split outputs must be [num_splits, M, N] and [num_splits, M]");
    } else {
        TORCH_CHECK(d.dim() == 2 && d.size(0) == m && d.size(1) == n && sqr_sum.dim() == 1 && sqr_sum.size(0) == m,
                    "outputs must be [M, N] and [M]");
    }
    const int lda = stride_as_int(a, 0, "a"), ldb = stride_as_int(b, 0, "b");
    TORCH_CHECK(lda % 8 == 0 && ldb % 4 == 0 && reinterpret_cast<uintptr_t>(a.data_ptr()) % 16 == 0 &&
                    reinterpret_cast<uintptr_t>(b.data_ptr()) % 16 == 0,
                "a and b rows must be 16-byte aligned");
    const c10::hip::HIPGuardMasqueradingAsCUDA device_guard(d.get_device());
    if (m == 0)
        return;
    constexpr int fm = 4;
    const int nf = std::min(4, ceil_div(n, 16));
    const int row_blocks = ceil_div(m, 16 * fm), col_blocks = ceil_div(n, 16 * nf), k_steps = ceil_div(k, 128);
    const int cus = deep_gemm::cdna4::num_compute_units();
    int splits = num_splits.value_or(1);
    torch::Tensor ws_d, ws_sqr;
    if (!num_splits.has_value() && row_blocks * col_blocks < cus) {
        // more workgroups for the compute units: partials of K splits of at least 4 steps per wave
        splits = std::clamp(ceil_div(2 * cus, row_blocks * col_blocks), 1, std::max(1, k_steps / 16));
        if (splits > 1) {
            ws_d = torch::empty({splits, m, n}, d.options());
            ws_sqr = torch::empty({splits, m}, d.options());
        }
    }
    deep_gemm::cdna4::HcPrenormGemmArgs args{};
    args.a = reinterpret_cast<const std::uint16_t *>(a.data_ptr());
    args.b = b.data_ptr<float>();
    args.d = ws_d.defined() ? ws_d.data_ptr<float>() : d.data_ptr<float>();
    args.sqr_sum = ws_sqr.defined() ? ws_sqr.data_ptr<float>() : sqr_sum.data_ptr<float>();
    args.m = m, args.n = n, args.k = k, args.lda = lda, args.ldb = ldb, args.splits = splits;
    args.final_d = d.data_ptr<float>();
    args.final_sqr_sum = sqr_sum.data_ptr<float>();
    const std::string key = "hc_prenorm:" + std::to_string(fm) + "x" + std::to_string(nf);
    const auto module = jit_runtime.get_module(
        key,
        [&] {
            const std::string cfg = "HcPrenormGemmConfig<" + std::to_string(fm) + "," + std::to_string(nf) + ">";
            return "#include <deep_gemm/impls/cdna4/hc_prenorm_gemm.cuh>\n"
                   "extern \"C\" __global__ __attribute__((amdgpu_flat_work_group_size(256, 256), "
                   "amdgpu_waves_per_eu(1, 1))) "
                   "void dg_cdna4_tf32_hc_prenorm_gemm(HcPrenormGemmArgs p) { hc_prenorm_gemm_impl<" +
                   cfg +
                   ">(p); }\n"
                   "extern \"C\" __global__ __launch_bounds__(256) void "
                   "dg_cdna4_tf32_hc_prenorm_reduce(HcPrenormGemmArgs p) "
                   "{ hc_prenorm_reduce_impl(p); }\n";
        },
        {"dg_cdna4_tf32_hc_prenorm_gemm", "dg_cdna4_tf32_hc_prenorm_reduce"});
    const auto stream = c10::hip::getCurrentHIPStream(d.get_device()).stream();
    const unsigned lds = 4 * (16 * fm * 16 * nf + 16 * fm) * 4;
    int error = deep_gemm::cdna4::JitRuntime::launch_kernel(
        module->kernels[0], args, dim3(row_blocks, splits, col_blocks), dim3(256, 1, 1), lds, stream);
    if (error == 0 && ws_d.defined()) {
        const int64_t elements = int64_t(m) * n + m;
        const unsigned blocks = static_cast<unsigned>(std::min<int64_t>(ceil_div(elements, 256), 4 * cus));
        error = deep_gemm::cdna4::JitRuntime::launch_kernel(module->kernels[1], args, dim3(blocks, 1, 1),
                                                            dim3(256, 1, 1), 0, stream);
    }
    TORCH_CHECK(error == 0, "CDNA4 prenorm GEMM launch failed with code ", error);
}

// D (+)= sum_b A[b] @ B[b]^T for contiguous BF16 A [batches, M, k], B [batches, N, k] (einsum
// "bmk,bnk->mn"): one GEMM whose K runs through the batches, split over the compute units when
// the output has few tiles.
void cdna4_k_batched_gemm(const torch::Tensor &a, const torch::Tensor &b, const torch::Tensor &d, const OptTensor &c) {
    const auto device = d.device();
    TORCH_CHECK(a.is_cuda() && a.device() == device && b.is_cuda() && b.device() == device,
                "all tensors must be on the output accelerator device");
    TORCH_CHECK(a.dim() == 3 && b.dim() == 3 && a.is_contiguous() && b.is_contiguous() &&
                    a.scalar_type() == torch::kBFloat16 && b.scalar_type() == torch::kBFloat16,
                "a and b must be contiguous [batches, M, k] / [batches, N, k] BF16 tensors");
    const int batches = a.size(0), m = a.size(1), k = a.size(2), n = b.size(1);
    TORCH_CHECK(b.size(0) == batches && b.size(2) == k && d.dim() == 2 && d.size(0) == m && d.size(1) == n,
                "K-batched GEMM shape mismatch");
    TORCH_CHECK(d.scalar_type() == torch::kBFloat16 || d.scalar_type() == torch::kFloat, "d must be BF16 or FP32");
    TORCH_CHECK(n <= 1 || d.stride(1) == 1, "d must be N-major");
    if (c.has_value())
        TORCH_CHECK(c->sizes() == d.sizes() && c->strides() == d.strides() && c->scalar_type() == d.scalar_type(),
                    "c must match d in shape, strides and dtype");
    const c10::hip::HIPGuardMasqueradingAsCUDA device_guard(d.get_device());
    if (m == 0 || n == 0)
        return;
    if (batches == 0 || k == 0) {
        if (!c.has_value())
            d.zero_();
        else if (c->data_ptr() != d.data_ptr())
            d.copy_(*c);
        return;
    }
    TORCH_CHECK(k % 64 == 0, "K-batched GEMMs need k divisible by 64");
    TORCH_CHECK(int64_t(batches) * k <= std::numeric_limits<int>::max(), "K-batched GEMM K is out of range");
    const int total_k = batches * k, ldd = m > 1 ? stride_as_int(d, 0, "d") : n;
    deep_gemm::cdna4::KernelConfig config{deep_gemm::cdna4::KernelVariant::mfma_16x16x128_data_parallel};
    config.type = deep_gemm::cdna4::GemmType::KBatched;
    config.a_type = config.b_type = OperandType::Bf16;
    config.split_k = deep_gemm::cdna4::plan_splits(m, n, 2 * total_k, deep_gemm::cdna4::num_compute_units());
    if (config.split_k > 1)
        config.variant = deep_gemm::cdna4::KernelVariant::mfma_16x16x128_split_k;
    const int out_dtype = d.scalar_type() == torch::kFloat ? MXFP8_OUT_FP32 : MXFP8_OUT_BF16;
    const auto workspace_size = jit_runtime.workspace_size(config, m, n);
    auto workspace = workspace_size == 0 ? torch::Tensor()
                                         : torch::empty({static_cast<int64_t>(workspace_size / sizeof(float))},
                                                        d.options().dtype(torch::kFloat));
    GemmArgs args = make_args(a, std::nullopt, b, std::nullopt, d, c.has_value() ? c->data_ptr() : nullptr, m, n,
                              total_k, 2 * k, 2 * k, 0, 0, ldd);
    args.k_batch_tiles = k / 64;
    args.a_group_stride = int64_t(m) * k * 2;
    args.b_group_stride = int64_t(n) * k * 2;
    args.splits = config.split_k;
    args.ws = workspace.defined() ? workspace.data_ptr<float>() : nullptr;
    args.ldw = ceil_div(n, 4) * 4;
    launch_or_throw(args, config, out_dtype, d, 0);
}

// MQA logits of q [num_q, H, D] (packed FP4: D / 2 bytes) against KV rows from `source`
// (MqaLogitsSource) into the preallocated logits. The Python wrappers validate shapes and build the
// per-source tensors (metadata).
void cdna4_mqa_logits(const torch::Tensor &q, const OptTensor &q_sf, const torch::Tensor &kv, const OptTensor &kv_sf,
    const torch::Tensor &weights, const torch::Tensor &logits, int source, const OptTensor &ks, const OptTensor &ke,
    const OptTensor &group_ranges, const OptTensor &context_lens, const OptTensor &block_table, const OptTensor &run_starts, int next_n,
    int page_kv, const OptTensor &sparse_starts, int num_max_sparse_blocks, int sparse_block_kv, int token_groups, int column_rounds,
    int sparse_mode, const OptTensor &block_scores, const OptTensor &block_counts) {
    using deep_gemm::cdna4::MqaLogitsFormat;
    using deep_gemm::cdna4::MqaLogitsSource;
    const auto src = static_cast<MqaLogitsSource>(source);
    const bool paged = src == MqaLogitsSource::Paged || src == MqaLogitsSource::PagedSparse;
    const bool fp4 = is_packed_fp4(q);
    const auto format = !q_sf.has_value() ? MqaLogitsFormat::Fp8
                        : fp4             ? MqaLogitsFormat::MxFp4
                                          : MqaLogitsFormat::MxFp8;
    TORCH_CHECK(q.dim() == 3 && q.is_contiguous() && (fp4 || q.scalar_type() == torch::kFloat8_e4m3fn),
                "q must be a contiguous [num_q, H, D] FP8 or packed FP4 tensor");
    TORCH_CHECK(format != MqaLogitsFormat::Fp8 || !fp4, "FP4 Q needs scales");
    const int num_q = q.size(0), num_heads = q.size(1), head_dim = q.size(2) * (fp4 ? 2 : 1);
    TORCH_CHECK(head_dim == 32 || head_dim == 64 || head_dim == 128, "head dims of 32, 64 or 128 values");
    TORCH_CHECK(!fp4 || head_dim >= 64, "FP4 heads hold 64 or 128 values");
    if (q_sf.has_value())
        TORCH_CHECK(q_sf->scalar_type() == torch::kInt32 && q_sf->is_contiguous() &&
                        q_sf->numel() == int64_t(num_q) * num_heads,
                    "q_sf must hold one packed int32 per token and head");
    TORCH_CHECK(weights.dim() == 2 && weights.size(0) == num_q && weights.size(1) == num_heads &&
                    weights.stride(1) == 1 &&
                    weights.scalar_type() == (format == MqaLogitsFormat::Fp8 ? torch::kFloat : torch::kBFloat16),
                "weights must be [num_q, H] with contiguous rows, float with FP8 Q, BF16 with MX Q");
    TORCH_CHECK(logits.dim() == 2 && logits.size(0) == num_q && logits.stride(1) == 1 &&
                    logits.scalar_type() == (format == MqaLogitsFormat::Fp8 ? torch::kFloat : torch::kBFloat16),
                "logits must be [num_q, columns] with contiguous rows");
    const c10::hip::HIPGuardMasqueradingAsCUDA device_guard(q.get_device());
    if (num_q == 0 || logits.size(1) == 0)
        return;

    deep_gemm::cdna4::MqaLogitsArgs args{};
    args.q = reinterpret_cast<const std::uint8_t *>(q.data_ptr());
    args.q_sf = q_sf.has_value() ? q_sf->data_ptr<std::int32_t>() : nullptr;
    args.kv = reinterpret_cast<const std::uint8_t *>(kv.data_ptr());
    args.kv_sf = kv_sf.has_value() ? kv_sf->data_ptr() : nullptr;
    args.weights = weights.data_ptr();
    args.ld_weights = stride_as_int(weights, 0, "weights");
    args.logits = logits.data_ptr();
    args.ld_logits = logits.stride(0);
    TORCH_CHECK(block_scores.has_value() == block_counts.has_value(), "supply both block maximum outputs");
    if (block_scores) {
        TORCH_CHECK(src == MqaLogitsSource::Paged && format == MqaLogitsFormat::MxFp4 && block_scores->dim() == 2 &&
                        block_scores->size(0) == num_q && block_scores->size(1) >= ceil_div(int(logits.size(1)), 8) &&
                        block_scores->is_contiguous() && block_scores->scalar_type() == torch::kBFloat16 &&
                        block_scores->device() == q.device() && block_counts->dim() == 1 && block_counts->numel() == num_q &&
                        block_counts->is_contiguous() && block_counts->scalar_type() == torch::kInt32 &&
                        block_counts->device() == q.device(),
            "invalid paged MXFP4 block-max output tensors");
        args.block_scores = reinterpret_cast<uint16_t *>(block_scores->data_ptr());
        args.block_counts = block_counts->data_ptr<int32_t>();
        args.ld_block_scores = block_scores->stride(0);
        args.score_width = int(logits.size(1));
    }
    args.num_q = num_q;
    const auto ptr = [](const OptTensor &t) { return t.has_value() ? t->data_ptr<std::int32_t>() : nullptr; };
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
    TORCH_CHECK(token_groups >= 0 && token_groups <= 8 && column_rounds >= 0, "invalid paged MQA launch plan");
    TORCH_CHECK(sparse_mode >= 0 && sparse_mode <= 3, "invalid sparse MQA address plan");
    TORCH_CHECK(!sparse_mode || (src == MqaLogitsSource::PagedSparse && format == MqaLogitsFormat::MxFp4 && sparse_block_kv == 8 &&
                                    (page_kv == 64 || page_kv == 128) && head_dim == 128 && num_heads == 32),
        "sparse block address specialization requires eight-row blocks of MXFP4 keys");
    TORCH_CHECK((token_groups == 0 && column_rounds == 0) || (sparse_mode && token_groups == 0) ||
                    (src == MqaLogitsSource::Paged && run_starts.has_value()),
        "MQA launch overrides require paged varlen inputs");
    const int query_group = token_groups > 0 ? token_groups : run_starts.has_value() ? 4 : args.next_n;
    const auto plan = deep_gemm::cdna4::plan_mqa_logits(format, src, head_dim, num_heads, query_group);
    // Paged: a workgroup per token, the first of each group of a request's tokens works
    const int waves = src == MqaLogitsSource::Paged ? num_q : ceil_div(num_q, plan.tokens_per_wave());
    TORCH_CHECK(!group_ranges.has_value() || group_ranges->numel() == 2 * int64_t(waves),
                "the MQA logits metadata does not match the number of heads");
    const auto stream = c10::hip::getCurrentHIPStream(q.get_device()).stream();
    const int error = deep_gemm::cdna4::launch_mqa_logits(args, plan, waves, logits.size(1), stream, column_rounds, sparse_mode);
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
    if (m == 0 || n == 0 || k == 0)
        return {0, 0};
    const auto tiles = int64_t(ceil_div(m, 256)) * ceil_div(n, 256);
    return {tiles, deep_gemm::cdna4::plan_splits(m, n, k, deep_gemm::cdna4::num_compute_units())};
}

} // namespace

#include "apis/mega_moe_hip.hpp"

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME _C
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "DeepGEMM HIP extension";
    py::class_<deep_gemm::cdna4::HipJit, std::shared_ptr<deep_gemm::cdna4::HipJit>>(m, "Runtime", py::module_local());
    m.def("get_jit", [] { return jit_runtime.get_runtime(); });
    uni_symm::register_python_api<uni_symm::HIP>(m);
    m.def("cdna4_init", &cdna4_init, py::arg("root"));
    m.def("cdna4_fp8_gemm_nt", &cdna4_fp8_gemm_nt, py::arg("a"), py::arg("sfa"), py::arg("b"), py::arg("sfb"), py::arg("d"),
        py::arg("c") = std::nullopt, py::arg("alpha") = 1.0, py::arg("gran_k_a") = 32, py::arg("gran_k_b") = 32,
        py::arg("reference_plan") = false, py::arg("panels") = std::nullopt, py::arg("panel_sf") = std::nullopt);
    m.def("cdna4_fp8_gemm_plan", &cdna4_fp8_gemm_plan, py::arg("m"), py::arg("n"), py::arg("k"));
    m.def("cdna4_m_grouped_fp8_gemm_contiguous", &cdna4_m_grouped_fp8_gemm_contiguous, py::arg("a"), py::arg("sfa"),
          py::arg("b"), py::arg("sfb"), py::arg("d"), py::arg("grouped_layout"), py::arg("gran_k_a"),
          py::arg("gran_k_b"), py::arg("use_psum_layout"), py::arg("ensure_zero_padding"), py::arg("m_alignment"),
          py::arg("expected_m") = 0);
    m.def("cdna4_m_grouped_fp8_gemm_masked", &cdna4_m_grouped_fp8_gemm_masked, py::arg("a"), py::arg("sfa"), py::arg("b"), py::arg("sfb"),
        py::arg("d"), py::arg("masked_m"), py::arg("expected_m"), py::arg("gran_k_a"), py::arg("gran_k_b"), py::arg("c") = std::nullopt,
        py::arg("alpha") = 1.0, py::arg("output_sf") = std::nullopt, py::arg("panels") = std::nullopt, py::arg("panel_sf") = std::nullopt);
    m.def("cdna4_k_grouped_fp8_gemm", &cdna4_k_grouped_fp8_gemm, py::arg("a"), py::arg("sfa"), py::arg("b"),
          py::arg("sfb"), py::arg("d"), py::arg("c"), py::arg("grouped_layout"), py::arg("use_psum_layout"),
          py::arg("k_alignment"), py::arg("gran_k"), py::arg("k_major"));
    m.def("cdna4_mqa_logits", &cdna4_mqa_logits, py::arg("q"), py::arg("q_sf"), py::arg("kv"), py::arg("kv_sf"), py::arg("weights"),
        py::arg("logits"), py::arg("source"), py::arg("ks") = std::nullopt, py::arg("ke") = std::nullopt,
        py::arg("group_ranges") = std::nullopt, py::arg("context_lens") = std::nullopt, py::arg("block_table") = std::nullopt,
        py::arg("run_starts") = std::nullopt, py::arg("next_n") = 1, py::arg("page_kv") = 0, py::arg("sparse_starts") = std::nullopt,
        py::arg("num_max_sparse_blocks") = 0, py::arg("sparse_block_kv") = 0, py::arg("token_groups") = 0, py::arg("column_rounds") = 0,
        py::arg("sparse_mode") = 0, py::arg("block_scores") = std::nullopt, py::arg("block_counts") = std::nullopt);
    m.def("cdna4_mqa_logits_tokens_per_wave", &cdna4_mqa_logits_tokens_per_wave, py::arg("num_heads"));
    m.def("cdna4_k_batched_gemm", &cdna4_k_batched_gemm, py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("c") = std::nullopt);
    m.def("cdna4_hc_prenorm_gemm", &cdna4_hc_prenorm_gemm, py::arg("a"), py::arg("b"), py::arg("d"), py::arg("sqr_sum"),
          py::arg("num_splits") = std::nullopt);
    m.def("cdna4_mega_moe_layout", &cdna4_mega_moe_layout, py::arg("num_ranks"), py::arg("num_experts"),
          py::arg("num_max_tokens"), py::arg("num_topk"), py::arg("hidden"), py::arg("intermediate"),
          py::arg("shared_intermediate"), py::arg("with_sf"));
    m.def("cdna4_shared_stages", &cdna4_shared_stages);
    m.def("cdna4_mega_moe", &cdna4_mega_moe, py::arg("y"), py::arg("l1_weights"), py::arg("l1_weights_sf"), py::arg("l2_weights"),
        py::arg("l2_weights_sf"), py::arg("shared_l1_weights"), py::arg("shared_l1_weights_sf"), py::arg("shared_l2_weights"),
        py::arg("shared_l2_weights_sf"), py::arg("stats"), py::arg("buffer"), py::arg("buffer_ptrs"), py::arg("rank"),
        py::arg("num_max_tokens"), py::arg("num_experts"), py::arg("num_topk"), py::arg("activation_clamp"), py::arg("fast_math"),
        py::arg("resident_weight_panels") = false, py::arg("reference") = false, py::arg("shared_result") = std::nullopt,
        py::arg("shared_panel_mask") = 0, py::arg("trace") = std::nullopt, py::arg("position") = std::nullopt);
    m.def("cdna4_prepare_moe_weight_panels", &cdna4_prepare_moe_weight_panels, py::arg("weight"));
    m.def("cdna4_mega_mhc", &cdna4_mega_mhc, py::arg("x"), py::arg("residual"), py::arg("shifted_prev_mix"), py::arg("post_mix"),
        py::arg("comb_res_mix"), py::arg("fn"), py::arg("mix_scales"), py::arg("mix_bases"), py::arg("hc_norm_eps"), py::arg("hc_pre_eps"),
        py::arg("hc_post_scale"), py::arg("sinkhorn_eps"), py::arg("num_sinkhorn_iters"), py::arg("rmsnorm_weight"), py::arg("rmsnorm_eps"),
        py::arg("rmsnorm_scale"), py::arg("new_residual"), py::arg("new_prev_mix"), py::arg("new_post_mix"), py::arg("new_comb_res_mix"),
        py::arg("norm_bf16"), py::arg("write_y_bf16"), py::arg("y_fp8"), py::arg("y_sf"), py::arg("y_shared_sf"),
        py::arg("shared_sf_block_m"), py::arg("workspace"), py::arg("splits"), py::arg("fn_panels") = std::nullopt);
    m.def("cdna4_bf16_gate_partials", &cdna4_bf16_gate_partials, py::arg("x"), py::arg("weight"), py::arg("packed_weight") = std::nullopt,
        py::arg("packed_layout") = 0);
    m.def("cdna4_mega_gate_routing", &cdna4_mega_gate_routing, py::arg("logits"), py::arg("bias"),
          py::arg("image_bias"), py::arg("image_token_mask"), py::arg("mask"), py::arg("fix_routing_mask"),
          py::arg("force_random"), py::arg("to_physical_map"), py::arg("logical_count"), py::arg("topk_idx"),
          py::arg("topk_weights"), py::arg("unmapped_topk_idx"), py::arg("num_topk"), py::arg("num_shared_experts"),
          py::arg("rank_idx"), py::arg("routed_scaling_factor"));
}
