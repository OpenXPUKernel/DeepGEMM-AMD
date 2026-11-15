#pragma once
#include <torch/extension.h>
#include <c10/hip/HIPStream.h>
#include <sstream>
#include "../../cdna4/jit_runtime.hpp"
#include "../heuristics/cdna4_mega_moe.hpp"
#include <deep_gemm/layout/cdna4_mega_moe.cuh>

namespace deep_gemm::cdna4::mega_moe_host {
inline GemmArgs matrix(const uint8_t *input, const int32_t *input_sf, const torch::Tensor &weight,
                       const torch::Tensor &weight_sf, void *output, int rows, int columns, int inner, int sf_pitch,
                       bool packed) {
    TORCH_CHECK(weight.stride(-1) == 1 && weight.stride(-2) == inner / (packed ? 2 : 1),
                "Mega MoE requires contiguous K-major weight rows");
    TORCH_CHECK(weight_sf.scalar_type() == torch::kInt32 && weight_sf.stride(-2) == 1);
    GemmArgs a{};
    a.A = input;
    a.B = reinterpret_cast<const uint8_t *>(weight.data_ptr());
    a.sfa = input_sf;
    a.sfb = weight_sf.data_ptr<int32_t>();
    a.D = output;
    a.M = rows;
    a.N = columns;
    a.K = inner;
    a.lda = inner;
    a.ldb = inner / (packed ? 2 : 1);
    a.lda_sf = sf_pitch;
    a.ldb_sf = weight_sf.stride(-1);
    a.ldd = columns;
    a.alpha = 1;
    a.splits = 1;
    a.m_alignment = 256;
    a.b_group_stride = weight.dim() == 3 ? weight.stride(0) : 0;
    a.sfb_group_stride = weight_sf.dim() == 3 ? weight_sf.stride(0) : 0;
    return a;
}

inline void constants(std::ostringstream &s, const char *name, const GemmArgs &a) {
    s << "#define " << name << "(a) do { ";
    if (a.K) {
        s << "(a).N=" << a.N << "; (a).K=" << a.K << "; (a).lda_sf=" << a.lda_sf << "; (a).ldb_sf=" << a.ldb_sf << "; "
          << "(a).b_group_stride=" << a.b_group_stride << "LL; (a).sfb_group_stride=" << a.sfb_group_stride << "LL; "
          << "(a).moe_record_stride=" << a.moe_record_stride << "; (a).moe_weight_word=" << a.moe_weight_word << "; "
          << "(a).moe_sf_pitch=" << a.moe_sf_pitch << "; (a).moe_fast_math=" << a.moe_fast_math << "; ";
    }
    s << "} while (0)\n";
}

inline void launch(MegaMoeArgs p, const torch::Tensor &buffer, const torch::Tensor &y, const torch::Tensor &w1,
                   const torch::Tensor &sf1, const torch::Tensor &w2, const torch::Tensor &sf2,
                   const std::optional<torch::Tensor> &sw1, const std::optional<torch::Tensor> &ssf1,
                   const std::optional<torch::Tensor> &sw2, const std::optional<torch::Tensor> &ssf2,
                   const torch::Tensor &l1_out, const torch::Tensor &l2_out, const torch::Tensor &shared_l1_out,
                   const torch::Tensor &shared_out, bool weight_panels) {
    using Config = heuristics::MegaMoeConfig;
    const auto &l = p.layout;
    const bool shared = sw1.has_value() && p.num_tokens > 0;
    TORCH_CHECK(l.with_sf && l.hidden % 256 == 0 && l.intermediate % 256 == 0 && w1.size(-1) * 2 == l.hidden &&
                    w2.size(-1) * 2 == l.intermediate,
                "Mega MoE requires aligned FP8 activations and packed FP4 routed weights");
    TORCH_CHECK(!shared || (l.shared_intermediate == l.intermediate && sw1->scalar_type() == torch::kFloat8_e4m3fn &&
                            sw2->scalar_type() == torch::kFloat8_e4m3fn && ssf1.has_value() && ssf2.has_value()),
                "shared experts require FP8 weights and scales");
    const int cus = num_compute_units(), blocks = std::min(cus, Config::max_blocks);
    int column_groups = Config::column_groups;
    if (const char *requested = std::getenv("DG_MEGA_MOE_COLUMN_GROUPS")) {
        char *end = nullptr;
        const long parsed = std::strtol(requested, &end, 10);
        TORCH_CHECK(end != requested && *end == '\0' && parsed > 0 && parsed <= 16,
                    "DG_MEGA_MOE_COLUMN_GROUPS must be in [1, 16]");
        column_groups = int(parsed);
    }
    int communication_blocks = Config::communication_blocks;
    if (const char *requested = std::getenv("DG_MEGA_MOE_GATHER_WORKGROUPS")) {
        char *end = nullptr;
        const long parsed = std::strtol(requested, &end, 10);
        TORCH_CHECK(end != requested && *end == '\0' && parsed > 0 && parsed <= 64,
                    "DG_MEGA_MOE_GATHER_WORKGROUPS must be in [1, 64]");
        communication_blocks = int(parsed);
    }
    TORCH_CHECK(blocks > communication_blocks + column_groups, "insufficient resident workgroups");
    TORCH_CHECK(int64_t(column_groups + 1) * 256 * std::max(l.hidden, l.intermediate) / 2 <= 0x7fffffffLL,
                "paired projection resource offsets exceed supported bounds");
    p.ordered_push = 1;
    p.direct_combine = 1;
    MegaMoeKernelArgs q{};
    q.moe = p;
    q.shared_tokens = shared ? p.num_tokens : 0;
    q.dispatch_blocks = communication_blocks;
    q.n1 = 2 * l.intermediate / 256;
    q.n2 = l.hidden / 256;
    q.max_blocks = (l.max_rows + 127) / 128 + (q.shared_tokens + 127) / 128;
    auto block_storage = torch::empty({q.max_blocks, 4}, y.options().dtype(torch::kInt32));
    auto state = torch::empty({192 + 2 * q.max_blocks + 2 * q.max_blocks * q.n2}, y.options().dtype(torch::kInt32));
    q.blocks = reinterpret_cast<MegaMoeRowBlock *>(block_storage.data_ptr<int>());
    q.state = state.data_ptr<int>();
    q.sf_ready = q.state + 192;
    q.l1_done = q.sf_ready + q.max_blocks;
    q.jobs = q.l1_done + q.max_blocks;
    q.job_ready = q.jobs + q.max_blocks * q.n2;
    auto *base = static_cast<uint8_t *>(buffer.data_ptr());
    auto ptr = [&](int64_t offset) { return base + offset; };
    const int64_t routed = int64_t(l.num_max_tokens) * l.num_ranks * std::min(l.num_topk, l.num_experts_per_rank);
    const int64_t last_input_row =
        (routed + int64_t(l.num_experts_per_rank) * (l.alignment - 1)) / l.alignment * l.alignment;
    constexpr int64_t control_bytes = 64 * 1024;
    TORCH_CHECK(last_input_row < l.max_rows && control_bytes <= (l.max_rows - last_input_row) * l.hidden,
                "grid epochs require a provably unused, initially-zero input tail");
    q.k_ready = reinterpret_cast<uint64_t *>(ptr(l.l1_acts + last_input_row * l.hidden));
    q.l1 = matrix(ptr(l.l1_acts), reinterpret_cast<int32_t *>(ptr(l.l1_sf)), w1, sf1, l1_out.data_ptr(), l.max_rows,
                  2 * l.intermediate, l.hidden, l.sf_pitch, true);
    q.l2 = matrix(ptr(l.l2_acts), reinterpret_cast<int32_t *>(ptr(l.l2_sf)), w2, sf2, l2_out.data_ptr(), l.max_rows,
                  l.hidden, l.intermediate, l.sf_pitch, true);
    q.l1.moe_acts = ptr(l.l2_acts);
    q.l1.moe_sf = reinterpret_cast<int32_t *>(ptr(l.l2_sf));
    q.l1.moe_records = reinterpret_cast<int32_t *>(ptr(l.l1_records));
    q.l1.moe_record_stride = l.record_ints;
    q.l1.moe_weight_word = l.hidden / 128;
    q.l1.moe_sf_pitch = l.sf_pitch;
    q.l1.moe_clamp = p.activation_clamp;
    q.l1.moe_fast_math = p.fast_math;
    q.l2.moe_output_rows = reinterpret_cast<int64_t *>(ptr(l.output_rows));
    if (shared) {
        q.shared_l1 =
            matrix(ptr(l.x), reinterpret_cast<int32_t *>(ptr(l.shared_l1_sf)), *sw1, *ssf1, shared_l1_out.data_ptr(),
                   p.num_tokens, 2 * l.intermediate, l.hidden, l.shared_sf_pitch, false);
        q.shared_l2 = matrix(ptr(l.shared_l2_acts), reinterpret_cast<int32_t *>(ptr(l.shared_l2_sf)), *sw2, *ssf2,
                             shared_out.data_ptr(), p.num_tokens, l.hidden, l.intermediate, l.shared_sf_pitch, false);
        q.shared_l1.moe_acts = ptr(l.shared_l2_acts);
        q.shared_l1.moe_sf = reinterpret_cast<int32_t *>(ptr(l.shared_l2_sf));
        q.shared_l1.moe_sf_pitch = l.shared_sf_pitch;
        q.shared_l1.moe_clamp = p.activation_clamp;
        q.shared_l1.moe_fast_math = p.fast_math;
    }

    std::ostringstream code;
    if (std::getenv("DG_MEGA_MOE_PHASE_TIMESTAMPS"))
        code << "#define DG_MEGA_MOE_PHASE_TIMESTAMPS 1\n";
    constants(code, "RESIDENT_L1", q.l1);
    constants(code, "RESIDENT_L2", q.l2);
    constants(code, "RESIDENT_S1", q.shared_l1);
    constants(code, "RESIDENT_S2", q.shared_l2);
    code << "#define CDNA4_MOE_WEIGHT_PANELS " << weight_panels
         << "\n"
            "#include <deep_gemm/impls/cdna4_fp8_fp4_mega_moe.cuh>\n"
            "extern \"C\" __global__ "
            "__attribute__((amdgpu_flat_work_group_size(256,256),amdgpu_waves_per_eu(1,1))) "
            "void dg_moe_resident(deep_gemm::cdna4::MegaMoeKernelArgs q){resident_moe::run<"
         << l.hidden << "," << l.intermediate << "," << l.num_topk << "," << column_groups << ">(q);}";
    const auto source = code.str();
    auto module = jit_runtime.get_module("cdna4_fp8_fp4_mega_moe:" + std::to_string(std::hash<std::string>{}(source)),
                                         [&] { return source; }, {"dg_moe_resident"}, {"-O2"});
    constexpr unsigned lds = Config::shared_memory_bytes;
    const int64_t table_words =
        3LL * l.num_ranks * l.num_experts_per_rank + l.num_ranks + 2LL * l.num_experts_per_rank + 2;
    TORCH_CHECK(((table_words + 7) / 8 * 8 + 128 * 8) * 4 <= lds, "gather metadata exceeds allocated shared memory");
    int occupancy = 0;
    TORCH_CHECK(hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, module->kernels[0], 256, lds) ==
                        hipSuccess &&
                    occupancy >= 1 && blocks <= occupancy * cus,
                "Mega MoE grid cannot be simultaneously resident");
    const auto stream = c10::hip::getCurrentHIPStream(y.get_device()).stream();
    // A caller opting into this policy owns an exclusive, serial execution
    // stream on the device. Independent/concurrent callers keep the cooperative
    // launch, which guarantees residency for the software grid barriers.
    const char *serial_stream = std::getenv("DG_MEGA_MOE_SERIAL_STREAM");
    const bool cooperative = !serial_stream || std::string(serial_stream) != "1";
    TORCH_CHECK(JitModule::launch_function(module->kernels[0], dim3(blocks), dim3(256), lds, stream, cooperative, q) ==
                    hipSuccess,
                "resident Mega MoE launch failed");
}
} // namespace deep_gemm::cdna4::mega_moe_host
