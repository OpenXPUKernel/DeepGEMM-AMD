#pragma once

#include "cdna4/decode_batch.hpp"
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

inline void launch(MegaMoeArgs p, const torch::Tensor &buffer, const torch::Tensor &y, const torch::Tensor &w1, const torch::Tensor &sf1,
    const torch::Tensor &w2, const torch::Tensor &sf2, const std::optional<torch::Tensor> &sw1, const std::optional<torch::Tensor> &ssf1,
    const std::optional<torch::Tensor> &sw2, const std::optional<torch::Tensor> &ssf2, const torch::Tensor &l1_out,
    const torch::Tensor &l2_out, const torch::Tensor &shared_l1_out, const torch::Tensor &shared_out, bool weight_panels,
    bool shared_computed = false, int shared_panel_mask = 0, const std::optional<torch::Tensor> &trace = std::nullopt,
    const std::optional<torch::Tensor> &position = std::nullopt) {
    using Config = heuristics::MegaMoeConfig;
    const auto &l = p.layout;
    const bool shared = !shared_computed && sw1.has_value() && p.num_tokens > 0;
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
    if (trace) {
        q.trace = reinterpret_cast<uint64_t *>(trace->data_ptr<int64_t>());
        q.first_position = position->data_ptr();
        q.position64 = position->scalar_type() == torch::kInt64;
    }
    q.shared_tokens = shared ? p.num_tokens : 0;
    q.dispatch_blocks = communication_blocks;
    q.n1 = 2 * l.intermediate / 256;
    q.n2 = l.hidden / 256;
    q.max_blocks = (l.max_rows + 127) / 128 + (q.shared_tokens + 127) / 128;
    auto block_storage = torch::empty({q.max_blocks, 4}, y.options().dtype(torch::kInt32));
    int join = 0;
    if (const char *requested = std::getenv("DG_MEGA_MOE_ROUTE_JOIN"); requested) {
        std::string mode(requested);
        if (mode == "1" || mode == "2" || mode == "3")
            join = mode[0] - '0';
    }
    const char *bitmap = std::getenv("DG_MEGA_MOE_ROUTE_BITMAP_GROUP");
    // The route bitmap uses Group * ceil(entries / 32) uint32 words before
    // the shared allocation is reused for the gather table and matrix stages.
    const int64_t entries = int64_t(p.num_tokens) * l.num_topk;
    const bool configured_decode = std::getenv("DG_CDNA4_DECODE_BATCH") &&
                                  (deep_gemm::cdna4::decode_batch_rows(p.num_tokens, 5) || deep_gemm::cdna4::decode_batch_rows(p.num_tokens, 4));
    const bool bitmap_fits = entries <= 4096 || (configured_decode && 8 * ((entries + 31) / 32) * sizeof(unsigned) <= Config::shared_memory_bytes);
    const char *early_shared = std::getenv("DG_MEGA_MOE_EARLY_SHARED_L1");
    if (!bitmap ||
        !(std::string(bitmap) == "1" || std::string(bitmap) == "2" || std::string(bitmap) == "4" || std::string(bitmap) == "8") ||
        !bitmap_fits || (early_shared && std::string(early_shared) == "1"))
        join = 0;
    const int state_words = join ? 192 + 5 * q.max_blocks + (join == 3 ? blocks : 0) : 192 + 2 * q.max_blocks + 2 * q.max_blocks * q.n2;
    auto state = torch::empty({state_words}, y.options().dtype(torch::kInt32));
    q.blocks = reinterpret_cast<MegaMoeRowBlock *>(block_storage.data_ptr<int>());
    q.state = state.data_ptr<int>();
    q.sf_ready = q.state + 192;
    q.l1_done = q.sf_ready + q.max_blocks;
    q.jobs = q.l1_done + q.max_blocks;
    q.job_ready = q.jobs + q.max_blocks * (join ? 2 : q.n2);
    auto *base = static_cast<uint8_t *>(buffer.data_ptr());
    auto ptr = [&](int64_t offset) { return base + offset; };
    q.k_ready = reinterpret_cast<uint64_t *>(ptr(l.barrier + deep_gemm::cdna4::kFabricBarrierBytes));
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
    if (const char *mode = std::getenv("DG_MEGA_MOE_HALF_ROW_PAIR"); mode && std::string(mode) == "1")
        code << "#define CDNA4_HALF_ROW_PAIR 1\n";
    if (const char *v = std::getenv("DG_MEGA_MOE_SHORT_COLUMN_WAVES"); v && std::string(v) != "0") {
        const std::string mode(v);
        TORCH_CHECK(mode == "1" || mode == "2" || mode == "3", "invalid short-panel column schedule");
        code << "#define CDNA4_SHORT_COLUMN_WAVES " << mode << "\n";
    }
    if (const char *v = std::getenv("DG_MEGA_MOE_ACTIVE_SLABS"); weight_panels && v && std::string(v) != "0") {
        int mode = 0, projections = 0;
        char trailing = 0;
        TORCH_CHECK(std::sscanf(v, "%d,%d%c", &mode, &projections, &trailing) == 2 && mode >= 1 && mode <= 4 && projections >= 1 &&
                        projections <= 3,
            "invalid active row slab schedule");
        code << "#define CDNA4_ACTIVE_SLABS " << mode << "\n#define CDNA4_ACTIVE_SLAB_PROJECTIONS " << projections << "\n";
    }
    if (const char *v = std::getenv("DG_MEGA_MOE_SHORT_MFMA16"); weight_panels && v && std::string(v) != "0") {
        const std::string mode(v);
        TORCH_CHECK(mode == "1" || mode == "2" || mode == "3" || mode == "4" || mode == "5", "invalid short MFMA16 plan");
        code << "#define CDNA4_SHORT_MFMA16 " << mode << "\n";
    }
    if (const char *v = std::getenv("DG_MEGA_MOE_RESERVED_TICKETS"); v) {
        const std::string mode(v);
        if (mode == "1" || mode == "2" || mode == "3" || mode == "4")
            code << "#define CDNA4_MOE_RESERVED_TICKETS " << mode << "\n";
    }
    if (const char *v = std::getenv("DG_MEGA_MOE_BALANCE_DOWN"); weight_panels && v) {
        const std::string mode(v);
        if (mode == "1" || mode == "2" || mode == "3" || mode == "4")
            code << "#define CDNA4_BALANCE_DOWN " << mode << "\n";
    }
    if (const char *v = std::getenv("DG_MEGA_MOE_PROGRESSIVE_K");
        v && (std::string(v) == "1" || std::string(v) == "2" || std::string(v) == "3") && l.intermediate / 128 < 32 &&
        std::getenv("DG_MEGA_MOE_EARLY_K") && std::string(std::getenv("DG_MEGA_MOE_EARLY_K")) == "2")
        code << "#define CDNA4_PROGRESSIVE_K " << v << "\n";
    if (const char *mode = std::getenv("DG_MEGA_MOE_UP_WEIGHT_AHEAD");
        weight_panels && mode &&
        (std::string(mode) == "1" || std::string(mode) == "2" || std::string(mode) == "3" || std::string(mode) == "4"))
        code << "#define CDNA4_UP_WEIGHT_AHEAD " << mode << "\n";
    if (const char *choice = std::getenv("DG_MEGA_MOE_PACKED_EPILOGUE");
        choice && (std::string(choice) == "1" || std::string(choice) == "2" || std::string(choice) == "3"))
        code << "#define CDNA4_MOE_PACKED_EPILOGUE " << choice << "\n";
    if (trace) {
        if (const char *v = std::getenv("DG_MOE_TRACE_PEER_JOIN"); v && std::string(v) == "1")
            code << "#define CDNA4_TRACE_PEER_JOIN 1\n";
        code << "#define CDNA4_ONLINE_MOE_TIMING 1\n";
        if (const char *bounds = std::getenv("DG_MOE_TRACE_RANGE"); bounds) {
            int lo = 0, hi = 0;
            char tail = 0;
            TORCH_CHECK(std::sscanf(bounds, "%d,%d%c", &lo, &hi, &tail) == 2 && lo >= 0 && hi >= lo,
                "expected inclusive nonnegative trace position bounds");
            code << "#define CDNA4_TRACE_MIN_POSITION " << lo << "\n#define CDNA4_TRACE_MAX_POSITION " << hi << "\n";
        }
    }
    if (const char *plan = std::getenv("DG_MEGA_MOE_TAIL_GROUPS"); deep_gemm::cdna4::decode_batch_rows(p.num_tokens, 5) && plan && std::string(plan) != "0") {
        int maximum = 0, adaptive = 0;
        char extra = 0;
        TORCH_CHECK(std::sscanf(plan, "%d,%d%c", &maximum, &adaptive, &extra) == 2 && (maximum == 16 || maximum == 18 || maximum == 20) &&
                        (adaptive == 0 || adaptive == 1),
            "invalid last-wave column group plan");
        code << "#define CDNA4_TAIL_GROUPS " << maximum << "\n#define CDNA4_ADAPTIVE_TAIL " << adaptive << "\n";
    }
    if (shared_panel_mask)
        code << "#define CDNA4_SHARED_FP8_PANELS " << shared_panel_mask << "\n";
    if (const char *plan = std::getenv("DG_MEGA_MOE_ROLL_STAGES"); plan && std::string(plan) != "0") {
        int mode = 0, mask = 0;
        char extra = 0;
        TORCH_CHECK(std::sscanf(plan, "%d,%d%c", &mode, &mask, &extra) == 2 && (mode == 1 || mode == 2) && mask > 0 && mask <= 7,
            "expected a ring scheduling mode and a matrix role mask");
        code << "#define CDNA4_MOE_ROLL_STAGES " << mode << "\n#define CDNA4_MOE_ROLL_MASK " << mask << "\n";
    }
    int transpose_rows = 0;
    if (const char *plan = std::getenv("DG_MEGA_MOE_GATHER_SCALE_TILE"); plan && std::string(plan) != "0") {
        int address = 0;
        char extra = 0;
        TORCH_CHECK(std::sscanf(plan, "%d,%d%c", &transpose_rows, &address, &extra) == 2 &&
                        (transpose_rows == 0 || transpose_rows == 16 || transpose_rows == 32 || transpose_rows == 64) &&
                        (address == 0 || address == 1),
            "expected 0/16/32/64 rows and a 0/1 address flag");
        if (l.hidden / 128 > 64)
            transpose_rows = 0;
        if (transpose_rows)
            code << "#define CDNA4_GATHER_SCALE_ROWS " << transpose_rows << "\n";
        if (address)
            code << "#define CDNA4_GATHER_EARLY_ADDRESS 1\n";
    }
    if (join)
        code << "#define CDNA4_ROUTE_JOIN " << join << "\n";
    if (const char *cols = std::getenv("DG_MEGA_MOE_SHARED_DURING_ROUTE");
        join >= 2 && shared && cols &&
        (std::string(cols) == "8" || std::string(cols) == "12" || std::string(cols) == "18" || std::string(cols) == "20") &&
        uint64_t(l.num_max_tokens + 1) * l.hidden <= UINT32_MAX)
        code << "#define CDNA4_SHARED_DURING_ROUTE " << cols << "\n";
    if (const char *policy = std::getenv("DG_MEGA_MOE_IMMUTABLE_CACHE"); policy && std::string(policy) != "0") {
        int up = 0, down = 0;
        char trailing = 0;
        TORCH_CHECK(std::sscanf(policy, "%d,%d%c", &up, &down, &trailing) == 2 && up >= 0 && up <= 2 && down >= 0 && down <= 2,
            "expert weight cache policies must be a pair in [0,2]");
        code << "#define CDNA4_UP_WEIGHT_CACHE " << up << "\n#define CDNA4_DOWN_WEIGHT_CACHE " << down << "\n";
    }
    if (const char *mode = std::getenv("DG_MEGA_MOE_WAVE_SLOT_REDUCE"); mode && std::string(mode) != "0") {
        const std::string plan(mode);
        TORCH_CHECK(plan == "1" || plan == "2" || plan == "3", "invalid wave slot reduction plan");
        code << "#define CDNA4_WAVE_SLOT_REDUCE " << plan << "\n";
    }
    if (const char *policy = std::getenv("DG_MEGA_MOE_STAGE_RETURN"); policy && (std::string(policy) == "1" || std::string(policy) == "2"))
        code << "#define CDNA4_MOE_STAGE_RETURN " << policy << "\n";
    if (const char *group = std::getenv("DG_MEGA_MOE_ROUTE_BITMAP_GROUP");
        group && bitmap_fits &&
        (std::string(group) == "1" || std::string(group) == "2" || std::string(group) == "4" || std::string(group) == "8"))
        code << "#define CDNA4_MOE_ROUTE_BITMAP_GROUP " << group << "\n";
    if (const char *v = std::getenv("DG_MEGA_MOE_ROUTE_PLANES"); v) {
        std::string mode(v);
        if (mode == "1" || mode == "2" || mode == "3" || mode == "4")
            code << "#define CDNA4_MOE_ROUTE_PLANES " << mode << "\n";
    }
    if (const char *flag = std::getenv("DG_MEGA_MOE_DRAIN_TAIL"); flag && std::string(flag) == "1")
        code << "#define CDNA4_MOE_DRAIN_TAIL 1\n";
    if (const char *flag = std::getenv("DG_MEGA_MOE_EARLY_K"); flag && (std::string(flag) == "1" || std::string(flag) == "2"))
        code << "#define CDNA4_MOE_EARLY_K " << flag << "\n";
    if (const char *flag = std::getenv("DG_MEGA_MOE_LOCAL_INPUT");
        flag && (std::string(flag) == "1" || (std::string(flag) == "target500" && deep_gemm::cdna4::decode_batch_rows(p.num_tokens, 5))) &&
        (int64_t(l.num_max_tokens) + 1) * l.hidden <= UINT32_MAX) {
        code << "#define CDNA4_MOE_LOCAL_INPUT 1\n";
        if (const char *eager = std::getenv("DG_MEGA_MOE_EAGER_LOCAL");
            eager && (std::string(eager) == "1" || std::string(eager) == "2") && l.num_ranks <= 8)
            code << "#define CDNA4_MOE_EAGER_LOCAL " << eager << "\n";
        else if (const char *minimum = std::getenv("DG_MEGA_MOE_MIXED_LOCAL_ROWS");
            minimum && (std::string(minimum) == "1" || std::string(minimum) == "32" || std::string(minimum) == "64") && l.l1_acts >= l.x &&
            l.l1_records - l.x + l.hidden <= UINT32_MAX)
            code << "#define CDNA4_MOE_MIXED_LOCAL_ROWS " << minimum << "\n";
    }
    if (const char *flag = std::getenv("DG_MEGA_MOE_PUBLISH_TASK_TABLE"); flag && std::string(flag) == "1")
        code << "#define CDNA4_MOE_PUBLISH_TASK_TABLE 1\n";
    if (const char *flag = std::getenv("DG_MEGA_MOE_BOUND_GATHER_ROWS"); flag && std::string(flag) == "1")
        code << "#define CDNA4_MOE_BOUND_GATHER_ROWS 1\n";
    if (const char *flag = std::getenv("DG_MEGA_MOE_RECEIVER_COUNTS"); flag && l.num_experts_per_rank <= 256) {
        const std::string mode(flag);
        if (mode == "1" || mode == "2")
            code << "#define CDNA4_MOE_RECEIVER_COUNTS 1\n";
        if (mode == "2")
            code << "#define CDNA4_MOE_DIRECT_RECEIVER_COUNTS 1\n";
    }
    if (const char *flag = std::getenv("DG_MEGA_MOE_PARALLEL_BLOCK_TABLE"); flag && std::string(flag) == "1")
        code << "#define CDNA4_MOE_PARALLEL_BLOCK_TABLE 1\n";
    if (const char *flag = std::getenv("DG_MEGA_MOE_WAVE_PER_EXPERT"); flag && std::string(flag) == "1")
        code << "#define CDNA4_MOE_WAVE_PER_EXPERT 1\n";
    if (const char *flag = std::getenv("DG_MEGA_MOE_ROUTE_WAVE_RANGES"); flag && std::string(flag) == "1")
        code << "#define CDNA4_MOE_ROUTE_WAVE_RANGES 1\n";
    if (const char *early = std::getenv("DG_MEGA_MOE_EARLY_SHARED_L1"); early && std::string(early) == "1")
        code << "#define CDNA4_MOE_EARLY_SHARED_L1 1\n";
    if (const char *shards = std::getenv("DG_MEGA_MOE_GATHER_SHARDS")) {
        const std::string count(shards);
        TORCH_CHECK(count == "1" || count == "2" || count == "4", "gather shards must be 1, 2 or 4");
        code << "#define CDNA4_MOE_GATHER_SHARDS " << count << "\n";
    }
    if (const char *flag = std::getenv("DG_MEGA_MOE_CACHE_GATHER_TABLE"); flag && std::string(flag) == "1")
        code << "#define CDNA4_MOE_CACHE_GATHER_TABLE 1\n";
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
    TORCH_CHECK(((table_words + 7) / 8 * 8 + 128 * 8 + (transpose_rows ? 128 * (l.hidden / 128 + 1) : 0)) * 4 <= lds,
        "gather metadata exceeds allocated shared memory");
    int occupancy = 0;
    TORCH_CHECK(hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, module->kernels[0], 256, lds) ==
                        hipSuccess &&
                    occupancy >= 1 && blocks <= occupancy * cus,
                "Mega MoE grid cannot be simultaneously resident");
    const auto stream = c10::hip::getCurrentHIPStream(y.get_device()).stream();
    if (join >= 2)
        TORCH_CHECK(hipMemsetAsync(q.state, 0, state_words * sizeof(int), stream) == hipSuccess, "route state initialization failed");
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
