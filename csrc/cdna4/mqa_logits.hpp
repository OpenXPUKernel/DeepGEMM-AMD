#pragma once

#include <algorithm>
#include <sstream>
#include <string>

#include "jit_runtime.hpp"
#include "deep_gemm/impls/cdna4/mqa_logits_args.h"

namespace deep_gemm::cdna4 {

// Kernel shape of mqa_logits.cuh: L lane groups per token, U groups of 4 / L tokens per wave
struct MqaLogitsPlan {
    MqaLogitsFormat format;
    MqaLogitsSource source;
    int head_dim, num_heads, lanes_per_token, token_groups;
    // columns per work item (MqaLogitsConfig::CHUNK)
    int chunk = 2048;

    int tokens_per_wave() const { return token_groups * 4 / lanes_per_token; }
    int tiles() const {
        const int rows = 4 * lanes_per_token;
        return token_groups * ((num_heads + rows - 1) / rows);
    }
    // MqaLogitsConfig::LDS_BYTES
    int lds_bytes() const {
        const bool fp4 = format == MqaLogitsFormat::MxFp4;
        const int halves = fp4 || head_dim <= 64 ? 1 : 2;
        const bool sparse = source == MqaLogitsSource::Sparse || source == MqaLogitsSource::PagedSparse;
        const int wave_chunk = chunk / 4;
        const int table =
            source == MqaLogitsSource::Contiguous ? 0 : 4 * 4 * (sparse ? wave_chunk / 8 + 1 : wave_chunk / 32 + 1);
        return tiles() * (halves * 1024 + (format == MqaLogitsFormat::Fp8 ? 0 : 256)) + table;
    }
};

// Up to 16 MFMA tiles (their Q operands stay in registers). Contiguous KV: one lane group per
// token and as many tokens per wave as fit; paged KV: the tokens of one request share the wave, a
// token spread over four lane groups (decoding has few tokens per request); sparse KV: one token.
inline MqaLogitsPlan plan_mqa_logits(MqaLogitsFormat format, MqaLogitsSource source, int head_dim, int num_heads,
                                     int tokens_per_request = 4) {
    MqaLogitsPlan plan{format, source, head_dim, num_heads, 1, 1};
    if (source == MqaLogitsSource::Contiguous) {
        const int tiles_per_group = (num_heads + 3) / 4;
        plan.token_groups = std::max(1, 16 / tiles_per_group);
    } else {
        plan.lanes_per_token = 4;
        const int tiles_per_token = (num_heads + 15) / 16;
        plan.token_groups =
            source == MqaLogitsSource::Paged ? std::clamp(std::min(tokens_per_request, 16 / tiles_per_token), 1, 8) : 1;
    }
    return plan;
}

inline const char *mqa_format_name(MqaLogitsFormat format) {
    return format == MqaLogitsFormat::Fp8 ? "Fp8" : format == MqaLogitsFormat::MxFp8 ? "MxFp8" : "MxFp4";
}

// kernel symbol of a source (the names of the CUDA kernels, for profiling)
inline const char *mqa_kernel_name(MqaLogitsSource source) {
    switch (source) {
    case MqaLogitsSource::Contiguous:
        return "dg_cdna4_mqa_logits";
    case MqaLogitsSource::Paged:
        return "dg_cdna4_paged_mqa_logits";
    case MqaLogitsSource::Sparse:
        return "dg_cdna4_sparse_mqa_logits";
    case MqaLogitsSource::PagedSparse:
        return "dg_cdna4_paged_sparse_mqa_logits";
    }
    return "dg_cdna4_mqa_logits";
}

inline const char *mqa_source_name(MqaLogitsSource source) {
    switch (source) {
    case MqaLogitsSource::Contiguous:
        return "Contiguous";
    case MqaLogitsSource::Paged:
        return "Paged";
    case MqaLogitsSource::Sparse:
        return "Sparse";
    case MqaLogitsSource::PagedSparse:
        return "PagedSparse";
    }
    return "unknown";
}

// grid: (waves, chunk rounds); a wave takes chunks y, y + rounds, ... of its tokens' columns.
// Chunks of 2048 columns, or shorter ones (more work items) while that leaves a compute unit's
// SIMDs without work.
inline int launch_mqa_logits(
    MqaLogitsArgs &args, MqaLogitsPlan plan, int waves, int max_columns, hipStream_t stream, int column_rounds = 0, int sparse_mode = 0) {
    if (waves <= 0 || max_columns <= 0)
        return 0;
    int policy = 0;
    if (const char *mode = std::getenv("DG_MQA_FIVE_POSITION");
        mode && column_rounds > 0 && args.run_starts && plan.source == MqaLogitsSource::Paged && plan.format == MqaLogitsFormat::MxFp4 &&
        plan.num_heads == 32 && plan.head_dim == 128 && (args.page_kv == 64 || args.page_kv == 128)) {
        if (std::string(mode) == "page")
            policy = 1;
        if (std::string(mode) == "split" && plan.token_groups == 3)
            policy = 2;
        if (std::string(mode) == "five" && plan.token_groups == 5)
            policy = 3;
    }
    int registers = 0, vector_output = 0;
    if (const char *io = std::getenv("DG_MQA_PAGED_IO"); io && std::string(io) != "0" && args.run_starts && column_rounds > 0 &&
                                                         plan.source == MqaLogitsSource::Paged && plan.format == MqaLogitsFormat::MxFp4 &&
                                                         plan.num_heads == 32 && plan.head_dim == 128) {
        char rest = 0;
        TORCH_CHECK(std::sscanf(io, "%d,%d%c", &registers, &vector_output, &rest) == 2 && registers >= 0 && registers <= 2 &&
                        vector_output >= 0 && vector_output <= 1,
            "invalid paged MQA query storage and write plan");
    }
    if (sparse_mode == 3) {
        registers = 1;
        vector_output = 1;
    }
    if (policy >= 2)
        waves = ((args.num_q + 4) / 5) * (policy == 2 ? 2 : 1);
    // workgroups of four waves, a quarter of a chunk each: two waves per SIMD at least
    const int slots = 2 * num_compute_units();
    plan.chunk = 2048;
    while (plan.chunk > 256 && int64_t(waves) * ((max_columns + plan.chunk - 1) / plan.chunk) < slots)
        plan.chunk /= 2;
    std::ostringstream name;
    name << "mqa_logits:" << mqa_format_name(plan.format) << ":" << mqa_source_name(plan.source) << ":" << plan.head_dim
         << ":" << plan.num_heads << ":" << plan.lanes_per_token << ":" << plan.token_groups << ":" << plan.chunk;
    if (args.block_scores)
        name << ":blockmax8";
    if (sparse_mode)
        name << ":sparse_mode=" << sparse_mode << ":page=" << args.page_kv << ":stride=" << args.page_stride;
    if (registers || vector_output)
        name << ":regs=" << registers << ":store=" << vector_output;
    if (policy)
        name << ":page=" << args.page_kv << ":stride=" << args.page_stride << ":five=" << policy;
    // two waves per SIMD (256 VGPRs each), unless the LDS of a workgroup allows only one (a
    // higher bound would make the compiler spill)
    const int waves_per_simd = std::clamp(163840 / std::max(plan.lds_bytes(), 1), 1, 2);
    const auto module = jit_runtime.get_module(
        name.str(),
        [&] {
            std::ostringstream source;
            if (registers)
                source << "#define CDNA4_MQA_REGISTER_Q " << registers << "\n";
            if (vector_output)
                source << "#define CDNA4_MQA_VECTOR_OUTPUT 1\n";
            if (args.block_scores)
                source << "#define CDNA4_MQA_BLOCK_MAX 1\n";
            if (sparse_mode >= 2)
                source << "#define CDNA4_MQA_SPARSE_STARTS 1\n";
            if (policy)
                source << "#define CDNA4_MQA_PAGE_FAST 1\n";
            if (policy >= 2)
                source << "#define CDNA4_MQA_FIVE_GROUP 1\n";
            source << "#include <deep_gemm/impls/cdna4/mqa_logits.cuh>\n"
                      "extern \"C\" __global__ __attribute__((amdgpu_flat_work_group_size(256,256),amdgpu_waves_per_eu("
                   << waves_per_simd << "))) void " << mqa_kernel_name(plan.source) << "(MqaLogitsArgs p){";
            auto cfg = [&](int groups) {
                std::ostringstream t;
                t << "MqaLogitsConfig<MqaLogitsFormat::" << mqa_format_name(plan.format)
                  << ",MqaLogitsSource::" << mqa_source_name(plan.source) << "," << plan.head_dim << "," << plan.num_heads << ","
                  << plan.lanes_per_token << "," << groups << "," << plan.chunk << ">";
                return t.str();
            };
            if (sparse_mode)
                source << "p.sparse_block_kv=8;";
            if (policy || sparse_mode)
                source << "p.page_kv=" << args.page_kv << ";p.page_stride=" << args.page_stride << "LL;";
            const char *out = plan.format == MqaLogitsFormat::Fp8 ? "float" : "std::uint16_t";
            if (policy >= 2)
                source << "mqa_five_positions<" << cfg(plan.token_groups) << "," << cfg(2) << "," << out << ">(p);}";
            else
                source << "mqa_logits_impl<" << cfg(plan.token_groups) << "," << out << ">(p);}";
            return source.str();
        },
        {mqa_kernel_name(plan.source)},
        // MFMA results go to VGPRs: the weighted ReLUs read them right away (AGPRs cost a move each).
        // Without NaN semantics a ReLU is one v_max (IEEE maxnum quiets its inputs first).
        {"-mllvm", "-amdgpu-mfma-vgpr-form", "-fno-honor-nans"});
    const int chunks = (max_columns + plan.chunk - 1) / plan.chunk;
    const int rounds = std::clamp(column_rounds > 0 ? column_rounds : 16 * num_compute_units() / waves, 1, chunks);
    return JitRuntime::launch_kernel(module->kernels[0], args, dim3(waves, rounds, 1), dim3(256, 1, 1),
        plan.lds_bytes() - (registers == 1 ? plan.tiles() * 1280 : 0), stream);
}

} // namespace deep_gemm::cdna4
