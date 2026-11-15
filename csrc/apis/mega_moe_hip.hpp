#pragma once
#include "../jit_kernels/impls/cdna4_fp8_fp4_mega_moe.hpp"
namespace {
// Mega MoE (mega_moe.cuh): the regions of the symmetric buffer, for the Python views
constexpr int kMegaMoeAlignment = 256;

py::dict cdna4_mega_moe_layout(int num_ranks, int num_experts, int num_max_tokens, int num_topk, int hidden,
                               int intermediate, int shared_intermediate, bool with_sf) {
    TORCH_CHECK(num_ranks > 0 && num_ranks <= uni_symm::kMaxRanks && num_experts % num_ranks == 0,
                "the experts must divide over at most ", uni_symm::kMaxRanks, " ranks");
    TORCH_CHECK(hidden % 128 == 0 && intermediate % 128 == 0 && shared_intermediate % 128 == 0,
                "hidden and intermediate sizes must be multiples of 128");
    TORCH_CHECK(num_topk > 0 && num_max_tokens > 0, "invalid token counts");
    TORCH_CHECK(int64_t(num_ranks) * num_experts <= 16384, "ranks x experts must be at most 16384 (the row layout "
                                                           "is computed from the count matrix in LDS)");
    const auto l =
        deep_gemm::cdna4::make_mega_moe_layout(num_ranks, num_experts, num_max_tokens, num_topk, hidden, intermediate,
                                               shared_intermediate, with_sf, kMegaMoeAlignment);
    TORCH_CHECK(int64_t(num_ranks) * num_max_tokens * num_topk < (int64_t(1) << 32), "too many routed rows");
    py::dict d;
    d["bytes"] = l.bytes, d["max_rows"] = l.max_rows, d["sf_pitch"] = l.sf_pitch,
    d["shared_sf_pitch"] = l.shared_sf_pitch;
    d["x"] = l.x, d["x_sf"] = l.x_sf, d["topk_idx"] = l.topk_idx, d["topk_weights"] = l.topk_weights;
    d["shared_l1_sf"] = l.shared_l1_sf, d["shared_l2_acts"] = l.shared_l2_acts, d["shared_l2_sf"] = l.shared_l2_sf;
    d["l1_acts"] = l.l1_acts, d["l1_sf"] = l.l1_sf, d["l2_acts"] = l.l2_acts, d["l2_sf"] = l.l2_sf;
    d["positions"] = l.positions, d["send_counts"] = l.send_counts, d["psum_ends"] = l.psum_ends;
    d["l1_records"] = l.l1_records, d["record_ints"] = l.record_ints;
    d["output_rows"] = l.output_rows, d["combine"] = l.combine;
    return d;
}

// One Mega MoE call (see mega_moe.cuh): the communication kernels around the routed grouped GEMMs
// and the shared expert's dense GEMMs, in stream order. The Python wrapper validates the tensors.
torch::Tensor cdna4_prepare_moe_weight_panels(const torch::Tensor &weight) {
    TORCH_CHECK(weight.is_cuda() && is_packed_fp4(weight) && weight.dim() == 3 && weight.is_contiguous(),
                "MoE panel preparation requires contiguous packed-FP4 [E,N,K/2] weights");
    TORCH_CHECK(weight.size(0) > 0 && weight.size(1) > 0 && weight.size(1) % 256 == 0 && weight.size(2) > 0 &&
                    weight.size(2) % 64 == 0 && weight.size(1) <= std::numeric_limits<int>::max() &&
                    weight.size(2) <= std::numeric_limits<int>::max(),
                "MoE panel preparation requires positive N256/K128-aligned dimensions");
    const c10::hip::HIPGuardMasqueradingAsCUDA guard(weight.get_device());
    const auto stream = c10::hip::getCurrentHIPStream(weight.get_device()).stream();
    hipStreamCaptureStatus capture;
    TORCH_CHECK(hipStreamIsCapturing(stream, &capture) == hipSuccess && capture == hipStreamCaptureStatusNone,
                "prepare MoE panel weights before graph capture, not inside a complete MoE call");
    auto output = torch::empty_like(weight);
    struct Args {
        const void *source;
        void *destination;
        int64_t chunks;
        int n, k_bytes;
    };
    static_assert(sizeof(Args) == 32);
    Args args{weight.data_ptr(), output.data_ptr(), weight.numel() / 16, int(weight.size(1)), int(weight.size(2))};
    auto module = jit_runtime.get_module(
        "moe_weight_panels:v1",
        [] {
            return std::string("#include <deep_gemm/impls/cdna4_mega_moe_weights.cuh>\n"
                               "extern \"C\" __global__ __launch_bounds__(256) void "
                               "dg_moe_prepare_panels(MoePanelPermutationArgs p){moe_permute_weight_panels(p);}\n");
        },
        {"dg_moe_prepare_panels"});
    const int blocks = int(std::min<int64_t>((args.chunks + 255) / 256, 4 * deep_gemm::cdna4::num_compute_units()));
    TORCH_CHECK(deep_gemm::cdna4::JitModule::launch_function(module->kernels[0], dim3(blocks), dim3(256), 0, stream,
                                                             false, args) == hipSuccess,
                "MoE panel preparation launch failed");
    return output;
}

void cdna4_mega_moe(const torch::Tensor &y, const torch::Tensor &l1_weights, const OptTensor &l1_weights_sf,
                    const torch::Tensor &l2_weights, const OptTensor &l2_weights_sf, const OptTensor &shared_l1_weights,
                    const OptTensor &shared_l1_weights_sf, const OptTensor &shared_l2_weights,
                    const OptTensor &shared_l2_weights_sf, const OptTensor &stats, const torch::Tensor &buffer,
                    const std::vector<int64_t> &buffer_ptrs, int rank, int num_max_tokens, int num_experts,
                    int num_topk, double activation_clamp, bool fast_math, bool resident_weight_panels = false,
                    bool reference = false) {
    const bool with_sf = l1_weights_sf.has_value(), shared = shared_l1_weights.has_value();
    const int tokens = y.size(0), hidden = y.size(1), ranks = static_cast<int>(buffer_ptrs.size());
    const int intermediate = l1_weights.size(1) / 2;
    TORCH_CHECK(!reference || !resident_weight_panels, "reference lowering takes ordinary weights");
    const int shared_intermediate = shared ? shared_l2_weights->size(1) : 0;
    const auto l =
        deep_gemm::cdna4::make_mega_moe_layout(ranks, num_experts, num_max_tokens, num_topk, hidden, intermediate,
                                               shared_intermediate, with_sf, kMegaMoeAlignment);
    TORCH_CHECK(buffer.nbytes() >= static_cast<size_t>(l.bytes), "the symmetric buffer is too small");
    TORCH_CHECK(rank >= 0 && rank < ranks && buffer_ptrs[rank] == reinterpret_cast<int64_t>(buffer.data_ptr()),
                "buffer_ptrs[rank] must be this rank's buffer");
    const c10::hip::HIPGuardMasqueradingAsCUDA device_guard(y.get_device());
    const auto stream = c10::hip::getCurrentHIPStream(y.get_device()).stream();
    const int cus = deep_gemm::cdna4::num_compute_units();

    deep_gemm::cdna4::MegaMoeArgs args{};
    args.view = uni_symm::SymmView<uni_symm::kMaxRanks>::make(buffer_ptrs, rank);
    args.layout = l;
    args.num_tokens = tokens;
    args.y = reinterpret_cast<std::uint16_t *>(y.data_ptr());
    args.stats = stats.has_value() ? stats->data_ptr<int>() : nullptr;
    args.activation_clamp = static_cast<float>(activation_clamp);
    args.fast_math = fast_math;
    const auto bf16 = y.options().dtype(torch::kBFloat16);
    const auto l1_out = torch::empty({l.max_rows, 2 * intermediate}, bf16);
    const auto l2_out = torch::empty({l.max_rows, hidden}, bf16);
    const bool shared_tokens = shared && tokens > 0;
    const auto shared_l1_out = shared_tokens ? torch::empty({tokens, 2 * shared_intermediate}, bf16) : torch::Tensor();
    const auto shared_out = shared_tokens ? torch::empty({tokens, hidden}, bf16) : torch::Tensor();
    const auto u16 = [](const torch::Tensor &t) { return reinterpret_cast<const std::uint16_t *>(t.data_ptr()); };
    args.l1_out = u16(l1_out), args.l2_out = u16(l2_out);
    args.shared_l1_out = shared_tokens ? u16(shared_l1_out) : nullptr;
    args.shared_out = shared_tokens ? u16(shared_out) : nullptr;

    const bool native = !reference && with_sf && is_packed_fp4(l1_weights) && is_packed_fp4(l2_weights) &&
                        hidden >= 512 && intermediate >= 512 && hidden % 256 == 0 && intermediate % 256 == 0 &&
                        (!shared || shared_intermediate == intermediate);
    TORCH_CHECK(native || !resident_weight_panels, "prepared panels require aligned FP8 x FP4 MoE");
    if (native) {
        deep_gemm::cdna4::mega_moe_host::launch(args, buffer, y, l1_weights, *l1_weights_sf, l2_weights, *l2_weights_sf,
                                                shared_l1_weights, shared_l1_weights_sf, shared_l2_weights,
                                                shared_l2_weights_sf, l1_out, l2_out, shared_l1_out, shared_out,
                                                resident_weight_panels);
        return;
    }
    const int finish_lds = l.num_experts_per_rank * 4;
    const auto module = jit_runtime.get_module(
        "cdna4_mega_moe_reference",
        [] {
            return R"(
#include <deep_gemm/impls/cdna4_mega_moe_common.cuh>
#define K(name, body) extern "C" __global__ __launch_bounds__(256) void dg_cdna4_mega_moe_##name(MegaMoeArgs p) { body; }
K(route_deterministic,mega_moe_route_deterministic_impl(p))
K(publish_barrier,mega_moe_barrier_impl<true>(p))
K(barrier,mega_moe_barrier_impl<false>(p))
K(push_ordered,mega_moe_push_ordered_impl(p))
K(finish,mega_moe_finish_impl(p))
K(swiglu_fp8,mega_moe_swiglu_impl<true>(p))
K(swiglu_bf16,mega_moe_swiglu_impl<false>(p))
K(combine,mega_moe_combine_impl(p))
K(reduce,mega_moe_reduce_impl(p))
)";
        },
        {"dg_cdna4_mega_moe_route_deterministic", "dg_cdna4_mega_moe_publish_barrier", "dg_cdna4_mega_moe_barrier",
         "dg_cdna4_mega_moe_push_ordered", "dg_cdna4_mega_moe_finish", "dg_cdna4_mega_moe_swiglu_fp8",
         "dg_cdna4_mega_moe_swiglu_bf16", "dg_cdna4_mega_moe_combine", "dg_cdna4_mega_moe_reduce"});
    enum {
        kRouteDeterministic,
        kPublishBarrier,
        kBarrier,
        kPushOrdered,
        kFinish,
        kSwigluFp8,
        kSwigluBf16,
        kCombine,
        kReduce
    };
    const auto launch = [&](const deep_gemm::cdna4::MegaMoeArgs &a, int kernel, int blocks, int threads, int lds,
                            const char *name, hipStream_t on) {
        auto kernel_args = a;
        const int error = deep_gemm::cdna4::JitRuntime::launch_kernel(module->kernels[kernel], kernel_args,
                                                                      dim3(blocks, 1, 1), dim3(threads, 1, 1), lds, on);
        TORCH_CHECK(error == 0, "CDNA4 Mega MoE ", name, " launch failed with code ", error);
    };
    const int experts = l.num_experts_per_rank, grid = 4 * cus;
    // the push and combine are bound by the fabric: an eighth of the compute units moves as much
    const int comm_grid = std::max(8, cus / 8);
    auto *base = static_cast<std::uint8_t *>(buffer.data_ptr());
    const auto act_dtype = with_sf ? torch::kFloat8_e4m3fn : torch::kBFloat16;
    const auto view = [&](int64_t offset, std::vector<int64_t> sizes, torch::ScalarType dtype) {
        return torch::from_blob(base + offset, sizes, buffer.options().dtype(dtype));
    };
    const auto scales = [&](int64_t offset, int64_t rows, int64_t k, int pitch) -> OptTensor {
        if (!with_sf)
            return std::nullopt;
        return torch::from_blob(base + offset, {rows, k / 128}, {1, pitch}, buffer.options().dtype(torch::kInt32));
    };
    // the shared expert on a side stream: it needs only this rank's tokens, so it overlaps the
    // dispatch and the routed experts; the reduction joins it
    static std::vector<std::pair<hipEvent_t, hipEvent_t>> fork_join;
    hipEvent_t join = nullptr;
    if (shared_tokens) {
        const int device = y.get_device();
        if (fork_join.size() <= static_cast<size_t>(device))
            fork_join.resize(device + 1, {nullptr, nullptr});
        auto &[fork_event, join_event] = fork_join[device];
        if (fork_event == nullptr) {
            TORCH_CHECK(hipEventCreateWithFlags(&fork_event, hipEventDisableTiming) == hipSuccess &&
                        hipEventCreateWithFlags(&join_event, hipEventDisableTiming) == hipSuccess);
        }
        join = join_event;
        const auto side = at::hip::getStreamFromPoolMasqueradingAsCUDA(false, device);
        TORCH_CHECK(hipEventRecord(fork_event, stream) == hipSuccess &&
                    hipStreamWaitEvent(side.stream(), fork_event, 0) == hipSuccess);
        shared_l1_out.record_stream(side);
        shared_out.record_stream(side);
        const at::hip::HIPStreamGuardMasqueradingAsCUDA side_guard(side);
        auto shared_args = args;
        shared_args.shared_part = 1;
        if (with_sf)
            launch(shared_args, kFinish, std::min(ceil_div(tokens, 4), grid), 256, 0, "finish", side.stream());
        cdna4_fp8_gemm_nt(view(l.x, {tokens, hidden}, act_dtype),
                          scales(l.shared_l1_sf, tokens, hidden, l.shared_sf_pitch), *shared_l1_weights,
                          shared_l1_weights_sf, shared_l1_out, std::nullopt, 1.0, 32, 32);
        launch(shared_args, with_sf ? kSwigluFp8 : kSwigluBf16, std::min(ceil_div(tokens, 4), grid), 256, 0, "swiglu",
               side.stream());
        cdna4_fp8_gemm_nt(view(l.shared_l2_acts, {tokens, shared_intermediate}, act_dtype),
                          scales(l.shared_l2_sf, tokens, shared_intermediate, l.shared_sf_pitch), *shared_l2_weights,
                          shared_l2_weights_sf, shared_out, std::nullopt, 1.0, 32, 32);
        TORCH_CHECK(hipEventRecord(join_event, side.stream()) == hipSuccess);
    }

    args.ordered_push = 1;
    args.direct_combine = 0;
    const auto cs = stream, bs = stream, dispatch_stream = stream;
    const int per_chunk = experts;
    const int table_lds = (3 * ranks * experts + ranks + 2 * experts + 2) * 4;
    launch(args, kRouteDeterministic, num_experts, 256, 0, "route", dispatch_stream);
    launch(args, kPublishBarrier, 1, 256, ranks * num_experts * 4, "barrier", dispatch_stream);
    const auto ends = view(l.psum_ends, {experts}, torch::kInt32);
    for (int begin = 0; begin < experts; begin += per_chunk) {
        auto part = args;
        part.expert_begin = begin;
        part.expert_count = std::min(per_chunk, experts - begin);

        launch(part, kPushOrdered, comm_grid, 256, (ranks * (experts + 1) + 1) * 4, "dispatch", dispatch_stream);
        launch(part, kBarrier, 1, 128, 0, "barrier", dispatch_stream);

        {

            if ((((with_sf))))
                launch(part, kFinish, grid, 256, finish_lds, "finish", cs);
            cdna4_m_grouped_fp8_gemm_contiguous_impl(
                view(l.l1_acts, {l.max_rows, hidden}, act_dtype), scales(l.l1_sf, l.max_rows, hidden, l.sf_pitch),
                l1_weights, l1_weights_sf, l1_out, ends, 32, 32, true, false, kMegaMoeAlignment, 0);
            launch(part, with_sf ? kSwigluFp8 : kSwigluBf16, grid, 256, experts * 4, "swiglu", cs);
            cdna4_m_grouped_fp8_gemm_contiguous_impl(view(l.l2_acts, {l.max_rows, intermediate}, act_dtype),
                                                     scales(l.l2_sf, l.max_rows, intermediate, l.sf_pitch), l2_weights,
                                                     l2_weights_sf, l2_out, ends, 32, 32, true, false,
                                                     kMegaMoeAlignment, 0);
        }

        launch(part, kCombine, comm_grid, 256, table_lds, "combine", bs);
    }

    args.barrier_slot = 0;
    launch(args, kBarrier, 1, 128, 0, "barrier", stream);
    if (join != nullptr)
        TORCH_CHECK(hipStreamWaitEvent(stream, join, 0) == hipSuccess);
    if (tokens > 0)
        launch(args, kReduce, std::min(ceil_div(tokens, 4), grid), 256, 0, "reduce", stream);
}

} // namespace
