#pragma once

#include <cstdint>

#include <uni_symm/device/fabric.hpp>

namespace deep_gemm::cdna4 {

// The regions of a Mega MoE symmetric buffer (byte offsets; every rank's buffer has this layout).
// Peers write the receive counts and lists (dispatch) and the combine slots; they read the inputs.
struct MegaMoeLayout {
    int num_ranks, num_experts, num_experts_per_rank, num_max_tokens, num_topk;
    int hidden, intermediate, shared_intermediate;
    // FP8 activations with packed UE8M0 scales of 32 values (else BF16)
    int with_sf, act_bytes;
    // routed rows of all local experts, each expert's rows starting at a multiple of `alignment`
    int alignment, max_rows;
    // pitches of the MN-major packed scales of the routed rows and of the local tokens
    int sf_pitch, shared_sf_pitch;

    // a routed row's record, pushed with the row: its K-major scale words (with_sf), the top-k
    // weight and the source (rank * max tokens + token) * topk + slot
    int record_ints;

    int64_t barrier;       // two 128-byte slots: counter (uint32), signals (int32) at byte +64
    int64_t send_counts;   // [experts] int32: this rank's rows for every expert (route; zeroed after)
    int64_t send_lists;    // [expert][max_tokens * topk] token-slot entries ordered by route positions
    int64_t positions;     // [max tokens * topk] int32: each slot's position among its expert's rows from this rank
    int64_t counts;        // [ranks][experts] int32: every rank's rows for every expert (published by each rank)
    int64_t dst_rows;      // [experts] int32: this rank's first row in every expert's rows (on the expert's rank)
    int64_t psum_ends;     // [experts per rank] int32: each local expert's end row (the GEMMs' PSUM layout)
    int64_t x, x_sf, topk_idx, topk_weights;   // inputs: x_sf [tokens][hidden / 128] int32 (K-major)
    int64_t shared_l1_sf, shared_l2_acts, shared_l2_sf;
    int64_t l1_acts, l1_records, l1_sf;
    int64_t l2_acts, l2_sf;
    int64_t output_rows;   // [max rows] peer BF16 destination pointers for direct L2 combine
    int64_t combine;       // [topk][max tokens][hidden] BF16: the routed results of every slot
    int64_t bytes;
};

#if !defined(__HIPCC_RTC__)
inline MegaMoeLayout make_mega_moe_layout(const int num_ranks, const int num_experts, const int num_max_tokens,
                                          const int num_topk, const int hidden, const int intermediate,
                                          const int shared_intermediate, const bool with_sf, const int alignment) {
    MegaMoeLayout l{};
    l.num_ranks = num_ranks, l.num_experts = num_experts, l.num_experts_per_rank = num_experts / num_ranks;
    l.num_max_tokens = num_max_tokens, l.num_topk = num_topk;
    l.hidden = hidden, l.intermediate = intermediate, l.shared_intermediate = shared_intermediate;
    l.with_sf = with_sf, l.act_bytes = with_sf ? 1 : 2, l.alignment = alignment;
    // a source token reaches a rank in at most min(topk, experts per rank) slots
    const int64_t routed = int64_t(num_max_tokens) * num_ranks *
                           (num_topk < l.num_experts_per_rank ? num_topk : l.num_experts_per_rank);
    l.max_rows = static_cast<int>((routed + int64_t(l.num_experts_per_rank) * (alignment - 1) + alignment - 1) /
                                  alignment * alignment);
    l.sf_pitch = (l.max_rows + 3) / 4 * 4, l.shared_sf_pitch = (num_max_tokens + 3) / 4 * 4;

    int64_t offset = 0;
    const auto take = [&](const int64_t bytes) {
        const int64_t at = offset;
        offset = (offset + bytes + 255) / 256 * 256;
        return at;
    };
    const int64_t er = l.num_experts_per_rank, t = num_max_tokens, rows = l.max_rows;
    const int sf = with_sf ? 1 : 0, shared = shared_intermediate > 0 ? 1 : 0;
    l.record_ints = ((with_sf ? hidden / 128 : 0) + 2 + 3) / 4 * 4;
    l.barrier = take(256);
    l.send_counts = take(int64_t(num_experts) * 4);
    l.positions = take(t * num_topk * 4);
    l.send_lists = take(int64_t(num_experts) * t * num_topk * 4);
    l.counts = take(int64_t(num_ranks) * num_experts * 4);
    l.dst_rows = take(int64_t(num_experts) * 4);
    l.psum_ends = take(er * 4);
    l.x = take(t * hidden * l.act_bytes);
    l.x_sf = take(sf * t * (hidden / 128) * 4);
    l.topk_idx = take(t * num_topk * 8);
    l.topk_weights = take(t * num_topk * 4);
    l.shared_l1_sf = take(sf * shared * int64_t(hidden / 128) * l.shared_sf_pitch * 4);
    l.shared_l2_acts = take(shared * t * shared_intermediate * l.act_bytes);
    l.shared_l2_sf = take(sf * shared * int64_t(shared_intermediate / 128) * l.shared_sf_pitch * 4);
    l.l1_acts = take(rows * hidden * l.act_bytes);
    l.l1_records = take(rows * l.record_ints * 4);
    l.l1_sf = take(sf * int64_t(hidden / 128) * l.sf_pitch * 4);
    l.l2_acts = take(rows * intermediate * l.act_bytes);
    l.l2_sf = take(sf * int64_t(intermediate / 128) * l.sf_pitch * 4);
    l.output_rows = take(rows * 8);
    l.combine = take(int64_t(num_topk) * t * hidden * 2);
    l.bytes = offset;
    return l;
}
#endif

// A Mega MoE launch: this rank's view of the buffers and the call's local tensors
struct MegaMoeArgs {
    uni_symm::SymmView<uni_symm::kMaxRanks> view;
    MegaMoeLayout layout;
    int num_tokens;              // this rank's tokens of the call
    const uint16_t* l1_out;      // [max rows][2 intermediate] BF16, gate and up interleaved by 8 columns
    const uint16_t* l2_out;      // [max rows][hidden] BF16
    const uint16_t* shared_l1_out;  // [tokens][2 shared intermediate] BF16
    const uint16_t* shared_out;  // [tokens][hidden] BF16
    uint16_t* y;                 // [tokens][hidden] BF16
    int* stats;                  // [experts per rank] received rows, accumulated (or null)
    float activation_clamp;      // +inf: none
    int fast_math;
    int ordered_push;
    int direct_combine;
    int barrier_slot; // independent dispatch/combine phases on different streams
    int expert_begin, expert_count; // zero count means all local experts
    int shared_part;             // finish / swiglu: the shared expert's rows (else the routed rows)
};

}  // namespace deep_gemm::cdna4
