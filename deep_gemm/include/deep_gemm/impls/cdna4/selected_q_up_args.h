// Private-to-this-translation-unit state for the fixed-K Q-up specialization.
// General GEMM callers keep the public GemmArgs layout unchanged.
#pragma once
#include <deep_gemm/impls/cdna4/gemm_args.h>

namespace deep_gemm::cdna4 {
struct SelectedQUpArgs : GemmArgs {
    int *moe_tile_ready;
    int moe_comm_blocks, moe_comm_grouped;
    const int *moe_source_counts;
    int moe_rank, moe_ranks, moe_peer_order;
    const int *moe_release_control;
    int moe_single_release, moe_skip_compute;
};
} // namespace deep_gemm::cdna4

using deep_gemm::cdna4::SelectedQUpArgs;
