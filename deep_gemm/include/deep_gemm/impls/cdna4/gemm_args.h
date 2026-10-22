// Host/device ABI shared by generated CDNA4 HIP kernels.
// Keep this header free of HIP intrinsics so host planners and device JIT code
// agree on layout without pulling in a runtime or compiler specific header.
#pragma once
#include <cstdint>

namespace deep_gemm::cdna4 {

// How output tiles map onto groups (see scheduler.cuh)
enum class GemmType : int {
  Normal = 0,
  // A [M, K], B [G, N, K]; grouped_layout[m] is the group of row m (-1: padding)
  MGroupedContiguous = 1,
  // grouped_layout[g] is the end row of group g; group g starts at align(end[g - 1], m_alignment)
  MGroupedContiguousPsum = 2,
  // A [G, max_m, K], B [G, N, K], D [G, max_m, N]; grouped_layout[g] is the row count of group g
  MGroupedMasked = 3,
  // D [G, M, N]; group g covers grouped_layout[g] (padded) K values
  KGroupedContiguous = 4,
  // grouped_layout[g] is the end K of group g; group g starts at align(end[g - 1], k_alignment)
  KGroupedContiguousPsum = 5,
  // D = sum_b A[b] @ B[b]^T for K-major A [batches, M, k], B [batches, N, k]: one GEMM whose K
  // (batches x k) runs through the batches, k_batch_tiles K tiles each (group strides: batches)
  KBatched = 6,
};

// Element type of an operand. FP4 (E2M1) values are packed two per byte, the lower K index in the
// low nibble. BF16 operands come in pairs and have no scales.
enum class OperandType : int { Fp8 = 0, Fp4 = 1, Bf16 = 2 };

struct GemmArgs {
  const std::uint8_t* A;
  const std::uint8_t* B;
  const std::int32_t* sfa;
  const std::int32_t* sfb;
  void* D;
  const void* C;
  int M, N, K;
  int lda, ldb;
  int lda_sf, ldb_sf;
  int ldd;
  float alpha;
  int splits;
  float* ws;
  int ldw;
  // Grouped GEMMs. Group strides are in elements of their tensors (bytes for
  // FP8 operands, int32 words for packed scales, outputs for D and C).
  const std::int32_t* grouped_layout;
  int num_groups;
  // Optional expert window for a pipelined PSUM GEMM; zero count means all groups.
  int group_begin, group_count;
  int k_alignment;
  // row alignment of the group starts of a PSUM layout
  int m_alignment;
  // small-M kernel, masked and PSUM layouts: workgroups per group and column block, each taking
  // every group_m_slots-th m tile of the group
  int group_m_slots;
  int zero_padding;
  // Optional fused Mega MoE L1 epilogue. Gate/up columns are interleaved by eight.
  std::uint8_t* moe_acts;
  std::int32_t* moe_sf;
  const std::int32_t* moe_records;
  int moe_record_stride, moe_weight_word, moe_sf_pitch;
  float moe_clamp;
  int moe_fast_math;
  const int64_t* moe_output_rows;
  int* moe_tile_ready;
  int moe_comm_blocks;
  int moe_comm_grouped;
  const int* moe_source_counts;
  int moe_rank, moe_ranks, moe_peer_order;
  const int* moe_release_control;
  int moe_single_release;
  int moe_skip_compute; // diagnostic: same resident kernel, completed local GEMM input
  // KBatched: K tiles per batch
  int k_batch_tiles;
  std::int64_t a_group_stride, b_group_stride;
  std::int64_t sfa_group_stride, sfb_group_stride;
  std::int64_t d_group_stride;
};

// D[s] = A @ B^T and sqr_sum[s] = sum of squares of A's rows over K split s of splits
// (hc_prenorm_gemm.cuh): BF16 A [M, K], FP32 B [N, K], FP32 D [splits, M, N] and
// sqr_sum [splits, M] (a workspace of partials when the result is reduced)
// Mega mHC (mega_mhc.cuh): BF16 [tokens, hidden] and [tokens, 4, hidden] activations, FP32 mixes
struct MegaMhcArgs {
    const std::uint16_t* x;
    const std::uint16_t* residual;
    const float* post_mix;
    const float* comb_res_mix;
    const float* shifted_prev_mix;  // null: normal mHC
    const float* fn;                // [24, 4 hidden]
    const float* mix_scales;
    const float* mix_bases;
    const std::uint16_t* rmsnorm_weight;
    std::uint16_t* new_residual;
    float* new_prev_mix;            // null: normal mHC
    float* new_post_mix;
    float* new_comb_res_mix;
    // shifted: the norm input from the residual kernel (y_bf16 or a scratch), replaced by y when
    // write_y_bf16; normal: y_bf16
    std::uint16_t* norm_bf16;
    int write_y_bf16;
    std::uint8_t* y_fp8;            // null: no FP8 y
    std::int32_t* y_sf;             // packed UE8M0 of 128 values per word
    std::int64_t y_sf_stride_token, y_sf_stride_word;
    std::int32_t* y_shared_sf;      // null, or the Mega MoE shared-expert rows of shared_sf_block_m tokens
    std::int64_t y_shared_sf_stride_word;
    int shared_sf_block_m;
    float* mix_partials;            // [token blocks, splits, 64, 24]
    float* sqr_partials;            // [token blocks, splits, 64]
    float* prev_mix;                // the new previous mix: new_prev_mix (shifted) or [tokens, 4] (normal)
    int num_tokens, hidden, splits;
    float hc_norm_eps, hc_pre_eps, hc_post_scale, sinkhorn_eps;
    int num_sinkhorn_iters;
    float rmsnorm_eps, rmsnorm_scale;
};

struct HcPrenormGemmArgs {
  const std::uint16_t* a;
  const float* b;
  float* d;
  float* sqr_sum;
  int m, n, k;
  int lda, ldb;  // elements
  int splits;
  // the reduction of a workspace (splits partials) into final_d [M, N] and final_sqr_sum [M]
  float* final_d;
  float* final_sqr_sum;
};
} // namespace deep_gemm::cdna4
