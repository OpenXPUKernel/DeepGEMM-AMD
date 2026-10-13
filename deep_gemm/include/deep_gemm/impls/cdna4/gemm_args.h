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
  int k_alignment;
  // row alignment of the group starts of a PSUM layout
  int m_alignment;
  // small-M kernel, masked and PSUM layouts: workgroups per group and column block, each taking
  // every group_m_slots-th m tile of the group
  int group_m_slots;
  int zero_padding;
  // KBatched: K tiles per batch
  int k_batch_tiles;
  std::int64_t a_group_stride, b_group_stride;
  std::int64_t sfa_group_stride, sfb_group_stride;
  std::int64_t d_group_stride;
};
} // namespace deep_gemm::cdna4
