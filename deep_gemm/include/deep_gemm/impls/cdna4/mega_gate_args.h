// Host/device ABI of the CDNA4 MoE gate routing kernel (mega_gate.cuh).
#pragma once
#include <cstdint>

namespace deep_gemm::cdna4 {

struct MegaGateArgs {
  const float* logits;            // [num_tokens, ld_logits]: x @ weight^T
  const float* bias;              // [num_routed_experts] or null
  const float* image_bias;        // [num_routed_experts] for image tokens, or null
  const std::uint8_t* image_token_mask;
  const std::uint8_t* mask;       // inactive tokens get index -1 and weight 0
  const std::uint8_t* fix_routing_mask;  // tokens that keep the routing in unmapped_topk_idx
  const std::uint8_t* force_random;      // tokens routed to random experts
  const std::int32_t* to_physical_map;   // [logical experts, num_duplicate_experts] or null
  const std::int32_t* logical_count;     // [logical experts]
  std::int64_t* topk_idx;         // [num_tokens, num_topk + num_shared_experts]
  float* topk_weights;
  std::int64_t* unmapped_topk_idx;  // [num_tokens, num_topk] (row stride unmapped_stride) or null
  std::int64_t unmapped_stride;
  int ld_logits;
  int num_tokens, num_routed_experts, num_shared_experts, num_duplicate_experts, num_topk;
  std::uint32_t rank_idx;
  float routed_scaling_factor;
};

}  // namespace deep_gemm::cdna4
