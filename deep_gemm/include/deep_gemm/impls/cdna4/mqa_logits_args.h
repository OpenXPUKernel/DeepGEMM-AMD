// Host/device ABI of the CDNA4 MQA logits kernels (mqa_logits.cuh).
#pragma once
#include <cstdint>

namespace deep_gemm::cdna4 {

// Element formats of Q and KV
enum class MqaLogitsFormat : int {
  // E4M3 Q without scales, E4M3 KV with one float scale per token; float weights and logits
  Fp8 = 0,
  // E4M3 Q and KV with UE8M0 scales of 32 values packed into one int32 per head / token; BF16
  // weights and logits
  MxFp8 = 1,
  // packed E2M1 (two values per byte, the lower index in the low nibble), otherwise as MxFp8
  MxFp4 = 2,
};

// Where the KV rows of a q token come from, and where its logits go
enum class MqaLogitsSource : int {
  // kv [num_kv, D]: token i takes KV [ks[i], ke[i]) into logits[i, kv - ks[i]]
  Contiguous = 0,
  // paged KV cache: token i takes positions [0, context_lens[i]) into logits[i, position]
  Paged = 1,
  // kv [num_kv, D]: token i takes the blocks of its sparse index row into logits[i, column]
  Sparse = 2,
  // paged KV cache, sparse index rows of logical blocks
  PagedSparse = 3,
};

struct MqaLogitsArgs {
  const std::uint8_t* q;       // [num_q, H, D] (D / 2 bytes per row with FP4)
  const std::int32_t* q_sf;    // [num_q, H] (MX formats)
  const std::uint8_t* kv;      // [num_kv, D] or the base of the paged cache
  const void* kv_sf;           // [num_kv] int32 (MX) or float (Fp8), unless paged
  const void* weights;         // [num_q, H] float (Fp8) or BF16 (MX), row stride ld_weights
  void* logits;                // float (Fp8) or BF16 (MX), row stride ld_logits
  std::int64_t ld_logits;
  int ld_weights;
  int num_q, num_kv;
  // Contiguous / Sparse
  const std::int32_t* ks;
  const std::int32_t* ke;
  // optional [ceil(num_q / tokens per wave), 2] KV range of each group of tokens (Contiguous)
  const std::int32_t* group_ranges;
  // Paged / PagedSparse: a page holds page_kv rows of D values, then their page_kv scales
  const std::int32_t* context_lens;  // [num_q]
  const std::int32_t* block_table;   // row of token i: i (varlen, run_starts) or i / next_n
  const std::int32_t* run_starts;    // [num_q] first token of the request of token i (varlen), or null
  int block_table_stride, next_n, page_kv;
  std::int64_t page_stride;          // bytes
  // Sparse / PagedSparse (from the sparse metadata): first KV row of each selected block,
  // [num_q, num_max_sparse_blocks]; paged: page * page_kv + row in the page. ks / ke (Sparse) or
  // context_lens (PagedSparse) give the tokens' KV lengths.
  const std::int32_t* sparse_starts;
  int num_max_sparse_blocks, sparse_block_kv;
};

}  // namespace deep_gemm::cdna4
