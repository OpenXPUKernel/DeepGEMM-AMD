#pragma once
namespace deep_gemm::cdna4::heuristics {
struct MegaMoeConfig {
    static constexpr int block_m = 128, block_n = 256, column_groups = 6;
    static constexpr int max_blocks = 256, communication_blocks = 16;
    static constexpr unsigned shared_memory_bytes = 3 * (128 * 128 + 512 * 64 + 256 * 4 + 512 * 4);
};
} // namespace deep_gemm::cdna4::heuristics
