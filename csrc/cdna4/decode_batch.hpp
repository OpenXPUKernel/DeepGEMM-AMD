#pragma once

#include <cstdlib>

namespace deep_gemm::cdna4 {

// Opt in to the launch shapes of one speculative decode batch. Other rows
// follow the regular planner, including ordinary requests on the same server.
inline bool decode_batch_rows(int rows, int positions) {
    const char* value = std::getenv("DG_CDNA4_DECODE_BATCH");
    if (!value || !*value)
        return rows == 100 * positions;
    char* end = nullptr;
    const long batch = std::strtol(value, &end, 10);
    return *end == '\0' && batch > 0 && batch <= 256 && rows == batch * positions;
}

} // namespace deep_gemm::cdna4
