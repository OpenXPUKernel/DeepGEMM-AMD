#pragma once

#include <cstdlib>

namespace deep_gemm::cdna4 {

// Opt in to the launch shapes of one speculative decode batch. Other rows
// follow the regular planner, including ordinary requests on the same server.
inline bool decode_batch_rows(int rows, int positions) {
    const char* width = std::getenv("DG_BENCH_SPEC_WIDTH");
    if (positions == 5) {
        if (const char* verified = std::getenv("DEEPINFER_AMD_MAX_VERIFY_DRAFTS"))
            width = verified;
    }
    if (width && ((width[0] >= '1' && width[0] <= '5') || width[0] == '8') && width[1] == '\0') {
        if (positions == 4 || positions == 5)
            positions += width[0] - '4';
    }
    const char* value = std::getenv("DG_CDNA4_DECODE_BATCH");
    if (!value || !*value)
        return rows == 100 * positions;
    int batch = 0;
    for (int i = 0; value[i]; ++i) {
        if (i >= 3 || value[i] < '0' || value[i] > '9')
            return false;
        batch = batch * 10 + value[i] - '0';
    }
    return batch >= 1 && batch <= 256 && rows == batch * positions;
}

} // namespace deep_gemm::cdna4
