// Host/device ABI shared by generated CDNA4 HIP kernels.
// Keep this header free of HIP intrinsics so host planners and device JIT code
// agree on layout without pulling in a runtime or compiler specific header.
#pragma once
#include <cstdint>
#include <cstddef>

namespace deep_gemm::cdna4 {
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
};
} // namespace deep_gemm::cdna4
