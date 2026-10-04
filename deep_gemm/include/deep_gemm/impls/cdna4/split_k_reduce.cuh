// Deterministic split-K reduction; generated only for the split-K path.
#pragma once
#include <type_traits>
#include "device_common.h"

// D = alpha * sum_s ws[s] (+ C), partials added in split order so the result does not depend on
// which workgroup finished first. One thread per 4 columns; blockIdx.y walks the rows.
template <class OutT> __device__ __forceinline__ void splitk_reduce_impl(GemmArgs p) {
  const int col = (blockIdx.x * 256 + threadIdx.x) * 4;
  if (col >= p.N)
    return;
  constexpr int VB = 4 * sizeof(OutT); // bytes of 4 outputs
  const bool vec_ok = (p.ldd * (int)sizeof(OutT)) % VB == 0 && (uintptr_t)p.D % VB == 0 &&
                      (!p.C || (uintptr_t)p.C % VB == 0) && col + 4 <= p.N;
  const size_t split_stride = (size_t)p.M * p.ldw;
  for (int row = blockIdx.y; row < p.M; row += gridDim.y) {
    const float *w = p.ws + (size_t)row * p.ldw + col;
    v4f s = *(const v4f *)w;
    for (int k = 1; k < p.splits; ++k)
      s += *(const v4f *)(w + k * split_stride);
    s *= p.alpha;
    const size_t off = (size_t)row * p.ldd + col;
    if (vec_ok) {
      if constexpr (std::is_same<OutT, uint16_t>::value) {
        if (p.C) {
          const uint2 c = *(const uint2 *)((const uint16_t *)p.C + off);
          float c0, c1, c2, c3;
          unpack_bf16x2(c.x, c0, c1);
          unpack_bf16x2(c.y, c2, c3);
          s += v4f{c0, c1, c2, c3};
        }
        *(uint2 *)((uint16_t *)p.D + off) = uint2{pack_bf16x2(s[0], s[1]), pack_bf16x2(s[2], s[3])};
      } else {
        if (p.C)
          s += *(const v4f *)((const float *)p.C + off);
        *(v4f *)((float *)p.D + off) = s;
      }
    } else {
      for (int e = 0; e < 4 && col + e < p.N; ++e) {
        float v = s[e];
        if (p.C)
          v += load_out((const OutT *)p.C + off + e);
        store_out((OutT *)p.D + off + e, v);
      }
    }
  }
}
