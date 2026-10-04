// Vectorized output stores with optional accumulation and edge handling.
#pragma once
#include <type_traits>
#include "device_common.h"

// Out[row][col .. col + 7] = v (+ C[row][col .. col + 7]) for the columns < N; v is already scaled
// by alpha. Out and C share the row stride ldo (elements).
template <class OutT>
__device__ __forceinline__ void store8(void *out, const void *C, int ldo, int M, int N, int row, int col,
                                       v4f lo, v4f hi, bool vec_ok) {
  if (row >= M || col >= N)
    return;
  const size_t off = (size_t)row * ldo + col;
  if (vec_ok && col + 8 <= N) {
    if constexpr (std::is_same<OutT, uint16_t>::value) {
      if (C) {
        const v4i c = *(const v4i *)((const uint16_t *)C + off);
        float c0, c1;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          unpack_bf16x2(c[i], c0, c1);
          lo[2 * i] += c0;
          lo[2 * i + 1] += c1;
          unpack_bf16x2(c[2 + i], c0, c1);
          hi[2 * i] += c0;
          hi[2 * i + 1] += c1;
        }
      }
      const v4i o = {(int)pack_bf16x2(lo[0], lo[1]), (int)pack_bf16x2(lo[2], lo[3]),
                     (int)pack_bf16x2(hi[0], hi[1]), (int)pack_bf16x2(hi[2], hi[3])};
      *(v4i *)((uint16_t *)out + off) = o;
    } else {
      if (C) {
        lo += *(const v4f *)((const float *)C + off);
        hi += *(const v4f *)((const float *)C + off + 4);
      }
      *(v4f *)((float *)out + off) = lo;
      *(v4f *)((float *)out + off + 4) = hi;
    }
  } else {
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      if (col + e >= N)
        break;
      float v = e < 4 ? lo[e] : hi[e - 4];
      if (C)
        v += load_out((const OutT *)C + off + e);
      store_out((OutT *)out + off + e, v);
    }
  }
}
