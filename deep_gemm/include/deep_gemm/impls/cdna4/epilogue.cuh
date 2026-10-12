// Vectorized output stores with optional accumulation and edge handling.
#pragma once
#include "device_common.h"
#include <type_traits>

// Out[row][col .. col + 7] = v (+ C[row][col .. col + 7]) for the columns < N
template <class OutT>
__device__ __forceinline__ void store8(void *out, const void *C, int ldo, int M, int N, int row,
                                       int col, v4f lo, v4f hi, bool vec_ok) {
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

// C[row][col .. col + 7] as FP32 (0 past M and N). Loading C ahead of a group of store8 calls keeps
// the loads in flight together: C may alias the output, so the compiler cannot move a load past
// an earlier store, and each accumulation would wait for its own load.
template <class OutT>
__device__ __forceinline__ void load_c8(const void *C, int ldo, int M, int N, int row, int col,
                                        bool vec_ok, v4f &lo, v4f &hi) {
  lo = hi = v4f{};
  if (row >= M || col >= N)
    return;
  const size_t off = (size_t)row * ldo + col;
  if (vec_ok && col + 8 <= N) {
    if constexpr (std::is_same<OutT, uint16_t>::value) {
      const v4i c = *(const v4i *)((const uint16_t *)C + off);
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        float c0, c1, c2, c3;
        unpack_bf16x2(c[i], c0, c1);
        unpack_bf16x2(c[2 + i], c2, c3);
        lo[2 * i] = c0;
        lo[2 * i + 1] = c1;
        hi[2 * i] = c2;
        hi[2 * i + 1] = c3;
      }
    } else {
      lo = *(const v4f *)((const float *)C + off);
      hi = *(const v4f *)((const float *)C + off + 4);
    }
  } else {
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      if (col + e >= N)
        break;
      const float v = load_out((const OutT *)C + off + e);
      if (e < 4)
        lo[e] = v;
      else
        hi[e - 4] = v;
    }
  }
}

// store8 for 4 consecutive outputs of a row (persistent kernel with an MN-major B)
template <class OutT>
__device__ __forceinline__ void store4(void *out, const void *C, int ldo, int M, int N, int row,
                                       int col, v4f v, bool vec_ok) {
  if (row >= M || col >= N)
    return;
  const size_t off = (size_t)row * ldo + col;
  if (vec_ok && col + 4 <= N) {
    if constexpr (std::is_same<OutT, uint16_t>::value) {
      if (C) {
        const v2i c = *(const v2i *)((const uint16_t *)C + off);
        float c0, c1;
        unpack_bf16x2(c[0], c0, c1);
        v[0] += c0;
        v[1] += c1;
        unpack_bf16x2(c[1], c0, c1);
        v[2] += c0;
        v[3] += c1;
      }
      *(v2u *)((uint16_t *)out + off) = v2u{pack_bf16x2(v[0], v[1]), pack_bf16x2(v[2], v[3])};
    } else {
      if (C)
        v += *(const v4f *)((const float *)C + off);
      *(v4f *)((float *)out + off) = v;
    }
  } else {
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      if (col + e >= N)
        break;
      float x = v[e];
      if (C)
        x += load_out((const OutT *)C + off + e);
      store_out((OutT *)out + off + e, x);
    }
  }
}
