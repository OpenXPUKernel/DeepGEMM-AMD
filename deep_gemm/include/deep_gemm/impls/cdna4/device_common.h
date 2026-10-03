#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <cstring>
#include "gemm_args.h"
using deep_gemm::cdna4::GemmArgs;

typedef float v2f __attribute__((ext_vector_type(2)));
typedef int v2i __attribute__((ext_vector_type(2)));
typedef unsigned v2u __attribute__((ext_vector_type(2)));
typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));
typedef float v4f __attribute__((ext_vector_type(4)));
typedef unsigned v4u __attribute__((ext_vector_type(4)));
typedef float v16f __attribute__((ext_vector_type(16)));

enum MfmaFmt { kFmtFp8 = 0, kFmtBf8 = 1, kFmtFp6 = 2, kFmtBf6 = 3, kFmtFp4 = 4 };

__host__ __device__ inline float bf16_to_f32(std::uint16_t h) {
  std::uint32_t u = (std::uint32_t)h << 16; float f; __builtin_memcpy(&f, &u, 4); return f;
}
__host__ __device__ inline std::uint16_t f32_to_bf16(float f) {
  std::uint32_t u; __builtin_memcpy(&u, &f, 4);
  if ((u & 0x7fffffff) > 0x7f800000) return 0x7fc0;
  u += 0x7fff + ((u >> 16) & 1); return (std::uint16_t)(u >> 16);
}
__host__ __device__ inline float ue8m0_to_f32(std::uint8_t c) { return __builtin_ldexpf(1.f, (int)c - 127); }

template <class T> __device__ inline void store_out(T* p, float v);
template <> __device__ inline void store_out<float>(float* p, float v) { *p = v; }
template <> __device__ inline void store_out<std::uint16_t>(std::uint16_t* p, float v) { *p = f32_to_bf16(v); }
template <class T> __device__ inline float load_out(const T* p);
template <> __device__ inline float load_out<float>(const float* p) { return *p; }
template <> __device__ inline float load_out<std::uint16_t>(const std::uint16_t* p) { return bf16_to_f32(*p); }

__device__ __forceinline__ std::uint32_t sf_code(const std::int32_t* sf, int ld, int row, int k) {
  return ((std::uint32_t)sf[(std::size_t)(k / 128) * ld + row] >> (8 * ((k / 32) & 3))) & 0xff;
}
