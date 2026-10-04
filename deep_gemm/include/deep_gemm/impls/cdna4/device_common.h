// Device types, conversions and memory primitives shared by the CDNA4 kernels.
#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstddef>
#include "gemm_args.h"
using deep_gemm::cdna4::GemmArgs;

typedef float v2f __attribute__((ext_vector_type(2)));
typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));
typedef float v4f __attribute__((ext_vector_type(4)));
typedef unsigned v4u __attribute__((ext_vector_type(4)));
typedef float v16f __attribute__((ext_vector_type(16)));

// OCP E4M3 operand format for scaled MFMA.
constexpr int kFmtFp8 = 0;

__host__ __device__ inline float bf16_to_f32(std::uint16_t h) {
  std::uint32_t u = (std::uint32_t)h << 16;
  float f;
  __builtin_memcpy(&f, &u, 4);
  return f;
}
__host__ __device__ inline std::uint16_t f32_to_bf16(float f) {
  std::uint32_t u;
  __builtin_memcpy(&u, &f, 4);
  if ((u & 0x7fffffff) > 0x7f800000)
    return 0x7fc0;
  u += 0x7fff + ((u >> 16) & 1);
  return (std::uint16_t)(u >> 16);
}

template <class T> __device__ inline void store_out(T *p, float v);
template <> __device__ inline void store_out<float>(float *p, float v) { *p = v; }
template <> __device__ inline void store_out<std::uint16_t>(std::uint16_t *p, float v) {
  *p = f32_to_bf16(v);
}
template <class T> __device__ inline float load_out(const T *p);
template <> __device__ inline float load_out<float>(const float *p) { return *p; }
template <> __device__ inline float load_out<std::uint16_t>(const std::uint16_t *p) {
  return bf16_to_f32(*p);
}

// s_waitcnt immediate (CDNA4 encoding) that waits for vmcnt <= n and nothing else
constexpr int waitcnt_vm(int n) { return (n & 15) | ((n >> 4) << 14) | (7 << 4) | (15 << 8); }

typedef __attribute__((address_space(3))) void *lds_ptr_t;

typedef __bf16 v2bf __attribute__((ext_vector_type(2)));

__device__ __forceinline__ uint32_t pack_bf16x2(float a, float b) { // v_cvt_pk_bf16_f32, RNE
  return __builtin_bit_cast(uint32_t, __builtin_convertvector((v2f){a, b}, v2bf));
}
__device__ __forceinline__ void unpack_bf16x2(uint32_t x, float &a, float &b) {
  a = __uint_as_float(x << 16);
  b = __uint_as_float(x & 0xffff0000u);
}

#if defined(__HIP_DEVICE_COMPILE__)
__device__ __forceinline__ __amdgpu_buffer_rsrc_t make_rsrc(const void *base, uint32_t bytes) {
  return __builtin_amdgcn_make_buffer_rsrc(const_cast<void *>(base), (short)0, (int)bytes, 0x00020000);
}
#endif

// Join the two halves of a scaled MFMA operand.
__device__ __forceinline__ v8i cat8(const v4i &lo, const v4i &hi) {
  return __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7);
}
