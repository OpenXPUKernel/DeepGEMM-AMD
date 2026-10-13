// Device types, conversions and memory primitives shared by the CDNA4 kernels.
#pragma once
#include "gemm_args.h"
#include <cstdint>
#include <hip/hip_runtime.h>
#include <type_traits>
using deep_gemm::cdna4::GemmArgs;

typedef int v2i __attribute__((ext_vector_type(2)));
typedef unsigned v2u __attribute__((ext_vector_type(2)));
typedef float v2f __attribute__((ext_vector_type(2)));
typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));
typedef float v4f __attribute__((ext_vector_type(4)));
typedef unsigned v4u __attribute__((ext_vector_type(4)));

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
  return __builtin_amdgcn_make_buffer_rsrc(const_cast<void *>(base), (short)0, (int)bytes,
                                           0x00020000);
}

// An explicit restricted LDS pointer prevents the compiler from inserting
// global-load waits before a read from the other pipeline stage.
__device__ __forceinline__ v2i ds_read_tr8(const __attribute__((address_space(3)))
                                           v2i *__restrict__ p) {
  return __builtin_amdgcn_ds_read_tr8_b64_v2i32(
      const_cast<__attribute__((address_space(3))) v2i *>(p));
}

__device__ __forceinline__ v2i ds_read_tr16(const __attribute__((address_space(3)))
                                            v2i *__restrict__ p) {
  typedef short v4s __attribute__((ext_vector_type(4)));
  return __builtin_bit_cast(v2i, __builtin_amdgcn_ds_read_tr16_b64_v4i16(
                                     (__attribute__((address_space(3))) v4s *)const_cast<
                                         __attribute__((address_space(3))) v2i *>(p)));
}

// The second half of a 16-byte buffer store is read one cycle after issue.
__device__ __forceinline__ void buffer_store16(v4u v, __amdgpu_buffer_rsrc_t r, uint32_t voff,
                                               int soff) {
  __builtin_amdgcn_raw_buffer_store_b128(v, r, voff, soff, 0);
  __builtin_amdgcn_sched_barrier(0);
  asm volatile("s_nop 0");
  __builtin_amdgcn_sched_barrier(0);
}

#endif

// f(integral_constant<int, I>) for I = B .. E - 1: a compile-time unrolled loop. The MFMA slots
// of a K step are too many for `#pragma unroll` to be reliable, and a rolled loop would put the
// register arrays in scratch.
template <int B, int E, class F> __device__ __forceinline__ void static_for(F &&f) {
  if constexpr (B < E) {
    f(std::integral_constant<int, B>{});
    static_for<B + 1, E>(f);
  }
}

// D += A B^T on 16x16 tiles with K = 32 BF16 values per lane row (8 per lane)
__device__ __forceinline__ v4f mfma_bf16_16x16x32(v4i a, v4i b, v4f c) {
  typedef __bf16 v8bf __attribute__((ext_vector_type(8)));
  return __builtin_amdgcn_mfma_f32_16x16x32_bf16(__builtin_bit_cast(v8bf, a),
                                                  __builtin_bit_cast(v8bf, b), c, 0, 0, 0);
}

// Join the two halves of a scaled MFMA operand.
__device__ __forceinline__ v8i cat8(const v4i &lo, const v4i &hi) {
  return __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7);
}
