#pragma once
#include <deep_gemm/impls/cdna4/device_common.h>
#include <deep_gemm/impls/cdna4/gemm_args.h>
#include <deep_gemm/impls/cdna4/mega_moe_pipeline_memory.cuh>

// A staged 32 x 256 BF16 gate/up tile becomes 32 x 128 activations. Eight lanes per row,
// sixteen outputs per lane; adjacent lanes jointly reduce a 32-element quantization group.
// The BF16 stage preserves exactly the standalone GEMM -> SwiGLU rounding boundary.
template <bool BF16, int WM_ROWS = 128, bool COHERENT = false>
__device__ __forceinline__ void pipeline_swiglu_epilogue(const deep_gemm::cdna4::GemmArgs& p,
    const uint8_t* lds, int pitch, int m0, int mend, int n0, int round, int tid) {
  const int r = tid / 8, c = tid % 8;
  const int row = m0 + (r / 16) * WM_ROWS + 16 * round + r % 16;
  const uint16_t* src = reinterpret_cast<const uint16_t*>(lds + r * pitch) + 32 * c;
  float weight = row < mend ? 1.f : 0.f;
  if (row < mend && p.moe_records) {
    const auto* ptr = p.moe_records + int64_t(row) * p.moe_record_stride + p.moe_weight_word;
    if constexpr (COHERENT) weight = __int_as_float(__hip_atomic_load(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM));
    else weight = __int_as_float(*ptr);
  }
  float clamp, unused;
  unpack_bf16x2(pack_bf16x2(p.moe_clamp, 0.f), clamp, unused);
  float values[16], amax = 0.f;
#pragma unroll
  for (int block = 0; block < 2; ++block) {
    const v4u gate = *reinterpret_cast<const v4u*>(src + 16 * block);
    const v4u up = *reinterpret_cast<const v4u*>(src + 16 * block + 8);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float g[2], u[2];
      unpack_bf16x2(gate[j], g[0], g[1]);
      unpack_bf16x2(up[j], u[0], u[1]);
#pragma unroll
      for (int z = 0; z < 2; ++z) {
        const float x = fminf(g[z], clamp), y = fminf(fmaxf(u[z], -clamp), clamp);
        const float denominator = 1.f + (p.moe_fast_math ? __expf(-x) : expf(-x));
        const float silu = p.moe_fast_math ? x * __builtin_amdgcn_rcpf(denominator) : x / denominator;
        const float v = silu * y * weight;
        values[block * 8 + j * 2 + z] = v;
        amax = fmaxf(amax, fabsf(v));
      }
    }
  }
  const int64_t offset = int64_t(row) * (p.N / 2) + n0 / 2 + c * 16;
  if constexpr (BF16) {
    v4u packed[2];
#pragma unroll
    for (int j = 0; j < 8; ++j)
      packed[j / 4][j % 4] = pack_bf16x2(values[j * 2], values[j * 2 + 1]);
    if (row < mend) {
      auto* out = reinterpret_cast<v4u*>(p.moe_acts + offset * 2);
      if constexpr (COHERENT) { nv_pipeline_detail::coherent_store16(out, packed[0]); nv_pipeline_detail::coherent_store16(out+1, packed[1]); }
      else { out[0] = packed[0]; out[1] = packed[1]; }
    }
  } else {
    amax = fmaxf(amax, __shfl_xor(amax, 1));
    const uint32_t rounded = (__float_as_uint(amax) + 0x7fffffu - (0x60u << 16)) >> 23;
    const uint32_t exponent = max(rounded, 113u) - 8u;
    const float inverse = __uint_as_float((254u - exponent) << 23);
    v4u packed;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int lo = __builtin_amdgcn_cvt_pk_fp8_f32(values[4*j] * inverse, values[4*j+1] * inverse, 0, false);
      packed[j] = __builtin_amdgcn_cvt_pk_fp8_f32(values[4*j+2] * inverse, values[4*j+3] * inverse, lo, true);
    }
    if (row < mend) {
      if constexpr (COHERENT) nv_pipeline_detail::coherent_store16(p.moe_acts + offset, packed);
      else *reinterpret_cast<v4u*>(p.moe_acts + offset) = packed;
    }
    const int base = (tid % 64) & ~7;
    const uint32_t word = uint32_t(__shfl(int(exponent), base)) |
                          (uint32_t(__shfl(int(exponent), base + 2)) << 8) |
                          (uint32_t(__shfl(int(exponent), base + 4)) << 16) |
                          (uint32_t(__shfl(int(exponent), base + 6)) << 24);
    if (row < mend && c == 0) {
      auto* ptr = p.moe_sf + int64_t(n0 / 256) * p.moe_sf_pitch + row;
      if constexpr (COHERENT) __hip_atomic_store(ptr, int32_t(word), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      else *ptr = int32_t(word);
    }
  }
}
