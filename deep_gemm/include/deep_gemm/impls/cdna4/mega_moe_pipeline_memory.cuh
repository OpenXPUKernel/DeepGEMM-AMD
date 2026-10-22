#pragma once
#include <deep_gemm/impls/cdna4/device_common.h>
namespace nv_pipeline_detail {
__device__ __forceinline__ v4u coherent_load16(const void* ptr) {
  v4u value;
  const uint64_t address = reinterpret_cast<uint64_t>(ptr);
  // Keep the load and its wait indivisible: register allocation may otherwise
  // move a pending output to another VGPR before a later tied-operand wait.
  asm volatile("global_load_dwordx4 %0, %1, off sc1\n\ts_waitcnt vmcnt(0)"
               : "=&v"(value) : "v"(address) : "memory");
  return value;
}
__device__ __forceinline__ void coherent_store16(void* ptr, v4u value) {
  const uint64_t address = reinterpret_cast<uint64_t>(ptr);
  asm volatile("global_store_dwordx4 %0, %1, off sc1\n\ts_nop 0" :: "v"(address), "v"(value) : "memory");
}
}
