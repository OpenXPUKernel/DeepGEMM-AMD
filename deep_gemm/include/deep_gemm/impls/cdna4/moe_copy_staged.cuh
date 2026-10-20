#pragma once
#include <deep_gemm/impls/cdna4/device_common.h>

// Double-buffer a contiguous local slab. Prefetch the following batch before
// issuing the current peer writes, so those writes need not drain before every
// following load. A slab may span several output rows; LDS holds their targets.
template <int CACHE>
__device__ __forceinline__ void moe_copy_staged_7168(const GemmArgs& p, int begin, int end) {
  extern __shared__ __attribute__((aligned(1024))) uint8_t smem[];
  auto* destinations = reinterpret_cast<uint64_t*>(smem);
  constexpr int row_bytes = 7168 * 2, batch_bytes = 256 * 16 * 8;
  const int tid = threadIdx.x, lane = tid % 64;
  const int wave = __builtin_amdgcn_readfirstlane(tid / 64);
  for (int first_row = begin; first_row < end; first_row += 256) {
    const int rows = min(256, end - first_row), bytes = rows * row_bytes;
    if (tid < rows) destinations[tid] = uint64_t(p.moe_output_rows[first_row + tid]);
    asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)\n\ts_barrier" ::: "memory");
    const auto source = make_rsrc(reinterpret_cast<const uint16_t*>(p.D) + int64_t(first_row) * p.ldd, bytes);
    const int batches = (bytes + batch_bytes - 1) / batch_bytes;
    v4u values[2][8];
    auto load = [&](auto S, int batch) {
      constexpr int slot = decltype(S)::value;
      static_for<0, 8>([&](auto I) {
        constexpr int i = decltype(I)::value;
        const uint32_t offset = batch * batch_bytes + i * 4096 + tid * 16;
        auto& value = values[slot][i];
        const auto resource = source;
        if constexpr (CACHE == 16)
          asm volatile("buffer_load_dwordx4 %0, %1, %2, 0 offen sc1" : "=&v"(value) : "v"(offset), "s"(resource) : "memory");
        else
          asm volatile("buffer_load_dwordx4 %0, %1, %2, 0 offen" : "=&v"(value) : "v"(offset), "s"(resource) : "memory");
      });
      __builtin_amdgcn_sched_barrier(0);
    };
    auto wait = [&](auto S, auto C) {
      constexpr int slot = decltype(S)::value, count = decltype(C)::value;
      auto& v = values[slot];
      if constexpr (count == 8)
        asm volatile("s_waitcnt vmcnt(8)" : "+v"(v[0]), "+v"(v[1]), "+v"(v[2]), "+v"(v[3]),
            "+v"(v[4]), "+v"(v[5]), "+v"(v[6]), "+v"(v[7]) :: "memory");
      else
        asm volatile("s_waitcnt vmcnt(16)" : "+v"(v[0]), "+v"(v[1]), "+v"(v[2]), "+v"(v[3]),
            "+v"(v[4]), "+v"(v[5]), "+v"(v[6]), "+v"(v[7]) :: "memory");
    };
    auto store = [&](auto S, int batch) {
      constexpr int slot = decltype(S)::value;
      static_for<0, 8>([&](auto I) {
        constexpr int i = decltype(I)::value;
        const int offset = batch * batch_bytes + i * 4096 + wave * 1024;
        if (offset < bytes) {
          const int row = __builtin_amdgcn_readfirstlane(offset / row_bytes);
          const uint64_t address = destinations[row] + (offset - row * row_bytes) + lane * 16;
          const v4u value = values[slot][i];
          asm volatile("global_store_dwordx4 %0, %1, off\n\ts_nop 0" :: "v"(address), "v"(value) : "memory");
        }
      });
      __builtin_amdgcn_sched_barrier(0);
    };
    using S0 = std::integral_constant<int, 0>;
    using S1 = std::integral_constant<int, 1>;
    load(S0{}, 0); load(S1{}, 1);
    for (int batch = 0; batch < batches; batch += 2) {
      if (batch == 0) wait(S0{}, std::integral_constant<int, 8>{});
      else wait(S0{}, std::integral_constant<int, 16>{});
      store(S0{}, batch); load(S0{}, batch + 2);
      if (batch + 1 < batches) {
        wait(S1{}, std::integral_constant<int, 16>{});
        store(S1{}, batch + 1); load(S1{}, batch + 3);
      }
    }
    asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)\n\ts_barrier" ::: "memory");
    // Keep the final, out-of-bounds lookahead loads observable to the compiler.
    // Removing one of them on the last iteration would invalidate the fixed
    // vmcnt budget used before consuming the other stage.
    asm volatile("" :: "v"(values[0][0]), "v"(values[0][1]), "v"(values[0][2]), "v"(values[0][3]),
        "v"(values[0][4]), "v"(values[0][5]), "v"(values[0][6]), "v"(values[0][7]),
        "v"(values[1][0]), "v"(values[1][1]), "v"(values[1][2]), "v"(values[1][3]),
        "v"(values[1][4]), "v"(values[1][5]), "v"(values[1][6]), "v"(values[1][7]) : "memory");
  }
}
