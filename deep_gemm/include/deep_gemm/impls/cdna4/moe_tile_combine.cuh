#pragma once
#include <deep_gemm/impls/cdna4/device_common.h>
#include <deep_gemm/impls/cdna4/tile_scheduler.cuh>
#include <deep_gemm/impls/cdna4/moe_copy_staged.cuh>

template <class Cfg>
__device__ __forceinline__ uint64_t* moe_pipeline_trace(const GemmArgs& p) {
  const int tiles = ((p.M + Cfg::BM - 1) / Cfg::BM) * ((p.N + Cfg::BN - 1) / Cfg::BN);
  return reinterpret_cast<uint64_t*>(p.moe_tile_ready + ((tiles + 1) & ~1)) + int64_t(blockIdx.x) * 8;
}

// Combine full rows once all N tiles of an M block have completed. This keeps
// peer writes contiguous across the hidden dimension instead of sending tiny
// column fragments. Completion is an atomic release sequence of the producers.
template <class Cfg>
__device__ __forceinline__ void moe_tile_combine(const GemmArgs& p, int worker) {
  using Scheduler = typename Cfg::Scheduler;
  GemmArgs row_plan = p;
  row_plan.N = Cfg::BN;
  const int blocks = p.moe_peer_order ? p.num_groups * p.moe_ranks : Scheduler::num_tiles(row_plan);
  const int columns = (p.N + Cfg::BN - 1) / Cfg::BN;
  const int lane = threadIdx.x % 64, wave = __builtin_amdgcn_readfirstlane(threadIdx.x / 64);
  uint64_t entered = 0, waiting = 0, copying = 0, scheduling = 0, phase = 0, count = 0, max_copy = 0;
  if constexpr (Cfg::PIPELINE_PROFILE) entered = wall_clock64();
  for (int ticket = worker; ticket < blocks; ticket += p.moe_comm_blocks) {
    if constexpr (Cfg::PIPELINE_PROFILE) phase = wall_clock64();
    int row_begin, row_end, flag_index, expected;
    if (p.moe_peer_order) {
      const int expert = ticket / p.moe_ranks, source = (ticket % p.moe_ranks + p.moe_rank) % p.moe_ranks;
      const int expert_begin = Scheduler::psum_start(p, expert), expert_end = p.grouped_layout[expert];
      int before = 0;
      for (int rank = 0; rank < source; ++rank)
        before += p.moe_source_counts[rank * p.num_groups * p.moe_ranks + p.moe_rank * p.num_groups + expert];
      const int count = p.moe_source_counts[source * p.num_groups * p.moe_ranks + p.moe_rank * p.num_groups + expert];
      row_begin = expert_begin + before; row_end = row_begin + count;
      flag_index = expert;
      expected = ((expert_end - expert_begin + Cfg::BM - 1) / Cfg::BM) * columns;
    } else {
      const auto tile = Scheduler::tile(row_plan, ticket);
      row_begin = tile.m0; row_end = min(tile.m_end, tile.m0 + Cfg::BM);
      flag_index = tile.m0 / Cfg::BM; expected = columns;
    }
    if (row_begin == row_end) continue;
    if constexpr (Cfg::PIPELINE_PROFILE) {
      const uint64_t now = wall_clock64(); scheduling += now - phase; phase = now;
    }
    const auto* flag = p.moe_tile_ready + flag_index;
    if (threadIdx.x == 0) {
      const uint64_t start = wall_clock64();
      while (__hip_atomic_load(flag, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < expected) {
        __builtin_amdgcn_s_sleep(4);
        if (wall_clock64() - start > 6000000000ull) __builtin_trap();
      }
      if constexpr (Cfg::COPY_CACHE == 0)
        (void)__hip_atomic_load(flag, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
      else {
        // Device-specific acquire for SC1 payload reads: the producer release
        // writes back dirty data before publishing the count. Wait for the
        // coherent flag load, then keep subsequent coherent payload loads after
        // it. Those loads bypass non-coherent caches; buffer_inv is unnecessary
        // for them and would evict immutable weights on the compute team.
        // This must not be used with ordinary cached mutable-payload loads.
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      }
    }
    __syncthreads();
    if constexpr (Cfg::PIPELINE_PROFILE) {
      const uint64_t now = wall_clock64(); waiting += now - phase; phase = now;
    }
    if (p.N == 7168 && p.ldd == p.N) {
      moe_copy_staged_7168<Cfg::COPY_CACHE>(p, row_begin, row_end);
    } else for (int row = row_begin + wave; row < row_end; row += 4) {
      const auto source_resource = make_rsrc(reinterpret_cast<const uint16_t*>(p.D) +
          int64_t(row) * p.ldd, uint32_t(p.N * 2));
      const uint64_t output = uint64_t(p.moe_output_rows[row]);
      for (int first = 0; first < p.N * 2; first += 8192) {
        v4u values[8]{};
#pragma unroll
        for (int i = 0; i < 8; ++i)
          if (first + i * 1024 < p.N * 2)
            values[i] = __builtin_amdgcn_raw_buffer_load_b128(source_resource,
                first + i * 1024 + lane * 16,
                0, Cfg::COPY_CACHE); // 16=SC1; 0 uses caches after completion acquire
#pragma unroll
        for (int i = 0; i < 8; ++i)
          if (first + i * 1024 < p.N * 2) {
            const uint64_t address = output + first + i * 1024 + lane * 16;
            asm volatile("global_store_dwordx4 %0, %1, off\n\ts_nop 0" :: "v"(address), "v"(values[i]) : "memory");
          }
      }
    }
    __syncthreads();
    if constexpr (Cfg::PIPELINE_PROFILE) {
      const uint64_t duration = wall_clock64() - phase;
      copying += duration; max_copy = max(max_copy, duration); ++count;
    }
  }
  __threadfence_system();
  if constexpr (Cfg::PIPELINE_PROFILE) {
    const uint64_t end = wall_clock64();
    if (threadIdx.x == 0) {
      auto* r = moe_pipeline_trace<Cfg>(p);
      r[0] = 1; r[1] = count; r[2] = waiting; r[3] = copying;
      r[4] = entered; r[5] = end; r[6] = max_copy; r[7] = scheduling;
    }
  }
}
