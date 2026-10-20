// CDNA4 Mega MoE: expert-parallel MoE across the GPUs of a node on uni_symm symmetric buffers.
//
// A call runs, in stream order on every rank:
//   route    every (token, slot) takes a position among its expert's rows from this rank;
//   barrier  (one workgroup) publishes this rank's counts to every rank, and from all counts
//            lays out the rows: each expert's rows on its rank (experts aligned to the GEMM
//            tiles, sources in rank order), so every source knows its rows there;
//   push     one wave per token writes its row and a record (scales, top-k weight, source) into
//            the rows of its slots on the experts' ranks: remote writes, which the fabric
//            carries at twice the rate of remote reads;
//   barrier  every rank's rows arrived;
//   finish   the records' scales become the GEMM's MN-major layout (and the local tokens' for
//            the shared expert);
//   (L1 grouped GEMM over the PSUM layout of the experts' end rows, and the shared expert GEMM)
//   swiglu   clamp, silu(gate) * up * top-k weight (gate / up interleaved by 8 columns), cast to
//            FP8 with UE8M0 scales per 32 values (or BF16);
//   (L2 grouped GEMM, and the shared expert GEMM)
//   combine  each row's result is written into slot `slot` of its token on the source rank;
//   barrier  every rank's results arrived;
//   reduce   y = the sum of the token's slots (top-k order) + the shared expert, in FP32.
// Readers of peer-written memory acquire at system scope first: each XCD's L2 is invalidated by
// its own acquire.
#pragma once
#include <deep_gemm/impls/cdna4/device_common.h>
#include <deep_gemm/impls/cdna4/mega_moe_args.h>

using deep_gemm::cdna4::MegaMoeArgs;
using deep_gemm::cdna4::MegaMoeLayout;
using uni_symm::device::Scope;

namespace mega_moe {

// Same protocol and memory ordering as uni_symm::barrier_workgroup. Capture
// only the scalar diagnostic values: capturing the whole SymmView by reference
// makes HIPRTC materialize the large kernel-argument object in private memory.
template<int Ranks>
__device__ __forceinline__ void compact_barrier_workgroup(
    const uni_symm::SymmView<Ranks>& view,uint32_t* counter,int* signals,int tid) {
  using namespace uni_symm::device;
  const uint32_t status=load_relaxed<Scope::device>(counter)&3u;
  const int phase=int(status&1u),sign=int(status>>1);
  int* signal=signals+phase;
  fence_release<Scope::system>();workgroup_sync();
  if(tid<view.world_size)atomic_add_release<Scope::system>(view.map(signal,tid),sign?-1:1);
  if(tid==0) {
    store_relaxed<Scope::device>(counter,load_relaxed<Scope::device>(counter)+1u);
    const int target=sign?0:view.world_size,rank=view.rank;
    wait_until([signal,target]{return load_acquire<Scope::system>(signal)==target;},
      [signal,target,rank]{printf("uni_symm barrier timeout: rank %d, signal %d, target %d\n",rank,load_relaxed<Scope::system>(signal),target);});
  }
  workgroup_sync();fence_acquire<Scope::system>();
}

__device__ __forceinline__ uint8_t *base(const MegaMoeArgs &p) { return reinterpret_cast<uint8_t *>(p.view.base); }

template <class T> __device__ __forceinline__ T *region(const MegaMoeArgs &p, int64_t offset) {
  return reinterpret_cast<T *>(base(p) + offset);
}

__device__ __forceinline__ int global_wave() { return blockIdx.x * (blockDim.x / 64) + threadIdx.x / 64; }
__device__ __forceinline__ int num_waves() { return gridDim.x * (blockDim.x / 64); }

// copy `bytes` (a multiple of 16) from `src` to `dst` with a wave, eight 16-byte loads in flight per lane
__device__ __forceinline__ void wave_copy(uint8_t *dst, const uint8_t *src, int bytes, int lane) {
  for (int offset = lane * 16; offset < bytes; offset += 64 * 16 * 8) {
    v4i v[8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
      v[i] = offset + i * 64 * 16 < bytes ? *reinterpret_cast<const v4i *>(src + offset + i * 64 * 16) : v4i{};
#pragma unroll
    for (int i = 0; i < 8; ++i)
      if (offset + i * 64 * 16 < bytes)
        *reinterpret_cast<v4i *>(dst + offset + i * 64 * 16) = v[i];
  }
}

// the start row of local expert `e` from the end rows (PSUM layout)
__device__ __forceinline__ int expert_start(const int *ends, int e, int alignment) {
  return e == 0 ? 0 : (ends[e - 1] + alignment - 1) / alignment * alignment;
}

// the local expert of routed row `row`: the last expert starting at or before it
__device__ __forceinline__ int expert_of_row(const int *starts, int num_experts, int row) {
  int lo = 0, hi = num_experts - 1;
  while (lo < hi) {
    const int mid = (lo + hi + 1) / 2;
    if (starts[mid] <= row)
      lo = mid;
    else
      hi = mid - 1;
  }
  return lo;
}

}  // namespace mega_moe

namespace mega_moe {
// The receive table of this rank's experts in LDS, from the published counts: expert e's rows from
// source s (in the source's order) start at starts[e] + offsets[s][e]. Work index k visits source
// k % ranks first, so concurrent waves use every peer's link.
struct RowTable {
  int *counts;      // [ranks][experts]
  int *offsets;     // [ranks][experts]: rows of the earlier sources within the expert
  int *src_prefix;  // [ranks][experts + 1]: rows of the source in the earlier experts
  int *starts;      // [experts + 1]: aligned start rows, then the routed rows' end
  int *totals;      // [experts]
  int *max_from_source;

  // the routed row of work index k, with its source and expert, or -1
  __device__ int row_of(int k, int ranks, int experts, int &source, int &expert) const {
    const int s = k % ranks, j = k / ranks;
    const int *prefix = src_prefix + s * (experts + 1);
    if (j >= prefix[experts])
      return -1;
    int lo = 0, hi = experts - 1;
    while (lo < hi) {
      const int mid = (lo + hi + 1) / 2;
      if (prefix[mid] <= j)
        lo = mid;
      else
        hi = mid - 1;
    }
    source = s, expert = lo;
    return starts[lo] + offsets[s * experts + lo] + (j - prefix[lo]);
  }
};

__device__ __forceinline__ RowTable build_row_table(const MegaMoeArgs &p, int *smem) {
  const MegaMoeLayout &l = p.layout;
  const int ranks = l.num_ranks, experts = l.num_experts_per_rank;
  RowTable t;
  t.counts = smem;
  t.offsets = t.counts + ranks * experts;
  t.src_prefix = t.offsets + ranks * experts;
  t.starts = t.src_prefix + ranks * (experts + 1);
  t.totals = t.starts + experts + 1;
  t.max_from_source = t.totals + experts;
  const int *counts = region<const int>(p, l.counts) + p.view.rank * experts;
  for (int i = threadIdx.x; i < ranks * experts; i += blockDim.x)
    t.counts[i] = uni_symm::device::load_relaxed<Scope::system>(counts + (i / experts) * l.num_experts + i % experts);
  __syncthreads();
  for (int e = threadIdx.x; e < experts; e += blockDim.x) {
    int total = 0;
    for (int s = 0; s < ranks; ++s)
      t.offsets[s * experts + e] = total, total += t.counts[s * experts + e];
    t.totals[e] = total;
  }
  for (int s = threadIdx.x; s < ranks; s += blockDim.x) {
    int prefix = 0;
    for (int e = 0; e < experts; ++e)
      t.src_prefix[s * (experts + 1) + e] = prefix, prefix += t.counts[s * experts + e];
    t.src_prefix[s * (experts + 1) + experts] = prefix;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    int start = 0, most = 0;
    for (int e = 0; e < experts; ++e)
      t.starts[e] = start, start += (t.totals[e] + l.alignment - 1) / l.alignment * l.alignment;
    t.starts[experts] = start;
    for (int s = 0; s < ranks; ++s)
      most = max(most, t.src_prefix[s * (experts + 1) + experts]);
    *t.max_from_source = most;
  }
  __syncthreads();
  return t;
}
}  // namespace mega_moe

// Every (token, slot) of this rank takes the next position among its expert's rows from this
// rank. A row's position does not change its result.
__device__ __forceinline__ void mega_moe_route_impl(const MegaMoeArgs &p) {
  using namespace mega_moe;
  const MegaMoeLayout &l = p.layout;
  const int entries = p.num_tokens * l.num_topk;
  const int64_t *topk_idx = region<const int64_t>(p, l.topk_idx);
  int *send = region<int>(p, l.send_counts), *positions = region<int>(p, l.positions);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < entries; i += gridDim.x * blockDim.x) {
    const int64_t expert = topk_idx[i];
    if (expert >= 0) {
      const int at = atomicAdd(send + expert, 1);
      positions[i] = at;
      if (p.ordered_push)
        region<int>(p, l.send_lists)[expert * l.num_max_tokens * l.num_topk + at] = i;
    }
  }
}

// A barrier of the ranks. PUBLISH first writes this rank's counts into every rank, and after the
// barrier lays out the rows: this rank's first row in every expert's rows, and its own experts'
// end rows (LDS: the counts [ranks][experts])
template <bool PUBLISH> __device__ __forceinline__ void mega_moe_barrier_impl(const MegaMoeArgs &p) {
  using namespace mega_moe;
  const MegaMoeLayout &l = p.layout;
  const int ranks = l.num_ranks, experts = l.num_experts, per_rank = l.num_experts_per_rank, me = p.view.rank;
  if constexpr (PUBLISH)
    for (int i = threadIdx.x; i < ranks * experts; i += blockDim.x)
      uni_symm::device::store_relaxed<Scope::system>(
          p.view.map(region<int>(p, l.counts) + me * experts + i % experts, i / experts),
          region<const int>(p, l.send_counts)[i % experts]);
#if DG_MOE_COMPACT_BARRIER
  mega_moe::compact_barrier_workgroup(p.view, region<uint32_t>(p, l.barrier + p.barrier_slot * 128),
                                      region<int>(p, l.barrier + p.barrier_slot * 128 + 64),
                                      threadIdx.x);
#else
  uni_symm::device::barrier_workgroup(p.view, region<uint32_t>(p, l.barrier + p.barrier_slot * 128),
                                      region<int>(p, l.barrier + p.barrier_slot * 128 + 64),
                                      threadIdx.x);
#endif
  if constexpr (PUBLISH) {
    extern __shared__ int smem[];
    for (int i = threadIdx.x; i < ranks * experts; i += blockDim.x)
      smem[i] = uni_symm::device::load_relaxed<Scope::system>(region<const int>(p, l.counts) + i);
    __syncthreads();
    // one thread per destination rank walks its experts
    for (int d = threadIdx.x; d < ranks; d += blockDim.x) {
      int start = 0;
      for (int le = 0; le < per_rank; ++le) {
        const int e = d * per_rank + le;
        int before = 0, total = 0;
        for (int s = 0; s < ranks; ++s)
          before += s < me ? smem[s * experts + e] : 0, total += smem[s * experts + e];
        region<int>(p, l.dst_rows)[e] = start + before;
        if (d == me) {
          region<int>(p, l.psum_ends)[le] = start + total;
          if (p.stats != nullptr)
            p.stats[le] += total;
        }
        start += (total + l.alignment - 1) / l.alignment * l.alignment;
      }
    }
    // the next call's route counts from zero
    for (int e = threadIdx.x; e < experts; e += blockDim.x)
      region<int>(p, l.send_counts)[e] = 0;
  }
}

// One wave per token: its row and record are written to the rows of its slots on the experts'
// ranks (remote writes; the record carries the scales, the top-k weight and the source)
__device__ __forceinline__ void mega_moe_push_impl(const MegaMoeArgs &p) {
  using namespace mega_moe;
  const MegaMoeLayout &l = p.layout;
  const int lane = threadIdx.x % 64, topk = l.num_topk, per_rank = l.num_experts_per_rank;
  const int token_bytes = l.hidden * l.act_bytes, words = l.with_sf ? l.hidden / 128 : 0;
  const int64_t *topk_idx = region<const int64_t>(p, l.topk_idx);
  const int *positions = region<const int>(p, l.positions), *dst_rows = region<const int>(p, l.dst_rows);
  for (int t = global_wave(); t < p.num_tokens; t += num_waves()) {
    // lane j < topk: slot j's rank and row
    int dst = -1, row = 0;
    if (lane < topk) {
      const int64_t e = topk_idx[int64_t(t) * topk + lane];
      if (e >= 0 && (!p.expert_count || (e % per_rank >= p.expert_begin &&
                                            e % per_rank < p.expert_begin + p.expert_count)))
        dst = static_cast<int>(e) / per_rank, row = dst_rows[e] + positions[t * topk + lane];
    }
    const uint8_t *src = region<const uint8_t>(p, l.x) + int64_t(t) * token_bytes;
    for (int offset = lane * 16; offset < token_bytes; offset += 64 * 16 * 8) {
      v4i v[8];
#pragma unroll
      for (int i = 0; i < 8; ++i)
        v[i] = offset + i * 64 * 16 < token_bytes ? *reinterpret_cast<const v4i *>(src + offset + i * 64 * 16) : v4i{};
      for (int j = 0; j < topk; ++j) {
        const int d = __shfl(dst, j), r = __shfl(row, j);
        if (d < 0)
          continue;
        uint8_t *out = p.view.map(region<uint8_t>(p, l.l1_acts) + int64_t(r) * token_bytes, d);
#pragma unroll
        for (int i = 0; i < 8; ++i)
          if (offset + i * 64 * 16 < token_bytes)
            *reinterpret_cast<v4i *>(out + offset + i * 64 * 16) = v[i];
      }
    }
    for (int j = 0; j < topk; ++j) {
      const int d = __shfl(dst, j), r = __shfl(row, j);
      if (d < 0)
        continue;
      int *record = p.view.map(region<int>(p, l.l1_records) + int64_t(r) * l.record_ints, d);
      for (int q = lane; q < l.record_ints / 4; q += 64) {
        v4i v;
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          const int w = 4 * q + c;
          v[c] = w < words ? region<const int>(p, l.x_sf)[int64_t(t) * words + w]
                 : w == words ? __float_as_int(region<const float>(p, l.topk_weights)[int64_t(t) * topk + j])
                 : w == words + 1 ? static_cast<int>((uint32_t(p.view.rank) * l.num_max_tokens + t) * topk + j)
                                  : 0;
        }
        *reinterpret_cast<v4i *>(record + 4 * q) = v;
      }
    }
  }
  uni_symm::device::fence_release<Scope::system>();
}

// Destination-major dispatch avoids scanning every token again for every expert window.
// The route's per-expert lists allow consecutive waves to rotate over destination links.
__device__ __forceinline__ void mega_moe_push_ordered_impl(const MegaMoeArgs &p) {
  using namespace mega_moe;
  const auto& l = p.layout;
  const int ranks = l.num_ranks, experts = l.num_experts_per_rank;
  const int begin = p.expert_count ? p.expert_begin : 0;
  const int end = p.expert_count ? begin + p.expert_count : experts;
  extern __shared__ int prefix[];
  int* most = prefix + ranks * (experts + 1);
  const int* counts = region<const int>(p, l.counts) + p.view.rank * l.num_experts;
  if (threadIdx.x < ranks) {
    const int d = threadIdx.x;
    int sum = 0;
    for (int e = 0; e < experts; ++e) {
      prefix[d * (experts + 1) + e] = sum;
      if (e >= begin && e < end) sum += counts[d * experts + e];
    }
    prefix[d * (experts + 1) + experts] = sum;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    int max_rows = 0;
    for (int d = 0; d < ranks; ++d) max_rows = max(max_rows, prefix[d * (experts + 1) + experts]);
    *most = max_rows;
  }
  __syncthreads();
  const int lane = threadIdx.x % 64, token_bytes = l.hidden * l.act_bytes;
  const int words = l.with_sf ? l.hidden / 128 : 0;
  for (int k = global_wave(); k < ranks * *most; k += num_waves()) {
    const int d = (k % ranks + p.view.rank) % ranks, j = k / ranks;
    const int* pref = prefix + d * (experts + 1);
    if (j >= pref[experts]) continue;
    int lo = begin, hi = end - 1;
    while (lo < hi) {
      const int mid = (lo + hi + 1) / 2;
      if (pref[mid] <= j) lo = mid; else hi = mid - 1;
    }
    const int expert = d * experts + lo, at = j - pref[lo];
    const int entry = region<const int>(p, l.send_lists)[int64_t(expert) * l.num_max_tokens * l.num_topk + at];
    const int t = entry / l.num_topk;
    const int r = region<const int>(p, l.dst_rows)[expert] + at;
    wave_copy(p.view.map(region<uint8_t>(p, l.l1_acts) + int64_t(r) * token_bytes, d),
              region<const uint8_t>(p, l.x) + int64_t(t) * token_bytes, token_bytes, lane);
    int* record = p.view.map(region<int>(p, l.l1_records) + int64_t(r) * l.record_ints, d);
    for (int q = lane; q < l.record_ints / 4; q += 64) {
      v4i v;
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const int w = 4 * q + c;
        v[c] = w < words ? region<const int>(p, l.x_sf)[int64_t(t) * words + w]
               : w == words ? __float_as_int(region<const float>(p, l.topk_weights)[entry])
               : w == words + 1 ? static_cast<int>(uint32_t(p.view.rank) * l.num_max_tokens * l.num_topk + entry) : 0;
      }
      *reinterpret_cast<v4i*>(record + 4 * q) = v;
    }
  }
  uni_symm::device::fence_release<Scope::system>();
}

namespace mega_moe {
// The rows of a finish / swiglu launch: the shared expert's (the local tokens; the routed layout
// may not exist yet on the shared expert's stream) or the routed rows (LDS: the experts' starts)
struct Rows {
  bool shared;
  int count, experts;
  const int *starts, *ends;
  int begin = 0;

  __device__ bool valid(int row) const { return shared || row < ends[expert_of_row(starts, experts, row)]; }
};

__device__ __forceinline__ Rows launch_rows(const MegaMoeArgs &p, int *smem) {
  const MegaMoeLayout &l = p.layout;
  Rows rows{p.shared_part != 0, p.num_tokens, l.num_experts_per_rank, smem, region<const int>(p, l.psum_ends)};
  if (rows.shared)
    return rows;
  for (int e = threadIdx.x; e < rows.experts; e += blockDim.x)
    smem[e] = expert_start(rows.ends, e, l.alignment);
  __syncthreads();
  rows.begin = p.expert_count ? expert_start(rows.ends, p.expert_begin, l.alignment) : 0;
  rows.count = expert_start(rows.ends, p.expert_count ? p.expert_begin + p.expert_count : rows.experts, l.alignment);
  return rows;
}
}  // namespace mega_moe

// The routed rows' scales from their records into the GEMM's MN-major layout, or the local
// tokens' scales for the shared expert (with_sf only)
__device__ __forceinline__ void mega_moe_finish_impl(const MegaMoeArgs &p) {
  using namespace mega_moe;
  const MegaMoeLayout &l = p.layout;
  extern __shared__ int smem[];
  const int lane = threadIdx.x % 64, words = l.hidden / 128;
  uni_symm::device::fence_acquire<Scope::system>();
  const Rows rows = launch_rows(p, smem);
  for (int row = rows.begin + global_wave(); row < rows.count; row += num_waves()) {
    const bool shared = rows.shared;
    if (!rows.valid(row))
      continue;
    if (!shared && p.direct_combine && lane == 0) {
      const uint32_t source = region<const uint32_t>(p, l.l1_records)[int64_t(row) * l.record_ints +
          (l.with_sf ? l.hidden / 128 : 0) + 1];
      const int slot = source % l.num_topk, token = source / l.num_topk % l.num_max_tokens;
      const int rank = source / l.num_topk / l.num_max_tokens;
      region<int64_t>(p, l.output_rows)[row] = reinterpret_cast<int64_t>(p.view.map(
          region<uint16_t>(p, l.combine) + (int64_t(slot) * l.num_max_tokens + token) * l.hidden, rank));
    }
    if (!l.with_sf) continue;
    const int *src = shared ? region<const int>(p, l.x_sf) + int64_t(row) * words
                            : region<const int>(p, l.l1_records) + int64_t(row) * l.record_ints;
    int *dst = region<int>(p, shared ? l.shared_l1_sf : l.l1_sf);
    const int pitch = shared ? l.shared_sf_pitch : l.sf_pitch;
    for (int w = lane; w < words; w += 64)
      dst[int64_t(w) * pitch + row] = src[w];
  }
}

namespace mega_moe {
// NV's UE8M0 scale exponent of an E4M3 cast: ceil(log2(amax / 448)) through the integer carry,
// at least 2^-22 (the amax floor 1e-4 of the per-token cast)
__device__ __forceinline__ uint32_t ue8m0_exponent(float amax) {
  const uint32_t rounded = (__float_as_uint(amax) + 0x7fffffu - (0x60u << 16)) >> 23;
  return max(rounded, 105u + 8u) - 8u;
}

__device__ __forceinline__ float bf16_round(float x) {
  float r, unused;
  unpack_bf16x2(pack_bf16x2(x, 0.f), r, unused);
  return r;
}
}  // namespace mega_moe

// One wave per row (the routed rows, or the local tokens of the shared expert); lane q takes the
// 32 activations q (64 L1 columns: four blocks of 8 gate and 8 up columns)
template <bool WITH_SF> __device__ __forceinline__ void mega_moe_swiglu_impl(const MegaMoeArgs &p) {
  using namespace mega_moe;
  const MegaMoeLayout &l = p.layout;
  extern __shared__ int smem[];
  const int lane = threadIdx.x % 64;
  const Rows rows = launch_rows(p, smem);
  const float clamp = bf16_round(p.activation_clamp);
  for (int row = rows.begin + global_wave(); row < rows.count; row += num_waves()) {
    const bool shared = rows.shared;
    if (!rows.valid(row))
      continue;
    const int inter = shared ? l.shared_intermediate : l.intermediate;
    const uint16_t *in = (shared ? p.shared_l1_out : p.l1_out) + int64_t(row) * 2 * inter;
    uint8_t *out = region<uint8_t>(p, shared ? l.shared_l2_acts : l.l2_acts) + int64_t(row) * inter * l.act_bytes;
    int *sf = region<int>(p, shared ? l.shared_l2_sf : l.l2_sf);
    const int pitch = shared ? l.shared_sf_pitch : l.sf_pitch;
    const float weight = shared ? 1.f
                                : __int_as_float(region<const int>(p, l.l1_records)[int64_t(row) * l.record_ints +
                                                                                    (l.with_sf ? l.hidden / 128 : 0)]);
    for (int q0 = 0; q0 < inter / 32; q0 += 64) {
      const int q = q0 + lane;
      const bool active = q < inter / 32;
      float act[32];
      float amax = 0.f;
      if (active) {
        v4i raw[8];
#pragma unroll
        for (int i = 0; i < 8; ++i)
          raw[i] = reinterpret_cast<const v4i *>(in + 64 * q)[i];
#pragma unroll
        for (int b = 0; b < 4; ++b)
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            float g[2], u[2];
            unpack_bf16x2(static_cast<uint32_t>(raw[2 * b][i]), g[0], g[1]);
            unpack_bf16x2(static_cast<uint32_t>(raw[2 * b + 1][i]), u[0], u[1]);
#pragma unroll
            for (int e = 0; e < 2; ++e) {
              const float gate = fminf(g[e], clamp), up = fminf(fmaxf(u[e], -clamp), clamp);
              const float denominator = 1.f + (p.fast_math ? __expf(-gate) : expf(-gate));
              const float silu = p.fast_math ? gate * __builtin_amdgcn_rcpf(denominator) : gate / denominator;
              act[8 * b + 2 * i + e] = silu * up * weight;
              amax = fmaxf(amax, fabsf(act[8 * b + 2 * i + e]));
            }
          }
      }
      if constexpr (WITH_SF) {
        const uint32_t exponent = ue8m0_exponent(amax);
        const float inverse = __uint_as_float((254u - exponent) << 23);
        if (active) {
          v4i packed[2];
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            const int lo = __builtin_amdgcn_cvt_pk_fp8_f32(act[4 * i] * inverse, act[4 * i + 1] * inverse, 0, false);
            packed[i / 4][i % 4] =
                __builtin_amdgcn_cvt_pk_fp8_f32(act[4 * i + 2] * inverse, act[4 * i + 3] * inverse, lo, true);
          }
          reinterpret_cast<v4i *>(out + 32 * q)[0] = packed[0];
          reinterpret_cast<v4i *>(out + 32 * q)[1] = packed[1];
        }
        // the scale bytes of four consecutive groups make a word
        uint32_t word = exponent;
#pragma unroll
        for (int b = 1; b < 4; ++b)
          word |= static_cast<uint32_t>(__shfl(static_cast<int>(exponent), (lane & ~3) + b)) << (8 * b);
        if (active && q % 4 == 0)
          sf[int64_t(q / 4) * pitch + row] = static_cast<int>(word);
      } else if (active) {
        v4i packed[4];
#pragma unroll
        for (int i = 0; i < 16; ++i)
          packed[i / 4][i % 4] = static_cast<int>(pack_bf16x2(act[2 * i], act[2 * i + 1]));
#pragma unroll
        for (int i = 0; i < 4; ++i)
          reinterpret_cast<v4i *>(out + 64 * q)[i] = packed[i];
      }
    }
  }
}

// LDS: the receive table (the rows in the pull's order, so concurrent waves write to every peer)
__device__ __forceinline__ void mega_moe_combine_impl(const MegaMoeArgs &p) {
  using namespace mega_moe;
  const MegaMoeLayout &l = p.layout;
  extern __shared__ int smem[];
  const int ranks = l.num_ranks, experts = l.num_experts_per_rank, lane = threadIdx.x % 64;
  const RowTable table = build_row_table(p, smem);
  const int items = ranks * *table.max_from_source, row_bytes = l.hidden * 2;
  for (int k = global_wave(); k < items; k += num_waves()) {
    int s, e;
    const int r = table.row_of(k, ranks, experts, s, e);
    if (r < 0 || (p.expert_count && (e < p.expert_begin || e >= p.expert_begin + p.expert_count)))
      continue;
    const uint32_t source = static_cast<uint32_t>(
        region<const int>(p, l.l1_records)[int64_t(r) * l.record_ints + (l.with_sf ? l.hidden / 128 : 0) + 1]);
    const int slot = source % l.num_topk, token = (source / l.num_topk) % l.num_max_tokens;
    const int rank = source / l.num_topk / l.num_max_tokens;
    uint8_t *dst = p.view.map(region<uint8_t>(p, l.combine) + (int64_t(slot) * l.num_max_tokens + token) * row_bytes,
                              rank);
    wave_copy(dst, reinterpret_cast<const uint8_t *>(p.l2_out + int64_t(r) * l.hidden), row_bytes, lane);
  }
  uni_symm::device::fence_release<Scope::system>();
}

// One wave per token: 8 values per lane and step
__device__ __forceinline__ void mega_moe_reduce_impl(const MegaMoeArgs &p) {
  using namespace mega_moe;
  const MegaMoeLayout &l = p.layout;
  const int lane = threadIdx.x % 64;
  uni_symm::device::fence_acquire<Scope::system>();
  const int64_t *topk_idx = region<const int64_t>(p, l.topk_idx);
  const uint16_t *combine = region<const uint16_t>(p, l.combine);
  for (int t = global_wave(); t < p.num_tokens; t += num_waves()) {
    for (int h = lane * 8; h < l.hidden; h += 64 * 8) {
      float acc[8] = {};
      for (int slot = 0; slot < l.num_topk; ++slot) {
        if (topk_idx[int64_t(t) * l.num_topk + slot] < 0)
          continue;
        const v4i v = *reinterpret_cast<const v4i *>(combine + (int64_t(slot) * l.num_max_tokens + t) * l.hidden + h);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          float a, b;
          unpack_bf16x2(static_cast<uint32_t>(v[i]), a, b);
          acc[2 * i] += a, acc[2 * i + 1] += b;
        }
      }
      if (p.shared_out != nullptr) {
        const v4i v = *reinterpret_cast<const v4i *>(p.shared_out + int64_t(t) * l.hidden + h);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          float a, b;
          unpack_bf16x2(static_cast<uint32_t>(v[i]), a, b);
          acc[2 * i] += a, acc[2 * i + 1] += b;
        }
      }
      v4i out;
#pragma unroll
      for (int i = 0; i < 4; ++i)
        out[i] = static_cast<int>(pack_bf16x2(acc[2 * i], acc[2 * i + 1]));
      *reinterpret_cast<v4i *>(p.y + int64_t(t) * l.hidden + h) = out;
    }
  }
}
