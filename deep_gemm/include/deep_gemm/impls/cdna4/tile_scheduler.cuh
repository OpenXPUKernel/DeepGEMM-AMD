// Output-tile ordering shared by the GEMM execution modes, and the mapping of
// logical tile ids onto the groups of grouped GEMMs.
#pragma once
#include "gemm_args.h"
#include <hip/hip_runtime.h>

using deep_gemm::cdna4::GemmType;

// Map a linear block id to (tile_m, tile_n): groups of GROUP_M m-tiles walk down N together so that
// the WGs running at the same time share A rows and B columns in L2 / MALL.
template <int GROUP_M>
__device__ __forceinline__ void tile_coords(int bid, int tiles_m, int tiles_n, int &tm, int &tn) {
  const int per_group = GROUP_M * tiles_n;
  const int g = bid / per_group, first_m = g * GROUP_M;
  const int gm = min(tiles_m - first_m, GROUP_M);
  const int r = bid % per_group;
  tm = first_m + r % gm;
  tn = r / gm;
}

// Renumber interleaved workgroups into contiguous tile ranges for operand reuse.
__device__ __forceinline__ int xcd_remap(int bid, int tiles) {
  constexpr int NXCD = 8;
  const int per = (tiles + NXCD - 1) / NXCD; // tiles per XCD (the first `tall` get one more)
  const int tall = tiles % NXCD == 0 ? NXCD : tiles % NXCD;
  const int xcd = bid % NXCD, local = bid / NXCD;
  return xcd < tall ? xcd * per + local : tall * per + (xcd - tall) * (per - 1) + local;
}

__device__ __forceinline__ int align_up(int x, int a) { return (x + a - 1) / a * a; }

// Wave-wide (64 lanes) scans. Every lane of the wave must be active.
__device__ __forceinline__ int wave_inclusive_sum(int v) {
  const int lane = __lane_id();
#pragma unroll
  for (int d = 1; d < 64; d <<= 1) {
    const int t = __shfl_up(v, d, 64);
    v += lane >= d ? t : 0;
  }
  return v;
}
__device__ __forceinline__ int wave_sum(int v) {
#pragma unroll
  for (int d = 32; d > 0; d >>= 1)
    v += __shfl_xor(v, d, 64);
  return v;
}

// A tile of the output: origin rows m0 .. m0 + BM - 1 (within group g's rows for masked GEMMs, of
// D otherwise) and columns n0 .. n0 + BN - 1. Rows >= m_end are not stored; rows m_data ..
// m_end - 1 are the zero padding of a PSUM group. group < 0: no tile.
struct GroupTile {
  int m0, n0, group, m_end, m_data;
};

// K range of group g of a K-grouped GEMM: K values k0 .. k0 + kg - 1 of A and B, and scale words
// w0 .. of the group's packed scales (each word covers 4 * gran_k K values).
struct KGroup {
  int k0, kg, w0;
};

// The group that holds logical tile `lid`, given per-group tile counts count(g): the first group
// whose inclusive prefix exceeds lid. before = tiles of the groups ahead of it. group = G (and
// before = all tiles) if lid is past the last tile.
template <class Count>
__device__ __forceinline__ void find_group(int G, int lid, Count count, int &group, int &before) {
  const int lane = __lane_id();
  int carry = 0;
  for (int base = 0; base < G; base += 64) {
    const int gi = base + lane, n = min(64, G - base);
    const int incl = wave_inclusive_sum(gi < G ? count(gi) : 0) + carry;
    const int done = __popcll(__ballot(gi < G && incl <= lid));
    if (done < n) {
      group = base + done;
      before = done ? __shfl(incl, done - 1, 64) : carry;
      group = __builtin_amdgcn_readfirstlane(group);
      before = __builtin_amdgcn_readfirstlane(before);
      return;
    }
    carry = __builtin_amdgcn_readfirstlane(__shfl(incl, n - 1, 64));
  }
  group = G;
  before = carry;
}

template <GemmType TYPE, int BM, int BN, int GROUP_M> struct TileScheduler {
  static constexpr bool PSUM = TYPE == GemmType::MGroupedContiguousPsum;
  static constexpr bool MASKED = TYPE == GemmType::MGroupedMasked;
  static constexpr bool K_GROUPED =
      TYPE == GemmType::KGroupedContiguous || TYPE == GemmType::KGroupedContiguousPsum;
  // Tile counts are only known on the device; a tile lookup walks the group table.
  static constexpr bool DYNAMIC = PSUM || MASKED;

  // PSUM group g: rows start .. end - 1 hold data; with zero padding the rows up to the next
  // aligned row are stored too
  __device__ static int psum_start(const GemmArgs &p, int g) {
    return g ? align_up(p.grouped_layout[g - 1], p.m_alignment) : 0;
  }
  __device__ static int psum_stored_end(const GemmArgs &p, int g) {
    const int end = p.grouped_layout[g];
    return p.zero_padding ? min(p.M, align_up(end, p.m_alignment)) : end;
  }
  // m-tiles of group g (masked / psum)
  __device__ static int group_tiles_m(const GemmArgs &p, int g) {
    if constexpr (MASKED)
      return (p.grouped_layout[g] + BM - 1) / BM;
    else
      return max(0, (psum_stored_end(p, g) - psum_start(p, g) + BM - 1) / BM);
  }

  __device__ static int num_tiles(const GemmArgs &p) {
    const int tiles_m = (p.M + BM - 1) / BM, tiles_n = (p.N + BN - 1) / BN;
    if constexpr (DYNAMIC) {
      int group, before;
      find_group(
          p.num_groups, 0x7fffffff, [&](int g) { return group_tiles_m(p, g) * tiles_n; }, group,
          before);
      return before;
    } else if constexpr (K_GROUPED) {
      return p.num_groups * tiles_m * tiles_n;
    } else {
      return tiles_m * tiles_n;
    }
  }

  __device__ static GroupTile tile(const GemmArgs &p, int lid) {
    const int tiles_m = (p.M + BM - 1) / BM, tiles_n = (p.N + BN - 1) / BN;
    GroupTile t{0, 0, 0, p.M, p.M};
    int tm, tn;
    if constexpr (DYNAMIC) {
      int before;
      find_group(
          p.num_groups, lid, [&](int g) { return group_tiles_m(p, g) * tiles_n; }, t.group,
          before);
      if (t.group >= p.num_groups) {
        t.group = -1;
        return t;
      }
      tile_coords<GROUP_M>(lid - before, group_tiles_m(p, t.group), tiles_n, tm, tn);
      t.m0 = tm * BM;
      if constexpr (MASKED) {
        t.m_end = t.m_data = p.grouped_layout[t.group];
      } else {
        t.m0 += psum_start(p, t.group);
        t.m_end = psum_stored_end(p, t.group);
        t.m_data = p.grouped_layout[t.group];
      }
    } else if constexpr (K_GROUPED) {
      const int per_group = tiles_m * tiles_n;
      t.group = lid / per_group;
      tile_coords<GROUP_M>(lid - t.group * per_group, tiles_m, tiles_n, tm, tn);
      t.m0 = tm * BM;
    } else {
      tile_coords<GROUP_M>(lid, tiles_m, tiles_n, tm, tn);
      t.m0 = tm * BM;
      // padding rows (-1) compute with group 0 like the CUDA backend
      if constexpr (TYPE == GemmType::MGroupedContiguous)
        t.group = max(0, p.grouped_layout[t.m0]);
    }
    t.n0 = tn * BN;
    return t;
  }

  // sf_span: K values per packed scale word
  __device__ static KGroup k_group(const GemmArgs &p, int g, int sf_span) {
    static_assert(K_GROUPED, "K ranges only exist for K-grouped GEMMs");
    constexpr bool K_PSUM = TYPE == GemmType::KGroupedContiguousPsum;
    const int lane = __lane_id(), ka = p.k_alignment;
    // K of group i: padded (non-psum), or logical end - aligned start
    auto group_k = [&](int i) {
      if constexpr (K_PSUM)
        return p.grouped_layout[i] - (i ? align_up(p.grouped_layout[i - 1], ka) : 0);
      else
        return p.grouped_layout[i];
    };
    int k_sum = 0, w_sum = 0;
    for (int base = 0; base < g; base += 64) {
      const int i = base + lane;
      if (i < g) {
        const int ki = group_k(i);
        k_sum += ki;
        w_sum += (align_up(ki, ka) + sf_span - 1) / sf_span;
      }
    }
    KGroup r;
    r.w0 = __builtin_amdgcn_readfirstlane(wave_sum(w_sum));
    if constexpr (K_PSUM) {
      r.k0 = g ? align_up(p.grouped_layout[g - 1], ka) : 0;
      r.kg = p.grouped_layout[g] - r.k0;
    } else {
      r.k0 = __builtin_amdgcn_readfirstlane(wave_sum(k_sum));
      r.kg = p.grouped_layout[g];
    }
    return r;
  }
};
