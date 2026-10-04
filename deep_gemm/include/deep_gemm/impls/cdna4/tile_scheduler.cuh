// Shared output-tile ordering for the tiled and pipelined GEMMs.
#pragma once
#include <hip/hip_runtime.h>

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
