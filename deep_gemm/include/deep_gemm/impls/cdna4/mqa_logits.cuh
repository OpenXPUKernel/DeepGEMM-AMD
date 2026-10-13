// CDNA4 MQA logits: logits[i, n] = sum_h weights[i, h] relu(q[i, h] . kv[n]), times the KV scale
// of n with the Fp8 format.
//
// One wave takes QW q tokens and chunks of their KV positions (columns). Its QW x H (token, head)
// rows are the rows of T 16x16x128 scaled-MFMA tiles, staged once in the wave's LDS in operand
// order; a KV row is one column of an MFMA. Row i = 4 g + e of a tile lands in register e of lane group g, so the rows are
// ordered to put the heads of one token into the registers of L lane groups: the weighted ReLU
// sum over the heads runs inside the lanes, plus a butterfly over the L lane groups (L > 1 when a
// wave has fewer than four tokens, as in decoding). The wave loads its KV rows from global memory
// straight into registers: no LDS, no barriers. Column 4 r16 + j of a step of 64 is column r16 of
// MFMA j, so after a step every lane holds four adjacent logits of a row.
#pragma once
#include "device_common.h"
#include "mqa_logits_args.h"

using deep_gemm::cdna4::MqaLogitsArgs;
using deep_gemm::cdna4::MqaLogitsFormat;
using deep_gemm::cdna4::MqaLogitsSource;

// D values per head, H heads, L lane groups per token, U groups of 4 / L tokens per wave, CHUNK
// KV positions (columns) per work item
template <MqaLogitsFormat FMT, MqaLogitsSource SRC, int D_, int H_, int L_, int U_, int CHUNK_ = 2048>
struct MqaLogitsConfig {
  static constexpr int D = D_, H = H_, L = L_, U = U_, CHUNK = CHUNK_;
  static constexpr bool MX = FMT != MqaLogitsFormat::Fp8, FP4 = FMT == MqaLogitsFormat::MxFp4;
  static constexpr bool PAGED = SRC == MqaLogitsSource::Paged || SRC == MqaLogitsSource::PagedSparse;
  static constexpr bool SPARSE = SRC == MqaLogitsSource::Sparse || SRC == MqaLogitsSource::PagedSparse;
  static_assert(L == 1 || L == 2 || L == 4, "one, two or four lane groups per token");
  static_assert(!SPARSE || (L == 4 && U == 1), "a sparse token has its own KV rows: one token per wave");
  // bytes of a Q / KV row; an MFMA takes 128 values, the rest of a shorter row is zero
  static constexpr int ROW_BYTES = FP4 ? D / 2 : D;
  static_assert(ROW_BYTES == 32 || ROW_BYTES == 64 || ROW_BYTES == 128, "32 to 128 values per head");
  static constexpr int SCALE_BLOCKS = D / 32;
  static constexpr int FMT_CODE = FP4 ? kFmtFp4 : kFmtFp8;
  // tiles: U groups of TU tiles, each holding the heads of 4 / L tokens
  static constexpr int H_PAD = (H + 4 * L - 1) / (4 * L) * (4 * L);
  static constexpr int TU = H_PAD / (4 * L), T = U * TU, QW = U * 4 / L;
  static constexpr int J = 4, STEP = 16 * J;
  // the WAVES waves of a workgroup share the Q tiles and take quarters of a chunk
  static constexpr int WAVES = 4, NT = 64 * WAVES, WAVE_CHUNK = CHUNK / WAVES;
  static_assert(WAVE_CHUNK % STEP == 0, "whole steps per wave and chunk");
  // LDS of a workgroup: the Q operand halves of its tiles (1 KB per tile and half), their codes,
  // and (paged / sparse) each wave's row table of its part of the chunk: its pages (pages of 32
  // rows or more) or its sparse blocks' first rows (blocks of 8 rows or more)
  static constexpr int HALVES = FP4 || ROW_BYTES <= 64 ? 1 : 2;
  static constexpr int CODES_OFF = HALVES * T * 1024;
  static constexpr int TABLE_OFF = CODES_OFF + (MX ? T * 256 : 0);
  static constexpr int TABLE_ENTRIES = SPARSE ? WAVE_CHUNK / 8 + 1 : WAVE_CHUNK / 32 + 1;
  static constexpr int TABLE_PER_LANE = (TABLE_ENTRIES + 63) / 64;
  static constexpr int LDS_BYTES = TABLE_OFF + (PAGED || SPARSE ? WAVES * 4 * TABLE_ENTRIES : 0);
  // token (within the wave) and head of row 4 g + e of tile u TU + tt
  __host__ __device__ static constexpr int token_of(int u, int g) { return u * (4 / L) + g / L; }
  __host__ __device__ static constexpr int head_of(int tt, int g, int e) { return 4 * L * tt + 4 * (g % L) + e; }
  // operand half h of lane group g: row bytes 16 g + 64 h .. + 15, if they exist
  __host__ __device__ static constexpr bool half_exists(int g, int h) {
    return (FP4 ? h == 0 : true) && 16 * g + 64 * h + 16 <= ROW_BYTES;
  }
};

template <class Cfg, class OutT>
__device__ __forceinline__ void mqa_logits_impl(const MqaLogitsArgs &p) {
#if defined(__HIP_DEVICE_COMPILE__)
  constexpr int T = Cfg::T, TU = Cfg::TU, U = Cfg::U, L = Cfg::L, QW = Cfg::QW, J = Cfg::J, H = Cfg::H;
  constexpr int ROW_BYTES = Cfg::ROW_BYTES;
  const int lane = threadIdx.x % 64, g = lane / 16, r16 = lane % 16;
  const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x / 64);

  // ---- the workgroup's tokens q0 .. q0 + nq - 1 (Paged: of one request; a workgroup starts at
  // every QW-th token of a request)
  int q0, nq;
  if constexpr (Cfg::PAGED && !Cfg::SPARSE) {
    const int x = blockIdx.x;
    const int start = p.run_starts ? p.run_starts[x] : x / p.next_n * p.next_n;
    if ((x - start) % QW != 0)
      return;
    q0 = x;
    int n = 1;
    if (p.run_starts) {
      while (n < QW && x + n < p.num_q && p.run_starts[x + n] == start)
        ++n;
    } else {
      n = min(QW, start + p.next_n - x);
    }
    nq = n;
  } else {
    q0 = blockIdx.x * QW;
    nq = min(QW, p.num_q - q0);
  }

  // ---- columns [lo, hi) of the wave's tokens
  auto sparse_columns = [&](int tok) {
    const int len = Cfg::PAGED ? p.context_lens[tok] : p.ke[tok] - p.ks[tok];
    const int blocks = (max(len, 0) + p.sparse_block_kv - 1) / p.sparse_block_kv;
    return min(blocks, p.num_max_sparse_blocks) * p.sparse_block_kv;
  };
  int lo = 0, hi = 0;
  if constexpr (Cfg::SPARSE) {
    hi = sparse_columns(q0);
  } else if constexpr (Cfg::PAGED) {
    for (int i = 0; i < nq; ++i)
      hi = max(hi, p.context_lens[q0 + i]);
  } else if (p.group_ranges) {
    lo = p.group_ranges[2 * blockIdx.x];
    hi = p.group_ranges[2 * blockIdx.x + 1];
  } else {
    lo = 0x7fffffff;
    for (int i = 0; i < nq; ++i) {
      lo = min(lo, p.ks[q0 + i]);
      hi = max(hi, p.ke[q0 + i]);
    }
  }
  const int chunks = hi > lo ? (hi - lo + Cfg::CHUNK - 1) / Cfg::CHUNK : 0;
  if ((int)blockIdx.y >= chunks)
    return;

  // ---- Q tiles, staged in LDS in MFMA operand order: lane (g, r16) of every wave holds row r16 of
  // tile t (token token_of(u, r16 / 4), head head_of(tt, r16 / 4, r16 % 4)), bytes 16 g .. + 15
  // (half 0) and 64 + 16 g .. + 15 (half 1), and the scale code of its K block g. Half h of tile t
  // is a 1 KB array at (HALVES t + h) KB, 16 bytes per lane; codes follow, 4 bytes per lane and
  // tile. Wave w stages the tiles t % WAVES == w.
  extern __shared__ __attribute__((aligned(16))) uint8_t smem[];
  static_for<0, T>([&](auto I) {
    constexpr int t = decltype(I)::value, u = t / TU, tt = t % TU;
    if (t % Cfg::WAVES != wave)
      return;
    const int q = Cfg::token_of(u, r16 / 4), h = Cfg::head_of(tt, r16 / 4, r16 % 4);
    const bool valid = q < nq && h < H;
    const uint8_t *row = p.q + ((size_t)(q0 + q) * H + h) * ROW_BYTES + 16 * g;
#pragma unroll
    for (int hf = 0; hf < Cfg::HALVES; ++hf)
      *(v4i *)(smem + (Cfg::HALVES * t + hf) * 1024 + 16 * lane) =
          valid && Cfg::half_exists(g, hf) ? *(const v4i *)(row + 64 * hf) : v4i{};
    if constexpr (Cfg::MX)
      *(uint32_t *)(smem + Cfg::CODES_OFF + t * 256 + 4 * lane) =
          valid && g < Cfg::SCALE_BLOCKS ? (uint32_t)p.q_sf[(size_t)(q0 + q) * H + h] >> (8 * g) : 127u;
  });
  __syncthreads();
  // weights of the rows of lane group g: w[t][e] for token token_of(u, g), head head_of(tt, g, e)
  float w[T][4];
  static_for<0, T>([&](auto I) {
    constexpr int t = decltype(I)::value, u = t / TU, tt = t % TU;
    const int q = Cfg::token_of(u, g);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int h = Cfg::head_of(tt, g, e);
      float v = 0.f;
      if (q < nq && h < H) {
        const size_t idx = (size_t)(q0 + q) * p.ld_weights + h;
        v = Cfg::MX ? bf16_to_f32(((const uint16_t *)p.weights)[idx]) : ((const float *)p.weights)[idx];
      }
      w[t][e] = v;
    }
  });
  // the output rows of lane group g
  int tok_of_u[U], col_lo[U], col_hi[U], col_shift[U];
  bool stores[U];
#pragma unroll
  for (int u = 0; u < U; ++u) {
    const int q = Cfg::token_of(u, g), tok = q0 + min(q, nq - 1);
    tok_of_u[u] = tok;
    stores[u] = q < nq && g % L == 0;
    if constexpr (Cfg::SPARSE) {
      col_lo[u] = 0, col_hi[u] = sparse_columns(tok), col_shift[u] = 0;
    } else if constexpr (Cfg::PAGED) {
      col_lo[u] = 0, col_hi[u] = p.context_lens[tok], col_shift[u] = 0;
    } else {
      col_lo[u] = p.ks[tok], col_hi[u] = p.ke[tok], col_shift[u] = p.ks[tok];
    }
  }

  // ---- KV row of column n: its bytes and scale
  const int *block_table = nullptr;
  if constexpr (Cfg::PAGED && !Cfg::SPARSE)
    block_table = p.block_table + (size_t)(p.run_starts ? q0 : q0 / p.next_n) * p.block_table_stride;
  const int *sparse_row = nullptr;
  if constexpr (Cfg::SPARSE)
    sparse_row = p.sparse_starts + (size_t)q0 * p.num_max_sparse_blocks;
  // The wave's row table of its part of the current chunk, in LDS: entry i is that of page /
  // sparse block t0 + i. The next chunk's entries load into registers during this chunk
  // (TableLoad), and go to LDS when it starts.
  int *const table = (int *)(smem + Cfg::TABLE_OFF) + wave * Cfg::TABLE_ENTRIES;
  const int granule = Cfg::SPARSE ? p.sparse_block_kv : p.page_kv;
  int t0 = 0, c_end = 0;
  struct TableLoad {
    int entries[Cfg::TABLE_PER_LANE];
  };
  // the wave's columns [c_lo, c_hi) of chunk c (may be empty)
  auto chunk_range = [&](int c, int &c_lo, int &c_hi) {
    c_lo = lo + c * Cfg::CHUNK + wave * Cfg::WAVE_CHUNK;
    c_hi = min(hi, c_lo + Cfg::WAVE_CHUNK);
  };
  auto load_table = [&](int c) {
    TableLoad t{};
    if (c < chunks) {
      int c_lo, c_hi;
      chunk_range(c, c_lo, c_hi);
      const int first = c_lo / granule, entries = (c_hi - 1) / granule - first + 1;
      const int *src = Cfg::SPARSE ? sparse_row : block_table;
#pragma unroll
      for (int k = 0; k < Cfg::TABLE_PER_LANE; ++k)
        t.entries[k] = lane + 64 * k < entries ? src[first + lane + 64 * k] : 0;
    }
    return t;
  };
  auto store_table = [&](const TableLoad &t, int c_lo, int c_hi) {
    t0 = c_lo / granule;
    c_end = c_hi;
#pragma unroll
    for (int k = 0; k < Cfg::TABLE_PER_LANE; ++k)
      if (lane + 64 * k < Cfg::TABLE_ENTRIES)
        table[lane + 64 * k] = t.entries[k];
  };
  struct KvRow {
    const uint8_t *data;
    const void *scale;
  };
  auto kv_row = [&](int n) -> KvRow {
    // KV position (Contiguous / Paged) or column (sparse); past the chunk (prefetches), its last
    int pos = Cfg::PAGED || Cfg::SPARSE ? min(n, c_end - 1) : min(n, hi - 1);
    if constexpr (Cfg::SPARSE) // the block's first row, paged: page * page_kv + row in the page
      pos = table[pos / p.sparse_block_kv - t0] + pos % p.sparse_block_kv;
    if constexpr (Cfg::PAGED) {
      pos = max(pos, 0);
      const int page_index = Cfg::SPARSE ? pos / p.page_kv : table[pos / p.page_kv - t0];
      const uint8_t *page = p.kv + (int64_t)page_index * p.page_stride;
      const int r = pos % p.page_kv;
      return {page + r * ROW_BYTES, page + p.page_kv * ROW_BYTES + 4 * r};
    } else {
      pos = min(max(pos, 0), p.num_kv - 1);
      return {p.kv + (size_t)pos * ROW_BYTES, (const uint8_t *)p.kv_sf + 4 * (size_t)pos};
    }
  };

  // KV rows n0 + 4 r16 + j of a step (rows past the columns repeat the last one)
  struct KvStep {
    v4i kf[J][2];
    uint32_t ks[J];
    float kv_scale[J];
  };
  auto load_step = [&](int n0) {
    KvStep s;
    static_for<0, J>([&](auto JJ) {
      constexpr int j = decltype(JJ)::value;
      const KvRow row = kv_row(n0 + 4 * r16 + j);
#pragma unroll
      for (int hf = 0; hf < 2; ++hf)
        s.kf[j][hf] = Cfg::half_exists(g, hf) ? *(const v4i *)(row.data + 16 * g + 64 * hf) : v4i{};
      if constexpr (Cfg::MX) {
        s.ks[j] = g < Cfg::SCALE_BLOCKS ? *(const uint32_t *)row.scale >> (8 * g) : 127u;
        s.kv_scale[j] = 1.f;
      } else {
        s.ks[j] = 127u;
        s.kv_scale[j] = *(const float *)row.scale;
      }
    });
    return s;
  };

  OutT *const logits = (OutT *)p.logits;
  TableLoad next_table{};
  if constexpr (Cfg::PAGED || Cfg::SPARSE)
    next_table = load_table(blockIdx.y);
  for (int c = blockIdx.y; c < chunks; c += gridDim.y) {
    int c_lo, c_hi;
    chunk_range(c, c_lo, c_hi);
    if constexpr (Cfg::PAGED || Cfg::SPARSE) {
      store_table(next_table, c_lo, c_hi);
      next_table = load_table(c + gridDim.y);
    }
    if (c_lo >= c_hi)
      continue;
    KvStep kv_cur = load_step(c_lo);
    for (int n0 = c_lo; n0 < c_hi; n0 += Cfg::STEP) {
      // the next step's KV rows load during this step's MFMAs
      const KvStep kv_next = load_step(n0 + Cfg::STEP);
      __builtin_amdgcn_sched_barrier(0);
      const auto &kf = kv_cur.kf;
      const auto &ks = kv_cur.ks;
      const auto &kv_scale = kv_cur.kv_scale;
      // tile by tile: its Q operand from LDS, its four MFMAs, their weighted ReLUs into the partial
      // sums. Every kernel adds a token's heads in one order, so that all sources give the same
      // bits: class c holds the heads h with (h / 4) % 4 == c, two packed partials of it the heads
      // with h % 2 == 0 and 1 in increasing order; then p_c = x + y and (p0 + p1) + (p2 + p3). Lane
      // group g of tile tt holds class (L tt + g % L) % 4.
      v2f part[U][J][4 / L];
#pragma unroll
      for (int u = 0; u < U; ++u)
#pragma unroll
        for (int j = 0; j < J; ++j)
#pragma unroll
          for (int c = 0; c < 4 / L; ++c)
            part[u][j][c] = v2f{};
      // (the next tile's operand is read during this one's; the scheduler may not hoist further,
      // or the accumulators of all tiles become live at once)
      struct QTile {
        v4i halves[2];
        uint32_t code;
      };
      auto read_tile = [&](int t) {
        QTile q;
#pragma unroll
        for (int hf = 0; hf < 2; ++hf)
          q.halves[hf] = hf < Cfg::HALVES ? *(const v4i *)(smem + (Cfg::HALVES * t + hf) * 1024 + 16 * lane) : v4i{};
        q.code = Cfg::MX ? *(const uint32_t *)(smem + Cfg::CODES_OFF + t * 256 + 4 * lane) : 127u;
        return q;
      };
      // Software pipeline: iteration t issues the MFMAs of tile t and the weighted ReLUs of tile
      // t - 1, interleaved (one MFMA per 8 VALU instructions: a ReLU and an FMA per result).
      v4f acc[2][J];
      QTile cur = read_tile(0);
      static_for<0, T + 1>([&](auto I) {
        constexpr int t = decltype(I)::value;
        QTile next{};
        if constexpr (t + 1 < T)
          next = read_tile(t + 1);
        if constexpr (t < T) {
          static_for<0, J>([&](auto JJ) {
            constexpr int j = decltype(JJ)::value;
            acc[t % 2][j] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
                scaled_operand<Cfg::FP4>(cur.halves), scaled_operand<Cfg::FP4>(kf[j]), v4f{}, Cfg::FMT_CODE,
                Cfg::FMT_CODE, 0, cur.code, 0, ks[j]);
          });
        }
        if constexpr (t >= 1) {
          constexpr int tp = t - 1, u = tp / TU, cls = (tp % TU) % (4 / L);
#pragma unroll
          for (int j = 0; j < J; ++j) {
            const v4f &a = acc[tp % 2][j];
            const v2f r01 = {fmaxf(a[0], 0.f), fmaxf(a[1], 0.f)}, r23 = {fmaxf(a[2], 0.f), fmaxf(a[3], 0.f)};
            part[u][j][cls] = __builtin_elementwise_fma(v2f{w[tp][0], w[tp][1]}, r01, part[u][j][cls]);
            part[u][j][cls] = __builtin_elementwise_fma(v2f{w[tp][2], w[tp][3]}, r23, part[u][j][cls]);
          }
        }
        if constexpr (t >= 1 && t < T) {
          static_for<0, J>([&](auto) {
            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
            __builtin_amdgcn_sched_group_barrier(0x002, 6, 0); // VALU: 4 ReLUs, 2 packed FMAs
          });
        }
        __builtin_amdgcn_sched_barrier(0);
        cur = next;
      });
      float out[U][J];
#pragma unroll
      for (int u = 0; u < U; ++u)
#pragma unroll
        for (int j = 0; j < J; ++j) {
          float p[4 / L];
#pragma unroll
          for (int c = 0; c < 4 / L; ++c)
            p[c] = part[u][j][c].x + part[u][j][c].y;
          float s;
          if constexpr (L == 1) {
            s = (p[0] + p[1]) + (p[2] + p[3]);
          } else if constexpr (L == 2) { // classes 2 c + g % 2: pairs with the partner lane group
            s = (p[0] + __shfl_xor(p[0], 16)) + (p[1] + __shfl_xor(p[1], 16));
          } else { // class g
            s = p[0];
            s += __shfl_xor(s, 16);
            s += __shfl_xor(s, 32);
          }
          out[u][j] = s * kv_scale[j];
          // computed here, by every lane: otherwise the compiler sinks the MFMAs into the stores'
          // branches, where the masked lanes' operands are stale
          asm volatile("" : "+v"(out[u][j]));
        }
      // four adjacent logits per lane and token
#pragma unroll
      for (int u = 0; u < U; ++u) {
        if (!stores[u])
          continue;
        OutT *const row = logits + (int64_t)tok_of_u[u] * p.ld_logits - col_shift[u];
#pragma unroll
        for (int j = 0; j < J; ++j) {
          const int n = n0 + 4 * r16 + j;
          if (n >= col_lo[u] && n < col_hi[u]) {
            if constexpr (std::is_same<OutT, float>::value)
              row[n] = out[u][j];
            else
              row[n] = f32_to_bf16(out[u][j]);
          }
        }
      }
      kv_cur = kv_next;
    }
  }
#endif
}
