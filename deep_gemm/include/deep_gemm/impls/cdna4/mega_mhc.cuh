// CDNA4 Mega mHC: the hyper-connection residual update, the next layer's mixes and the RMSNorm
// (with an optional FP8 cast) of the CUDA mega_mhc, in three kernels.
//
// The residual kernel streams x and the residual of 64 tokens over a hidden range (a split) and
// writes new_residual = x post + comb^T residual in BF16. The rounded values are the A operand of
// the mix GEMM straight from the registers: a step of 64 hidden values takes two 16x16x32 BF16
// MFMAs per route (K = route hidden + h) against fn split into BF16 hi and lo (16 mantissa bits;
// CDNA4 has no TF32), and the split's partial mixes and sums of squares go to a workspace. Lane
// group g holds hidden 16 g .. + 15 of a step (MFMA u takes its values 8 u .. + 7), so the four
// groups read and write whole 128-byte lines of a row. fn is staged per step in LDS as MFMA
// operands, shared by the four waves (16 tokens each). Shifted mHC also writes the norm input
// (new_residual weighted by the given previous mix).
//
// The mix kernel takes a token per quad of lanes: the split partials in a fixed order, then the
// sigmoids and the Sinkhorn normalization. The norm kernel takes a token per workgroup and
// normalizes the norm input (normal mHC: new_residual weighted by the new previous mix) into BF16
// and FP8.
#pragma once
#include "device_common.h"

using deep_gemm::cdna4::MegaMhcArgs;

namespace mega_mhc {
constexpr int ROUTES = 4, OUTPUTS = 24, BM = 64, STEP = 64;
// a step's fn as MFMA operands: [route][MFMA u][fragment][hi, lo][lane] x 16 bytes
constexpr int STAGE_BYTES = ROUTES * 2 * 2 * 2 * 64 * 16;
constexpr int LDS_BYTES = 2 * STAGE_BYTES;
} // namespace mega_mhc

template <bool SHIFTED> __device__ __forceinline__ void mega_mhc_residual_impl(const MegaMhcArgs &p) {
#if defined(__HIP_DEVICE_COMPILE__)
  using namespace mega_mhc;
  extern __shared__ __attribute__((aligned(16))) uint8_t smem[];
  const int tid = threadIdx.x, lane = tid % 64, g = lane / 16, r16 = lane % 16;
  const int wave = __builtin_amdgcn_readfirstlane(tid / 64);
  const int H = p.hidden, m_block = blockIdx.x, split = blockIdx.y;
  const int m0 = m_block * BM, rows = min(BM, p.num_tokens - m0);
  const int row = 16 * wave + r16, token = min(m0 + row, p.num_tokens - 1);

  const int steps = H / STEP, split_steps = (steps + p.splits - 1) / p.splits;
  const int s_begin = min(steps, split * split_steps), s_end = min(steps, s_begin + split_steps);

  // rows past the tokens read zeros and drop their stores
  const auto x_rsrc = make_rsrc(p.x + (size_t)m0 * H, (uint32_t)(rows * H * 2));
  const auto res_rsrc = make_rsrc(p.residual + (size_t)m0 * ROUTES * H, (uint32_t)(rows * ROUTES * H * 2));
  const auto out_rsrc = make_rsrc(p.new_residual + (size_t)m0 * ROUTES * H, (uint32_t)(rows * ROUTES * H * 2));
  const auto norm_rsrc = make_rsrc(SHIFTED ? p.norm_bf16 + (size_t)m0 * H : p.x, SHIFTED ? (uint32_t)(rows * H * 2) : 0);

  float post[ROUTES], comb[ROUTES][ROUTES], prev[ROUTES];
#pragma unroll
  for (int j = 0; j < ROUTES; ++j) {
    post[j] = p.post_mix[token * ROUTES + j];
    prev[j] = SHIFTED ? p.shifted_prev_mix[token * ROUTES + j] : 0.f;
#pragma unroll
    for (int i = 0; i < ROUTES; ++i)
      comb[i][j] = p.comb_res_mix[(token * ROUTES + i) * ROUTES + j];
  }

  // Loads past the split repeat its last step (unconditional loads keep the prefetch in flight)
  // fn of a step: thread t loads 4 floats of output o, route j (96 rows of 64 values, 6 rounds)
  constexpr int FN_ROUNDS = 6;
  auto load_fn = [&](int s, v4f (&f)[FN_ROUNDS]) {
    s = min(s, s_end - 1);
#pragma unroll
    for (int rd = 0; rd < FN_ROUNDS; ++rd) {
      const int idx = rd * 16 + tid / 16, o = idx / ROUTES, j = idx % ROUTES;
      f[rd] = *(const v4f *)(p.fn + (size_t)o * ROUTES * H + j * H + s * STEP + 4 * (tid % 16));
    }
  };
  // ... into the operand slots: output o is row o % 16 of fragment o / 16; hidden hh of the step
  // is value 8 u + e (u = hh / 8 % 2, e = hh % 8) of lane group hh / 16
  auto store_fn = [&](int stage, const v4f (&f)[FN_ROUNDS]) {
#pragma unroll
    for (int rd = 0; rd < FN_ROUNDS; ++rd) {
      const int idx = rd * 16 + tid / 16, o = idx / ROUTES, j = idx % ROUTES, hh = 4 * (tid % 16);
      const int u = hh / 8 % 2, slot_lane = 16 * (hh / 16) + o % 16;
      uint8_t *slot = smem + stage * STAGE_BYTES + (((j * 2 + u) * 2 + o / 16) * 2 * 64 + slot_lane) * 16 + (hh % 8) * 2;
      v2u hi, lo;
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const uint32_t h = pack_bf16x2(f[rd][2 * e], f[rd][2 * e + 1]);
        float h0, h1;
        unpack_bf16x2(h, h0, h1);
        hi[e] = h, lo[e] = pack_bf16x2(f[rd][2 * e] - h0, f[rd][2 * e + 1] - h1);
      }
      *(v2u *)slot = hi;
      *(v2u *)(slot + 64 * 16) = lo;
    }
  };
  struct Step {
    v4i x[2], res[ROUTES][2]; // [u]: 8 BF16 of the lane's token from hidden 64 s + 16 g + 8 u
  };
  auto load_step = [&](int s, Step &t) {
    s = min(s, s_end - 1);
#pragma unroll
    for (int u = 0; u < 2; ++u) {
      const uint32_t h = s * STEP + 16 * g + 8 * u;
      t.x[u] = (v4i)__builtin_amdgcn_raw_buffer_load_b128(x_rsrc, (row * H + h) * 2, 0, 0);
#pragma unroll
      for (int i = 0; i < ROUTES; ++i)
        t.res[i][u] = (v4i)__builtin_amdgcn_raw_buffer_load_b128(res_rsrc, ((row * ROUTES + i) * H + h) * 2, 0, 0);
    }
  };

  v4f acc[2] = {v4f{}, v4f{}};
  float sq = 0.f;
  // step s from cur and LDS stage `stage`
  auto compute = [&](int s, const Step &cur, int stage) {
    const uint8_t *operands = smem + stage * STAGE_BYTES;
#pragma unroll
    for (int u = 0; u < 2; ++u) {
      const uint32_t h = s * STEP + 16 * g + 8 * u;
      float xv[8], rv[ROUTES][8];
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        unpack_bf16x2((uint32_t)cur.x[u][q], xv[2 * q], xv[2 * q + 1]);
#pragma unroll
        for (int i = 0; i < ROUTES; ++i)
          unpack_bf16x2((uint32_t)cur.res[i][u][q], rv[i][2 * q], rv[i][2 * q + 1]);
      }
      v4i nr[ROUTES];
      float rounded[ROUTES][8];
#pragma unroll
      for (int j = 0; j < ROUTES; ++j) {
        float v[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          v[e] = xv[e] * post[j];
#pragma unroll
          for (int i = 0; i < ROUTES; ++i)
            v[e] = __builtin_fmaf(comb[i][j], rv[i][e], v[e]);
        }
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          nr[j][q] = (int)pack_bf16x2(v[2 * q], v[2 * q + 1]);
          unpack_bf16x2((uint32_t)nr[j][q], rounded[j][2 * q], rounded[j][2 * q + 1]);
        }
        __builtin_amdgcn_raw_buffer_store_b128((v4u)nr[j], out_rsrc, ((row * ROUTES + j) * H + h) * 2, 0, 0);
#pragma unroll
        for (int e = 0; e < 8; ++e)
          sq = __builtin_fmaf(rounded[j][e], rounded[j][e], sq);
      }
      if constexpr (SHIFTED) {
        v4i n;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          float a[2];
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            a[e] = rounded[0][2 * q + e] * prev[0];
#pragma unroll
            for (int j = 1; j < ROUTES; ++j)
              a[e] = __builtin_fmaf(rounded[j][2 * q + e], prev[j], a[e]);
          }
          n[q] = (int)pack_bf16x2(a[0], a[1]);
        }
        __builtin_amdgcn_raw_buffer_store_b128((v4u)n, norm_rsrc, (row * H + h) * 2, 0, 0);
      }
#pragma unroll
      for (int j = 0; j < ROUTES; ++j)
#pragma unroll
        for (int f = 0; f < 2; ++f) {
          const uint8_t *slot = operands + (((j * 2 + u) * 2 + f) * 2 * 64 + lane) * 16;
          acc[f] = mfma_bf16_16x16x32(*(const v4i *)slot, nr[j], acc[f]);
          acc[f] = mfma_bf16_16x16x32(*(const v4i *)(slot + 64 * 16), nr[j], acc[f]);
        }
    }
  };

  // output rows 24 .. 31 (fragment 1) stay zero
  for (int i = tid; i < LDS_BYTES / 16; i += 256)
    ((v4i *)smem)[i] = v4i{};
  __syncthreads();
  Step even, odd;
  v4f fn_even[FN_ROUNDS], fn_odd[FN_ROUNDS];
  load_fn(s_begin, fn_even);
  store_fn(0, fn_even);
  load_step(s_begin, even);
  __syncthreads();
  // two steps per iteration on alternating registers (no copies between the prefetch and its use);
  // the barriers order LDS only, so the prefetches and stores stay in flight
  for (int s = s_begin; s < s_end; s += 2) {
    load_fn(s + 1, fn_odd);
    load_step(s + 1, odd);
    compute(s, even, 0);
    if (s + 1 < s_end)
      store_fn(1, fn_odd);
    asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
    if (s + 1 >= s_end)
      break;
    load_fn(s + 2, fn_even);
    load_step(s + 2, even);
    compute(s + 1, odd, 1);
    if (s + 2 < s_end)
      store_fn(0, fn_even);
    asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
  }

  // acc[f][e] of lane (g, r16) is the mix 16 f + 4 g + e of the wave's token r16
  const size_t task = (size_t)m_block * p.splits + split;
  float *mixes = p.mix_partials + (task * BM + row) * OUTPUTS;
  *(v4f *)(mixes + 4 * g) = acc[0];
  if (g < 2)
    *(v4f *)(mixes + 16 + 4 * g) = acc[1];
  sq += __shfl_xor(sq, 16);
  sq += __shfl_xor(sq, 32);
  if (g == 0)
    p.sqr_partials[task * BM + row] = sq;
#endif
}

// Mega MoE shared-expert scale rows: blocks of shared_sf_block_m tokens on 128-row boundaries,
// 128 tokens of a block interleaved 32 x 4
__device__ __forceinline__ int64_t shared_sf_row(int token, int block_m) {
  const int i = token % block_m;
  return (int64_t)(token / block_m) * ((block_m + 127) / 128 * 128) + i / 128 * 128 + i % 32 * 4 + i % 128 / 32;
}

// The mixes of a token per quad of lanes: lane k adds the split partials k, k + 4, ... in split
// order and the quad adds its lanes in a fixed butterfly (an order set by the split count alone),
// then the RMS scaling of new_residual, the sigmoids (lane k: previous and post mix k) and the
// Sinkhorn normalization of the 4 x 4 combination (lane k: row k, columns over the quad)
__device__ __forceinline__ void mega_mhc_mix_impl(const MegaMhcArgs &p) {
  using namespace mega_mhc;
  const int thread = blockIdx.x * blockDim.x + threadIdx.x, k = thread % ROUTES;
  const int token = min(thread / ROUTES, p.num_tokens - 1);
  const size_t task0 = (size_t)(token / BM) * p.splits;
  const int r = token % BM;
  const auto quad_sum = [](float v) {
    v += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0xb1, 0xf, 0xf, false)); // quad_perm 1 0 3 2
    return v + __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x4e, 0xf, 0xf, false)); // 2 3 0 1
  };
  // the loads of four of the lane's splits ahead of their sums
  constexpr int BATCH = 4;
  float mix[OUTPUTS] = {}, sqr = 0.f;
  for (int s0 = k; s0 < p.splits; s0 += BATCH * ROUTES) {
    v4f v[BATCH][OUTPUTS / 4];
    float sq[BATCH];
#pragma unroll
    for (int b = 0; b < BATCH; ++b) {
      const int s = min(s0 + b * ROUTES, p.splits - 1);
      const v4f *partial = (const v4f *)(p.mix_partials + ((task0 + s) * BM + r) * OUTPUTS);
#pragma unroll
      for (int q = 0; q < OUTPUTS / 4; ++q)
        v[b][q] = partial[q];
      sq[b] = p.sqr_partials[(task0 + s) * BM + r];
    }
#pragma unroll
    for (int b = 0; b < BATCH; ++b) {
      if (s0 + b * ROUTES >= p.splits)
        break;
#pragma unroll
      for (int q = 0; q < OUTPUTS / 4; ++q)
#pragma unroll
        for (int e = 0; e < 4; ++e)
          mix[4 * q + e] += v[b][q][e];
      sqr += sq[b];
    }
  }
#pragma unroll
  for (int o = 0; o < OUTPUTS; ++o)
    mix[o] = quad_sum(mix[o]);
  const float rms = rsqrtf(quad_sum(sqr) / (ROUTES * p.hidden) + p.hc_norm_eps);
  const auto scaled = [&](int o) { return mix[o] * rms * p.mix_scales[o < 4 ? 0 : o < 8 ? 1 : 2] + p.mix_bases[o]; };
  // the lane's mixes: previous and post mix k, and row k of the combination (selects of constant
  // indices keep the mixes in registers)
  float pre = 0.f, post = 0.f, c[ROUTES];
#pragma unroll
  for (int j = 0; j < ROUTES; ++j) {
    pre = k == j ? scaled(j) : pre;
    post = k == j ? scaled(ROUTES + j) : post;
  }
#pragma unroll
  for (int j = 0; j < ROUTES; ++j) {
    c[j] = scaled(8 + j);
#pragma unroll
    for (int i = 1; i < ROUTES; ++i)
      c[j] = k == i ? scaled(8 + 4 * i + j) : c[j];
  }
  const bool valid = thread / ROUTES < p.num_tokens;
  if (valid) {
    p.prev_mix[token * ROUTES + k] = 1.f / (1.f + __expf(-pre)) + p.hc_pre_eps;
    p.new_post_mix[token * ROUTES + k] = 1.f / (1.f + __expf(-post)) * p.hc_post_scale;
  }
  // one division per row or column
  const float row_max = fmaxf(fmaxf(c[0], c[1]), fmaxf(c[2], c[3]));
  float sum = 0.f;
#pragma unroll
  for (int j = 0; j < ROUTES; ++j)
    sum += c[j] = __expf(c[j] - row_max);
  float inv = 1.f / sum;
#pragma unroll
  for (int j = 0; j < ROUTES; ++j)
    c[j] = c[j] * inv + p.sinkhorn_eps;
  const auto normalize_cols = [&] {
#pragma unroll
    for (int j = 0; j < ROUTES; ++j)
      c[j] *= 1.f / (quad_sum(c[j]) + p.sinkhorn_eps);
  };
  normalize_cols();
  for (int it = 1; it < p.num_sinkhorn_iters; ++it) {
    inv = 1.f / (c[0] + c[1] + c[2] + c[3] + p.sinkhorn_eps);
#pragma unroll
    for (int j = 0; j < ROUTES; ++j)
      c[j] *= inv;
    normalize_cols();
  }
  if (valid)
    *(v4f *)(p.new_comb_res_mix + (token * ROUTES + k) * ROUTES) = v4f{c[0], c[1], c[2], c[3]};
}

// CHUNKS = hidden / 1024: thread t owns hidden 4 (t + 256 c) .. + 3 of chunk c
template <bool SHIFTED, int CHUNKS> __device__ __forceinline__ void mega_mhc_norm_impl(const MegaMhcArgs &p) {
  using namespace mega_mhc;
  __shared__ float wave_sums[2][4];
  const int tid = threadIdx.x, lane = tid % 64, wave = tid / 64, H = p.hidden;
  float weight[CHUNKS][4];
#pragma unroll
  for (int c = 0; c < CHUNKS; ++c) {
    const v2u w = *(const v2u *)(p.rmsnorm_weight + 4 * (tid + 256 * c));
    unpack_bf16x2(w[0], weight[c][0], weight[c][1]);
    unpack_bf16x2(w[1], weight[c][2], weight[c][3]);
#pragma unroll
    for (int e = 0; e < 4; ++e)
      weight[c][e] *= p.rmsnorm_scale;
  }

  // the norm input (shifted) or new_residual (normal) of a token, loaded a token ahead
  constexpr int INPUTS = SHIFTED ? 1 : ROUTES;
  auto load_input = [&](int token, v2u (&in)[INPUTS][CHUNKS]) {
#pragma unroll
    for (int j = 0; j < INPUTS; ++j)
#pragma unroll
      for (int c = 0; c < CHUNKS; ++c)
        in[j][c] = token >= p.num_tokens ? v2u{}
                   : SHIFTED ? *(const v2u *)(p.norm_bf16 + (size_t)token * H + 4 * (tid + 256 * c))
                             : *(const v2u *)(p.new_residual + ((size_t)token * ROUTES + j) * H + 4 * (tid + 256 * c));
  };
  v2u in[INPUTS][CHUNKS];
  load_input(blockIdx.x, in);
  int parity = 0;
  for (int token = blockIdx.x; token < p.num_tokens; token += gridDim.x, parity ^= 1) {
    v2u in_next[INPUTS][CHUNKS];
    load_input(token + gridDim.x, in_next);
    // normal mHC: the new previous mix of the mix kernel weights the routes
    v4f prev = SHIFTED ? v4f{} : *(const v4f *)(p.prev_mix + token * ROUTES);

    // the norm input in BF16, then its RMS over the workgroup in wave order
    float n[CHUNKS][4];
#pragma unroll
    for (int c = 0; c < CHUNKS; ++c) {
      if constexpr (SHIFTED) {
        unpack_bf16x2(in[0][c][0], n[c][0], n[c][1]);
        unpack_bf16x2(in[0][c][1], n[c][2], n[c][3]);
      } else {
        float v[ROUTES][4];
#pragma unroll
        for (int j = 0; j < ROUTES; ++j) {
          unpack_bf16x2(in[j][c][0], v[j][0], v[j][1]);
          unpack_bf16x2(in[j][c][1], v[j][2], v[j][3]);
        }
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          float a = v[0][e] * prev[0];
#pragma unroll
          for (int j = 1; j < ROUTES; ++j)
            a = __builtin_fmaf(v[j][e], prev[j], a);
          n[c][e] = a;
        }
#pragma unroll
        for (int e = 0; e < 4; e += 2)
          unpack_bf16x2(pack_bf16x2(n[c][e], n[c][e + 1]), n[c][e], n[c][e + 1]);
      }
    }
    float ss = 0.f;
#pragma unroll
    for (int c = 0; c < CHUNKS; ++c)
#pragma unroll
      for (int e = 0; e < 4; ++e)
        ss = __builtin_fmaf(n[c][e], n[c][e], ss);
    ss = wave_reduce<false>(ss);
    if (lane == 0)
      wave_sums[parity][wave] = ss;
    // LDS only: the next token's loads stay in flight
    asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
    const float rms = rsqrtf((wave_sums[parity][0] + wave_sums[parity][1] + wave_sums[parity][2] +
                              wave_sums[parity][3]) / H + p.rmsnorm_eps);

#pragma unroll
    for (int c = 0; c < CHUNKS; ++c) {
      const int h = 4 * (tid + 256 * c);
      v2u y;
      float yv[4];
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        y[e] = pack_bf16x2(n[c][2 * e] * rms * weight[c][2 * e], n[c][2 * e + 1] * rms * weight[c][2 * e + 1]);
        unpack_bf16x2(y[e], yv[2 * e], yv[2 * e + 1]);
      }
      if (p.write_y_bf16)
        *(v2u *)(p.norm_bf16 + (size_t)token * H + h) = y;
      if (p.y_fp8 != nullptr) {
        // UE8M0 scale of 32 values (8 lanes: two quads, joined by the half-row mirror):
        // ceil(log2(amax / 448)), at least 1e-4 / 448
        float amax = fmaxf(fmaxf(fabsf(yv[0]), fabsf(yv[1])), fmaxf(fabsf(yv[2]), fabsf(yv[3])));
        const auto dpp_max = [&](auto ctrl) {
          amax = fmaxf(amax, __int_as_float(__builtin_amdgcn_update_dpp(__float_as_int(amax), __float_as_int(amax),
                                                                        decltype(ctrl)::value, 0xf, 0xf, false)));
        };
        dpp_max(std::integral_constant<int, 0xb1>{});  // quad_perm 1 0 3 2
        dpp_max(std::integral_constant<int, 0x4e>{});  // quad_perm 2 3 0 1
        dpp_max(std::integral_constant<int, 0x141>{}); // row_half_mirror
        const uint32_t bits = __float_as_uint(fmaxf(amax, 1e-4f) / 448.f);
        const int exponent = min(max((int)(bits >> 23 & 0xff) + ((bits & 0x7fffff) != 0), 1), 254);
        const float inv = 1.f / __uint_as_float((uint32_t)exponent << 23);
        const int lo = __builtin_amdgcn_cvt_pk_fp8_f32(yv[0] * inv, yv[1] * inv, 0, false);
        const int packed = __builtin_amdgcn_cvt_pk_fp8_f32(yv[2] * inv, yv[3] * inv, lo, true);
        *(int *)(p.y_fp8 + (size_t)token * H + h) = packed;
        // the block's leader stores byte (t / 8) % 4 of the word of hidden 128 (t / 32 + 8 c) .. + 127
        if (lane % 8 == 0) {
          const int w = tid / 32 + 8 * c, b = tid / 8 % 4;
          ((uint8_t *)(p.y_sf + token * p.y_sf_stride_token + w * p.y_sf_stride_word))[b] = (uint8_t)exponent;
          if (p.y_shared_sf != nullptr)
            ((uint8_t *)(p.y_shared_sf + shared_sf_row(token, p.shared_sf_block_m) + w * p.y_shared_sf_stride_word))[b] =
                (uint8_t)exponent;
        }
      }
    }
#pragma unroll
    for (int j = 0; j < INPUTS; ++j)
#pragma unroll
      for (int c = 0; c < CHUNKS; ++c)
        in[j][c] = in_next[j][c];
  }
}
