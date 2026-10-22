// CDNA4 MoE gate routing: the top-k experts of every token from its gate logits (x @ weight^T, a
// BF16 GEMM), with the semantics of the CUDA mega gate: scores sqrt(softplus(logit)), ranking by
// score + bias (image tokens: the image bias), weights = the selected scores normalized over the
// top-k and scaled, shared experts after the routed ones with weight 1, every slot's expert mapped
// to a physical replica ((rank + 23333 token) % replicas), and masked, fixed and random routings.
//
// One wave per token: lane l holds experts EV l .. EV l + EV - 1 and keeps its best two unconsumed
// ones. Each of the k rounds takes the wave maximum of the lanes' best (DPP), the lowest lane
// holding it wins (the smallest expert among ties), and the winner promotes its second best,
// re-ranking its experts only when it wins again. Without a bias the ranking is on the raw logits
// (sqrt-softplus is monotonic) and only the selected experts get the activation.
#pragma once
#include <deep_gemm/impls/cdna4/device_common.h>
#include <deep_gemm/impls/cdna4/mega_gate_args.h>

using deep_gemm::cdna4::MegaGateArgs;

// softplus(x) = max(x, 0) + log1p(exp(-|x|)), log1p(t) = log(1 + t) t / ((1 + t) - 1)
__device__ __forceinline__ float sqrt_softplus(float x) {
  const float t = __expf(-fabsf(x)), u = 1.f + t;
  const float log1p_t = u == 1.f ? t : __fdividef(__logf(u) * t, u - 1.f);
  return __builtin_amdgcn_sqrtf(fmaxf(x, 0.f) + log1p_t);
}

// Philox4x32-10 with the counter layout of curand's Philox state: key = seed, counter words 2-3 =
// subsequence; outputs are consumed four per counter
struct Philox4x32 {
  uint32_t ctr[4], key[2], out[4];
  int used = 4;
  __device__ Philox4x32(uint64_t seed, uint64_t subsequence) {
    key[0] = (uint32_t)seed, key[1] = (uint32_t)(seed >> 32);
    ctr[0] = ctr[1] = 0, ctr[2] = (uint32_t)subsequence, ctr[3] = (uint32_t)(subsequence >> 32);
  }
  __device__ uint32_t next() {
    if (used == 4) {
      uint32_t c[4] = {ctr[0], ctr[1], ctr[2], ctr[3]}, k0 = key[0], k1 = key[1];
#pragma unroll
      for (int r = 0; r < 10; ++r) {
        const uint32_t lo0 = 0xD2511F53u * c[0], hi0 = __umulhi(0xD2511F53u, c[0]);
        const uint32_t lo1 = 0xCD9E8D57u * c[2], hi1 = __umulhi(0xCD9E8D57u, c[2]);
        c[0] = hi1 ^ c[1] ^ k0, c[1] = lo1, c[2] = hi0 ^ c[3] ^ k1, c[3] = lo0;
        k0 += 0x9E3779B9u, k1 += 0xBB67AE85u;
      }
#pragma unroll
      for (int i = 0; i < 4; ++i)
        out[i] = c[i];
      if (++ctr[0] == 0 && ++ctr[1] == 0 && ++ctr[2] == 0)
        ++ctr[3];
      used = 0;
    }
    return out[used++];
  }
  // (0, 1], as curand_uniform
  __device__ float uniform() { return next() * 2.3283064e-10f + 2.3283064e-10f / 2.f; }
};

__device__ __forceinline__ uint32_t wave_min_u32(uint32_t v) {
#pragma unroll
  for (int offset = 1; offset < 64; offset *= 2)
    v = min(v, (uint32_t)__shfl_xor((int)v, offset));
  return v;
}

// the best two values of the unconsumed slots, the earlier slot first among ties (slot -1: none)
template <int EV>
__device__ __forceinline__ void best_two(const float (&v)[EV], uint32_t consumed, float &b1, int &i1, float &b2,
                                         int &i2) {
  b1 = b2 = -__builtin_inff(), i1 = i2 = -1;
#pragma unroll
  for (int j = 0; j < EV; ++j) {
    const bool open = !(consumed >> j & 1);
    const bool first = open && (v[j] > b1 || i1 < 0), second = open && (v[j] > b2 || i2 < 0);
    b2 = first ? b1 : second ? v[j] : b2, i2 = first ? i1 : second ? j : i2;
    b1 = first ? v[j] : b1, i1 = first ? j : i1;
  }
}

// EV experts per lane (64 EV >= routed experts); BIASED: ranking by score + bias, else by logit
template <int EV, bool BIASED> __device__ __forceinline__ void mega_gate_impl(const MegaGateArgs &p) {
  const int lane = threadIdx.x % 64, waves = blockDim.x / 64;
  const int k = p.num_topk, slots = p.num_topk + p.num_shared_experts, num_experts = p.num_routed_experts;
  constexpr uint32_t kInvalid = 0xffffffffu;
  constexpr int VEC = EV % 4 == 0 ? 4 : EV % 2 == 0 ? 2 : 1;
  const bool lane_valid = lane * EV < num_experts;
  float bias[EV], image_bias[EV];
#pragma unroll
  for (int v = 0; v < EV; ++v) {
    const int e = lane * EV + v;
    bias[v] = BIASED && p.bias && e < num_experts ? p.bias[e] : 0.f;
    image_bias[v] = BIASED && p.image_bias && e < num_experts ? p.image_bias[e] : 0.f;
  }

  for (int token = blockIdx.x * waves + threadIdx.x / 64; token < p.num_tokens; token += gridDim.x * waves) {
    const float *row = p.logits + (int64_t)token * p.ld_logits;
    float value[EV];
    if (lane_valid) {
#pragma unroll
      for (int v = 0; v < EV; v += VEC) {
        if constexpr (VEC == 4) {
          const float4 x = *reinterpret_cast<const float4 *>(row + lane * EV + v);
          value[v] = x.x, value[v + 1] = x.y, value[v + 2] = x.z, value[v + 3] = x.w;
        } else if constexpr (VEC == 2) {
          const float2 x = *reinterpret_cast<const float2 *>(row + lane * EV + v);
          value[v] = x.x, value[v + 1] = x.y;
        } else {
          value[v] = row[lane * EV + v];
        }
      }
    }
    auto store = [&](int64_t physical, float weight, int64_t logical) {
      if (lane < slots) {
        p.topk_idx[(int64_t)token * slots + lane] = physical;
        p.topk_weights[(int64_t)token * slots + lane] = weight;
      }
      if (p.unmapped_topk_idx && lane < k)
        p.unmapped_topk_idx[token * p.unmapped_stride + lane] = logical;
    };
    if (p.mask && !p.mask[token]) {
      store(-1, 0.f, -1);
      continue;
    }
    if (p.force_random && p.force_random[token]) {
      // distinct experts out of all physical ones: lane i draws from n - i, then ranks at or above
      // every smaller earlier draw shift up by one
      uint32_t n = num_experts + p.num_shared_experts;
      if (p.to_physical_map) {
        uint32_t count = 0;
        for (int e = lane; e < num_experts + p.num_shared_experts; e += 64)
          count += (uint32_t)p.logical_count[e];
        n = (uint32_t)wave_reduce<false>((float)count);
      }
      Philox4x32 rng(p.rank_idx, (uint64_t)((uint32_t)token * 32u + (uint32_t)lane));
      const float weight = rng.uniform();
      uint32_t candidate = lane < slots ? rng.next() % (n - lane) : kInvalid, selected = kInvalid;
      for (int o = 0; o < slots; ++o) {
        const uint32_t min_candidate = wave_min_u32(candidate);
        const uint32_t min_lane = wave_min_u32(candidate == min_candidate ? (uint32_t)lane : kInvalid);
        if (candidate != kInvalid) {
          if (min_lane == (uint32_t)lane)
            selected = candidate, candidate = kInvalid;
          else if (candidate >= min_candidate && min_lane < (uint32_t)lane)
            ++candidate;
        }
      }
      store((int64_t)selected, weight, -1);
      continue;
    }

    int selected = -1;
    if (p.fix_routing_mask && p.fix_routing_mask[token]) {
      if (lane < k)
        selected = (int)p.unmapped_topk_idx[token * p.unmapped_stride + lane];
    } else {
      const bool image = BIASED && p.image_token_mask && p.image_token_mask[token];
      // padded slots count as consumed
      uint32_t consumed = 0;
#pragma unroll
      for (int v = 0; v < EV; ++v) {
        if constexpr (BIASED) {
          // opaque values keep the select per element (a select of the arrays moves them to scratch)
          float routed = bias[v], other = image_bias[v];
          asm volatile("" : "+v"(routed), "+v"(other));
          value[v] = sqrt_softplus(value[v]) + (image ? other : routed);
        }
        consumed |= lane * EV + v < num_experts ? 0u : 1u << v;
      }
      float b1, b2;
      int i1, i2;
      best_two(value, consumed, b1, i1, b2, i2);
      for (int round = 0; round < k; ++round) {
        const float best = wave_reduce<true>(i1 >= 0 ? b1 : -__builtin_inff());
        const int winner = __builtin_ctzll(__ballot(i1 >= 0 && b1 == best));
        const int e = __builtin_amdgcn_readlane(lane * EV + i1, winner);
        selected = lane == round ? e : selected;
        if (lane == winner) {
          consumed |= 1u << i1;
          if (i2 >= 0)
            b1 = b2, i1 = i2, b2 = -__builtin_inff(), i2 = -1;
          else
            best_two(value, consumed, b1, i1, b2, i2);
        }
      }
    }
    // the selected experts' scores from their logits (the rows are still cached)
    const float score = lane < k ? sqrt_softplus(row[max(selected, 0)]) : 0.f;
    // shared experts follow the routed ones
    if (lane >= k && lane < slots)
      selected = num_experts + lane - k;
    int64_t physical = selected;
    if (p.to_physical_map && lane < slots) {
      const uint32_t replicas = (uint32_t)p.logical_count[selected];
      const uint32_t replica = (p.rank_idx + (uint32_t)token * 23333u) % replicas;
      physical = p.to_physical_map[(int64_t)selected * p.num_duplicate_experts + replica];
    }
    const float sum = wave_reduce<false>(score) + 1e-20f;
    store(physical, lane < k ? score / sum * p.routed_scaling_factor : 1.f, selected);
  }
}
