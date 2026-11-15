// AMD GPU mHC: independent producer / mix / shifted-normalization workers.
// Inspired by the native persistent mHC dataflow, not by sequentially joining
// kernel bodies. This implementation uses resident CTA roles; it must be
// cooperatively launched with a host-verified occupancy bound.
#pragma once
#include <deep_gemm/impls/cdna4/selected_mhc_core.cuh>

namespace deep_gemm::cdna4::mhc_roles {

__device__ __forceinline__ void wait(int *flag, int value) {
    const int lane = threadIdx.x % 64;
    if (lane == 0) {
        while (__hip_atomic_load(flag, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) != value)
            __builtin_amdgcn_s_sleep(1);
    }
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
}

__device__ __forceinline__ uint32_t packed_amax_pair(const v4u (&out)[2], bool second) {
    uint32_t amax[2] = {};
#pragma unroll
    for (int t = 0; t < 2; ++t) {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            uint32_t lo = out[t][q] & 0x7fffu, hi = (out[t][q] >> 16) & 0x7fffu;
            // Match fmax(0, abs(NaN)): NaN does not contribute to the maximum.
            lo = lo > 0x7f80u ? 0u : lo;
            hi = hi > 0x7f80u ? 0u : hi;
            amax[t] = max(amax[t], max(lo, hi));
        }
    }
    uint32_t both = amax[0] | ((second ? amax[1] : 0u) << 16);
#if defined(__HIP_DEVICE_COMPILE__)
    const auto other1 = __shfl_xor(both, 1);
    asm("v_pk_max_u16 %0, %1, %2" : "=v"(both) : "v"(both), "v"(other1));
    const auto other2 = __shfl_xor(both, 2);
    asm("v_pk_max_u16 %0, %1, %2" : "=v"(both) : "v"(both), "v"(other2));
#endif
    return both;
}

template <int H, int PARTITIONS = 2, bool EXACT_SCALE_INVERSE = false, bool PARALLEL_STATS = false,
          bool PACKED_AMAX = false, bool LOCAL_U = false, bool LOCAL_U_SWIZZLE = false, bool COHERENT_STATS = false,
          bool COHERENT_U = false, bool PRECOMPUTED_RMS = false>
__device__ __forceinline__ void norm_pair(const MegaMhcArgs &p, int pair, int partition, const float *stats,
                                          const uint16_t *local_u = nullptr, int first_token = 0) {
    const int lane = threadIdx.x % 64, token = pair * 2;
    const bool second = token + 1 < p.num_tokens;
    constexpr int PACKS = H / 512;
    const int begin = partition * PACKS / PARTITIONS, end = (partition + 1) * PACKS / PARTITIONS;
    float sum = 0.f;
    const auto read_stat = [&](size_t index) {
        if constexpr (COHERENT_STATS)
            return __hip_atomic_load(stats + index, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        else
            return stats[index];
    };
    if constexpr (PRECOMPUTED_RMS) {
        if (lane < 2 && token + lane < p.num_tokens)
            sum = read_stat((token / 64) * p.splits * 64 + (token + lane) % 64);
    } else if constexpr (PARALLEL_STATS) {
        // Adjacent lanes own the two rows of a split. Issue independent split
        // loads together instead of serializing all of them in lanes 0 and 1.
        const int row = lane % 2;
        if (token + row < p.num_tokens)
            for (int s = lane / 2; s < p.splits; s += 32)
                sum += read_stat(((token / 64) * p.splits + s) * 64 + (token + row) % 64);
#pragma unroll
        for (int offset = 2; offset < 64; offset *= 2)
            sum += __shfl_xor(sum, offset);
    } else {
        if (lane < 2 && token + lane < p.num_tokens) {
            for (int s = 0; s < p.splits; ++s)
                sum += read_stat(((token / 64) * p.splits + s) * 64 + (token + lane) % 64);
        }
    }
    const float inv0 = PRECOMPUTED_RMS ? __shfl(sum, 0) : rsqrtf(__shfl(sum, 0) / H + p.rmsnorm_eps);
    const float inv1 = PRECOMPUTED_RMS ? __shfl(sum, 1) : rsqrtf(__shfl(sum, 1) / H + p.rmsnorm_eps);
    struct Pack {
        v4u a, b, w;
    };
    auto load = [&](int pack) {
        const int h = pack * 512 + lane * 8;
        Pack v;
        if constexpr (LOCAL_U) {
            const int width = (end - begin) * 512, local_h = h - begin * 512;
            const int row_a = token - first_token, row_b = (second ? token + 1 : token) - first_token;
            const int h_a = LOCAL_U_SWIZZLE ? ((local_h / 8) ^ (row_a & 15)) * 8 : local_h;
            const int h_b = LOCAL_U_SWIZZLE ? ((local_h / 8) ^ (row_b & 15)) * 8 : local_h;
            v.a = *reinterpret_cast<const v4u *>(local_u + row_a * width + h_a);
            v.b = *reinterpret_cast<const v4u *>(local_u + row_b * width + h_b);
        } else if constexpr (COHERENT_U) {
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                v.a[q] = __hip_atomic_load(reinterpret_cast<const uint32_t *>(p.norm_bf16 + (size_t)token * H + h) + q,
                                           __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                v.b[q] = __hip_atomic_load(
                    reinterpret_cast<const uint32_t *>(p.norm_bf16 + (size_t)(second ? token + 1 : token) * H + h) + q,
                    __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            }
        } else {
            v.a = *reinterpret_cast<const v4u *>(p.norm_bf16 + (size_t)token * H + h);
            v.b = *reinterpret_cast<const v4u *>(p.norm_bf16 + (size_t)(second ? token + 1 : token) * H + h);
        }
        v.w = *reinterpret_cast<const v4u *>(p.rmsnorm_weight + h);
        return v;
    };
    Pack cur = load(begin), next;
    if (begin + 1 < end)
        next = load(begin + 1);
    for (int pack = begin; pack < end; ++pack) {
        Pack prefetch;
        if (pack + 2 < end)
            prefetch = load(pack + 2);
        const int h = pack * 512 + lane * 8;
        v4u out[2];
        float ys[2][8];
        if constexpr (PACKED_AMAX) {
#pragma unroll
            for (int t = 0; t < 2; ++t) {
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    float a, b, wa, wb;
                    unpack_bf16x2(t ? cur.b[q] : cur.a[q], a, b);
                    unpack_bf16x2(cur.w[q], wa, wb);
                    const float inv = t ? inv1 : inv0;
                    out[t][q] = pack_bf16x2(a * inv * (wa * p.rmsnorm_scale), b * inv * (wb * p.rmsnorm_scale));
                    unpack_bf16x2(out[t][q], ys[t][2 * q], ys[t][2 * q + 1]);
                }
            }
        }
        uint32_t both_amax = 0;
        if constexpr (PACKED_AMAX)
            both_amax = packed_amax_pair(out, second);
#pragma unroll
        for (int t = 0; t < 2; ++t) {
            if constexpr (!PACKED_AMAX) {
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    float a, b, wa, wb;
                    unpack_bf16x2(t ? cur.b[q] : cur.a[q], a, b);
                    unpack_bf16x2(cur.w[q], wa, wb);
                    const float inv = t ? inv1 : inv0;
                    out[t][q] = pack_bf16x2(a * inv * (wa * p.rmsnorm_scale), b * inv * (wb * p.rmsnorm_scale));
                    unpack_bf16x2(out[t][q], ys[t][2 * q], ys[t][2 * q + 1]);
                }
            }
            if (t && !second)
                continue;
            if (p.write_y_bf16)
                *reinterpret_cast<v4u *>(p.norm_bf16 + (size_t)(token + t) * H + h) = out[t];
            if (p.y_fp8) {
                float am = 0.f;
                if constexpr (PACKED_AMAX)
                    am = __uint_as_float(((both_amax >> (t * 16)) & 0xffffu) << 16);
                else {
#pragma unroll
                    for (int e = 0; e < 8; ++e)
                        am = fmaxf(am, fabsf(ys[t][e]));
                    am = fmaxf(am, __shfl_xor(am, 1));
                    am = fmaxf(am, __shfl_xor(am, 2));
                }
                const uint32_t bits = __float_as_uint(fmaxf(am, 1e-4f) / 448.f);
                const int exponent = min(max(int((bits >> 23) & 255) + int((bits & 0x7fffff) != 0), 1), 254);
                const float si = EXACT_SCALE_INVERSE ? mega_mhc::inverse_scale(exponent)
                                                     : 1.f / __uint_as_float(uint32_t(exponent) << 23);
                v2u data;
#pragma unroll
                for (int q = 0; q < 2; ++q) {
                    int code = __builtin_amdgcn_cvt_pk_fp8_f32(ys[t][4 * q] * si, ys[t][4 * q + 1] * si, 0, false);
                    data[q] = __builtin_amdgcn_cvt_pk_fp8_f32(ys[t][4 * q + 2] * si, ys[t][4 * q + 3] * si, code, true);
                }
                *reinterpret_cast<v2u *>(p.y_fp8 + (size_t)(token + t) * H + h) = data;
                uint32_t word = exponent;
                word |= uint32_t(__shfl(exponent, (lane & ~15) + 4)) << 8;
                word |= uint32_t(__shfl(exponent, (lane & ~15) + 8)) << 16;
                word |= uint32_t(__shfl(exponent, (lane & ~15) + 12)) << 24;
                if (lane % 16 == 0) {
                    const int col = h / 128;
                    p.y_sf[(token + t) * p.y_sf_stride_token + col * p.y_sf_stride_word] = word;
                    if (p.y_shared_sf)
                        p.y_shared_sf[token + t + col * p.y_shared_sf_stride_word] = word;
                }
            }
        }
        if (pack + 1 < end)
            cur = next;
        if (pack + 2 < end)
            next = prefetch;
    }
}

template <int H, bool PACKED_FN, int PARTITIONS = 2>
__device__ __forceinline__ void run(MegaMhcArgs p, int *counters, const uint16_t *fn, int producers) {
#if defined(__HIP_DEVICE_COMPILE__)
    constexpr int STRIDE = 32;
    const int groups = (p.num_tokens + 63) / 64, tid = threadIdx.x, lane = tid % 64, wave = tid / 64;
    auto *norm_ready = counters;
    auto *mix_ready = counters + groups * STRIDE;
    auto *norm_done = counters + groups * STRIDE * 2;
    auto *mix_done = counters + groups * STRIDE * 3;
    auto *u_stats = p.sqr_partials + groups * p.splits * 64;
    if (blockIdx.x < producers) {
        for (int task = blockIdx.x; task < groups * p.splits; task += producers) {
            const int group = task / p.splits, split = task % p.splits;
            mega_mhc_residual_impl<true, PACKED_FN, true>(p, group, split, fn, u_stats);
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
            __syncthreads();
            if (tid == 0) {
                __hip_atomic_fetch_add(norm_ready + group * STRIDE, 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
                __hip_atomic_fetch_add(mix_ready + group * STRIDE, 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
            }
            __syncthreads();
        }
    } else {
        const int consumer = blockIdx.x - producers, count = gridDim.x - producers;
        if (wave == 0) {
            for (int item = consumer; item < groups * 4; item += count) {
                const int group = item / 4, part = item % 4;
                wait(mix_ready + group * STRIDE, p.splits);
                mega_mhc_mix_impl<1>(p, group * 64 + part * 16);
                if (lane == 0 && __hip_atomic_fetch_add(mix_done + group * STRIDE, 1, __ATOMIC_RELAXED,
                                                        __HIP_MEMORY_SCOPE_AGENT) == 3) {
                    __hip_atomic_store(mix_ready + group * STRIDE, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                    __hip_atomic_store(mix_done + group * STRIDE, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                }
            }
        } else {
            const int jobs = ((p.num_tokens + 1) / 2) * PARTITIONS;
            for (int task = consumer * 3 + wave - 1; task < jobs; task += count * 3) {
                const int pair = task / PARTITIONS, part = task % PARTITIONS, group = pair / 32;
                wait(norm_ready + group * STRIDE, p.splits);
                norm_pair<H, PARTITIONS>(p, pair, part, u_stats);
                const int group_rows = min(64, p.num_tokens - group * 64);
                const int group_tasks = ((group_rows + 1) / 2) * PARTITIONS;
                if (lane == 0 && __hip_atomic_fetch_add(norm_done + group * STRIDE, 1, __ATOMIC_RELAXED,
                                                        __HIP_MEMORY_SCOPE_AGENT) == group_tasks - 1) {
                    __hip_atomic_store(norm_ready + group * STRIDE, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                    __hip_atomic_store(norm_done + group * STRIDE, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                }
            }
        }
    }
#endif
}

} // namespace deep_gemm::cdna4::mhc_roles
