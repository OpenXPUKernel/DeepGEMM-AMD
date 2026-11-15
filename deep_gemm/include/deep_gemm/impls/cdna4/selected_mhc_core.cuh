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
#include <deep_gemm/impls/cdna4/device_common.h>

using deep_gemm::cdna4::MegaMhcArgs;

namespace mega_mhc {
constexpr int ROUTES = 4, OUTPUTS = 24, BM = 64, STEP = 64;
// a step's fn as MFMA operands: [route][MFMA u][fragment][hi, lo][lane] x 16 bytes
constexpr int STAGE_BYTES = ROUTES * 2 * 2 * 2 * 64 * 16;
constexpr int LDS_BYTES = 2 * STAGE_BYTES;

// Native mHC uses approximate reciprocal in the small Mix epilogue. The
// opt-in version below is for the positive-normal denominators produced by
// softmax/Sinkhorn with the model's positive epsilon. It is not an IEEE div
// replacement for arbitrary subnormal input, nor a global fast-math setting.
__device__ __forceinline__ float reciprocal_positive(float x) {
#if defined(__HIP_DEVICE_COMPILE__)
    float r;
    asm("v_rcp_f32 %0, %1" : "=v"(r) : "v"(x));
    return r;
#else
    return 1.f / x;
#endif
}
// Exact reciprocal of the clamped UE8M0 power of two, including 2^-127.
__device__ __forceinline__ float inverse_scale(int exponent) {
    return __uint_as_float(exponent == 254 ? 0x00400000u : uint32_t(254 - exponent) << 23);
}

// Register-bank experiment only: same BF16 operands and FP32 MFMA recurrence.
// The default continues to use the compiler intrinsic and its allocator.
template <bool FORCE_ACC> __device__ __forceinline__ v4f dot(v4i a, v4i b, v4f c) {
#if defined(__HIP_DEVICE_COMPILE__)
    if constexpr (FORCE_ACC) {
        asm volatile("v_mfma_f32_16x16x32_bf16 %0, %1, %2, %0" : "+a"(c) : "v"(a), "v"(b));
        return c;
    }
#endif
    return mfma_bf16_16x16x32(a, b, c);
}
} // namespace mega_mhc

template <bool SHIFTED, int PACKED_FN = 0, bool SAVE_U_SQR = false, int RES_STORE_POLICY = 0, bool PACKED_POST = false,
          bool STREAM_ROUTES = false, bool DMA_FN = false, bool PREFETCH_INPUT = true, bool PAIR_STATS = false,
          int TILE_M = 64, bool FORCE_ACC = false, int IGLP_MODE = -1, bool DISTRIBUTED_COEFF = false,
          bool FN_SWIZZLE = false, bool RETAIN_U = false, bool LOCAL_U_SWIZZLE = false, bool COHERENT_PARTIALS = false,
          bool TRIM_PREFETCH = false, bool TRIM_FINAL_SYNC = false>
__device__ __forceinline__ void mega_mhc_residual_impl(const MegaMhcArgs &p, int m_block = blockIdx.x,
                                                       int split = blockIdx.y, const uint16_t *packed_fn = nullptr,
                                                       float *u_sqr_partials = nullptr, uint16_t *local_u = nullptr) {
#if defined(__HIP_DEVICE_COMPILE__)
    using namespace mega_mhc;
    static_assert(TILE_M == 32 || TILE_M == 64 || TILE_M == 128, "Unsupported mHC producer tile");
    static_assert(!FN_SWIZZLE || PACKED_FN == 2, "Fn swizzle requires matching compact prepacking");
    static_assert(!TRIM_PREFETCH || !DMA_FN, "Tail trimming currently applies to register-prefetched Fn");
    constexpr int BM = TILE_M, THREADS = BM * 4;
    constexpr int STAGE_BYTES = PACKED_FN == 2 ? 24576 : mega_mhc::STAGE_BYTES;
    constexpr int LDS_BYTES = 2 * STAGE_BYTES;
    extern __shared__ __attribute__((aligned(16))) uint8_t smem[];
    const int tid = threadIdx.x, lane = tid % 64, g = lane / 16, r16 = lane % 16;
    const int wave = __builtin_amdgcn_readfirstlane(tid / 64);
    const int H = p.hidden;
    const int m0 = m_block * BM, rows = min(BM, p.num_tokens - m0);
    const int row = 16 * wave + r16, token = min(m0 + row, p.num_tokens - 1);

    const int steps = H / STEP, split_steps = (steps + p.splits - 1) / p.splits;
    const int s_begin = min(steps, split * split_steps), s_end = min(steps, s_begin + split_steps);

    // rows past the tokens read zeros and drop their stores
    const auto x_rsrc = make_rsrc(p.x + (size_t)m0 * H, (uint32_t)(rows * H * 2));
    const auto res_rsrc = make_rsrc(p.residual + (size_t)m0 * ROUTES * H, (uint32_t)(rows * ROUTES * H * 2));
    const auto out_rsrc = make_rsrc(p.new_residual + (size_t)m0 * ROUTES * H, (uint32_t)(rows * ROUTES * H * 2));
    const auto norm_rsrc =
        make_rsrc(SHIFTED ? p.norm_bf16 + (size_t)m0 * H : p.x, SHIFTED ? (uint32_t)(rows * H * 2) : 0);

    float post[ROUTES], comb[ROUTES][ROUTES], prev[ROUTES];
    float lane_post = 0.f, lane_prev = 0.f, lane_comb[ROUTES] = {};
    if constexpr (DISTRIBUTED_COEFF) {
        lane_post = p.post_mix[token * ROUTES + g];
        if constexpr (SHIFTED)
            lane_prev = p.shifted_prev_mix[token * ROUTES + g];
#pragma unroll
        for (int i = 0; i < ROUTES; ++i)
            lane_comb[i] = p.comb_res_mix[(token * ROUTES + i) * ROUTES + g];
    } else {
#pragma unroll
        for (int j = 0; j < ROUTES; ++j) {
            post[j] = p.post_mix[token * ROUTES + j];
            prev[j] = SHIFTED ? p.shifted_prev_mix[token * ROUTES + j] : 0.f;
#pragma unroll
            for (int i = 0; i < ROUTES; ++i)
                comb[i][j] = p.comb_res_mix[(token * ROUTES + i) * ROUTES + j];
        }
    }
    const auto post_coeff = [&](int j) {
        if constexpr (DISTRIBUTED_COEFF)
            return __shfl(lane_post, j * 16 + r16);
        else
            return post[j];
    };
    const auto prev_coeff = [&](int j) {
        if constexpr (DISTRIBUTED_COEFF)
            return __shfl(lane_prev, j * 16 + r16);
        else
            return prev[j];
    };
    const auto comb_coeff = [&](int i, int j) {
        if constexpr (DISTRIBUTED_COEFF)
            return __shfl(lane_comb[i], j * 16 + r16);
        else
            return comb[i][j];
    };

    // Loads past the split repeat its last step (unconditional loads keep the prefetch in flight)
    // fn of a step: thread t loads 4 floats of output o, route j (96 rows of 64 values, 6 rounds)
    constexpr int FN_ROUNDS = (PACKED_FN == 1 ? 8 : 6) * 256 / THREADS;
    auto load_fn = [&](int s, v4f(&f)[FN_ROUNDS]) {
        s = min(s, s_end - 1);
        if constexpr (PACKED_FN) {
#pragma unroll
            for (int rd = 0; rd < FN_ROUNDS; ++rd)
                f[rd] = *reinterpret_cast<const v4f *>(packed_fn + (size_t)s * (STAGE_BYTES / 2) +
                                                       (rd * THREADS + tid) * 8);
        } else {
#pragma unroll
            for (int rd = 0; rd < FN_ROUNDS; ++rd) {
                const int idx = rd * (THREADS / 16) + tid / 16, o = idx / ROUTES, j = idx % ROUTES;
                f[rd] = *(const v4f *)(p.fn + (size_t)o * ROUTES * H + j * H + s * STEP + 4 * (tid % 16));
            }
        }
    };
    // ... into the operand slots: output o is row o % 16 of fragment o / 16; hidden hh of the step
    // is value 8 u + e (u = hh / 8 % 2, e = hh % 8) of lane group hh / 16
    auto store_fn = [&](int stage, const v4f(&f)[FN_ROUNDS]) {
        if constexpr (PACKED_FN) {
#pragma unroll
            for (int rd = 0; rd < FN_ROUNDS; ++rd)
                reinterpret_cast<v4f *>(smem + stage * STAGE_BYTES)[rd * THREADS + tid] = f[rd];
        } else {
#pragma unroll
            for (int rd = 0; rd < FN_ROUNDS; ++rd) {
                const int idx = rd * (THREADS / 16) + tid / 16, o = idx / ROUTES, j = idx % ROUTES, hh = 4 * (tid % 16);
                const int u = hh / 8 % 2, slot_lane = 16 * (hh / 16) + o % 16;
                uint8_t *slot =
                    smem + stage * STAGE_BYTES + (((j * 2 + u) * 2 + o / 16) * 2 * 64 + slot_lane) * 16 + (hh % 8) * 2;
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
                t.res[i][u] =
                    (v4i)__builtin_amdgcn_raw_buffer_load_b128(res_rsrc, ((row * ROUTES + i) * H + h) * 2, 0, 0);
        }
    };

    v4f acc[2] = {v4f{}, v4f{}};
    float sq = 0.f, u_sq = 0.f;
    v2f sq_pair{}, u_sq_pair{};
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
            if constexpr (STREAM_ROUTES) {
                // Consume each rounded route immediately instead of retaining all
                // four routes as both FP32 and BF16 register arrays. The per-output
                // FMA and MFMA accumulation orders are identical to the default path.
                float pre_input[8] = {};
#pragma unroll
                for (int j = 0; j < ROUTES; ++j) {
                    float values[8], rounded_route[8];
                    v4i route;
#pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        values[e] = xv[e] * post_coeff(j);
#pragma unroll
                        for (int i = 0; i < ROUTES; ++i)
                            values[e] = __builtin_fmaf(comb_coeff(i, j), rv[i][e], values[e]);
                    }
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        route[q] = (int)pack_bf16x2(values[2 * q], values[2 * q + 1]);
                        unpack_bf16x2((uint32_t)route[q], rounded_route[2 * q], rounded_route[2 * q + 1]);
                    }
                    __builtin_amdgcn_raw_buffer_store_b128((v4u)route, out_rsrc, ((row * ROUTES + j) * H + h) * 2, 0,
                                                           RES_STORE_POLICY);
                    if constexpr (PAIR_STATS) {
#pragma unroll
                        for (int q = 0; q < 4; ++q) {
                            const v2f value{rounded_route[2 * q], rounded_route[2 * q + 1]};
                            sq_pair = __builtin_elementwise_fma(value, value, sq_pair);
                        }
                    }
#pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        if constexpr (!PAIR_STATS)
                            sq = __builtin_fmaf(rounded_route[e], rounded_route[e], sq);
                        if constexpr (SHIFTED) {
                            if (j == 0)
                                pre_input[e] = rounded_route[e] * prev_coeff(j);
                            else
                                pre_input[e] = __builtin_fmaf(rounded_route[e], prev_coeff(j), pre_input[e]);
                        }
                    }
#pragma unroll
                    for (int f = 0; f < 2; ++f) {
                        const int logical_o = min(16 * f + r16, 23);
                        const int o = FN_SWIZZLE ? (logical_o ^ ((g >> 1) * 4)) : logical_o;
                        const uint8_t *slot = PACKED_FN == 2 ? operands + ((((j * 2 + u) * 2) * 4 + g) * 24 + o) * 16
                                                             : operands + (((j * 2 + u) * 2 + f) * 2 * 64 + lane) * 16;
                        acc[f] = dot<FORCE_ACC>(*(const v4i *)slot, route, acc[f]);
                        acc[f] = dot<FORCE_ACC>(*(const v4i *)(slot + (PACKED_FN == 2 ? 4 * 24 * 16 : 64 * 16)), route,
                                                acc[f]);
                    }
                }
                if constexpr (SHIFTED) {
                    v4i norm;
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        norm[q] = (int)pack_bf16x2(pre_input[2 * q], pre_input[2 * q + 1]);
                        if constexpr (SAVE_U_SQR) {
                            float a, b;
                            unpack_bf16x2((uint32_t)norm[q], a, b);
                            if constexpr (PAIR_STATS)
                                u_sq_pair = __builtin_elementwise_fma(v2f{a, b}, v2f{a, b}, u_sq_pair);
                            else {
                                u_sq = __builtin_fmaf(a, a, u_sq);
                                u_sq = __builtin_fmaf(b, b, u_sq);
                            }
                        }
                    }
                    if constexpr (RETAIN_U) {
                        const int local_h = h - s_begin * STEP;
                        const int offset = LOCAL_U_SWIZZLE ? ((local_h / 8) ^ (row & 15)) * 8 : local_h;
                        *reinterpret_cast<v4i *>(local_u + row * (s_end - s_begin) * STEP + offset) = norm;
                    } else
                        __builtin_amdgcn_raw_buffer_store_b128((v4u)norm, norm_rsrc, (row * H + h) * 2, 0, 0);
                }
            } else {
                v4i nr[ROUTES];
                float rounded[ROUTES][8];
#pragma unroll
                for (int j = 0; j < ROUTES; ++j) {
                    if constexpr (PACKED_POST) {
#pragma unroll
                        for (int q = 0; q < 4; ++q) {
                            v2f v = v2f{xv[2 * q], xv[2 * q + 1]} * v2f{post_coeff(j), post_coeff(j)};
#pragma unroll
                            for (int i = 0; i < ROUTES; ++i)
                                v = __builtin_elementwise_fma(v2f{comb_coeff(i, j), comb_coeff(i, j)},
                                                              v2f{rv[i][2 * q], rv[i][2 * q + 1]}, v);
                            nr[j][q] = (int)pack_bf16x2(v[0], v[1]);
                            unpack_bf16x2((uint32_t)nr[j][q], rounded[j][2 * q], rounded[j][2 * q + 1]);
                        }
                    } else {
                        float v[8];
#pragma unroll
                        for (int e = 0; e < 8; ++e) {
                            v[e] = xv[e] * post_coeff(j);
#pragma unroll
                            for (int i = 0; i < ROUTES; ++i)
                                v[e] = __builtin_fmaf(comb_coeff(i, j), rv[i][e], v[e]);
                        }
#pragma unroll
                        for (int q = 0; q < 4; ++q) {
                            nr[j][q] = (int)pack_bf16x2(v[2 * q], v[2 * q + 1]);
                            unpack_bf16x2((uint32_t)nr[j][q], rounded[j][2 * q], rounded[j][2 * q + 1]);
                        }
                    }
                    // Shifted consumers read U immediately, but R' only in the next
                    // sublayer. An experimental non-temporal R' store can avoid evicting U.
                    // The default path retains the original cache policy and numerics.
                    __builtin_amdgcn_raw_buffer_store_b128((v4u)nr[j], out_rsrc, ((row * ROUTES + j) * H + h) * 2, 0,
                                                           RES_STORE_POLICY);
                    if constexpr (PAIR_STATS) {
#pragma unroll
                        for (int q = 0; q < 4; ++q) {
                            const v2f value{rounded[j][2 * q], rounded[j][2 * q + 1]};
                            sq_pair = __builtin_elementwise_fma(value, value, sq_pair);
                        }
                    } else {
#pragma unroll
                        for (int e = 0; e < 8; ++e)
                            sq = __builtin_fmaf(rounded[j][e], rounded[j][e], sq);
                    }
                }
                if constexpr (SHIFTED) {
                    v4i n;
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        float a[2];
                        if constexpr (PACKED_POST) {
                            v2f v = v2f{rounded[0][2 * q], rounded[0][2 * q + 1]} * v2f{prev_coeff(0), prev_coeff(0)};
#pragma unroll
                            for (int j = 1; j < ROUTES; ++j)
                                v = __builtin_elementwise_fma(v2f{rounded[j][2 * q], rounded[j][2 * q + 1]},
                                                              v2f{prev_coeff(j), prev_coeff(j)}, v);
                            a[0] = v[0];
                            a[1] = v[1];
                        } else {
#pragma unroll
                            for (int e = 0; e < 2; ++e) {
                                a[e] = rounded[0][2 * q + e] * prev_coeff(0);
#pragma unroll
                                for (int j = 1; j < ROUTES; ++j)
                                    a[e] = __builtin_fmaf(rounded[j][2 * q + e], prev_coeff(j), a[e]);
                            }
                        }
                        n[q] = (int)pack_bf16x2(a[0], a[1]);
                        if constexpr (SAVE_U_SQR) {
                            float ua, ub;
                            unpack_bf16x2((uint32_t)n[q], ua, ub);
                            if constexpr (PAIR_STATS)
                                u_sq_pair = __builtin_elementwise_fma(v2f{ua, ub}, v2f{ua, ub}, u_sq_pair);
                            else {
                                u_sq = __builtin_fmaf(ua, ua, u_sq);
                                u_sq = __builtin_fmaf(ub, ub, u_sq);
                            }
                        }
                    }
                    if constexpr (RETAIN_U) {
                        const int local_h = h - s_begin * STEP;
                        const int offset = LOCAL_U_SWIZZLE ? ((local_h / 8) ^ (row & 15)) * 8 : local_h;
                        *reinterpret_cast<v4i *>(local_u + row * (s_end - s_begin) * STEP + offset) = n;
                    } else
                        __builtin_amdgcn_raw_buffer_store_b128((v4u)n, norm_rsrc, (row * H + h) * 2, 0, 0);
                }
#pragma unroll
                for (int j = 0; j < ROUTES; ++j)
#pragma unroll
                    for (int f = 0; f < 2; ++f) {
                        const int logical_o = min(16 * f + r16, 23); // only unused output rows are duplicated
                        const int o = FN_SWIZZLE ? (logical_o ^ ((g >> 1) * 4)) : logical_o;
                        const uint8_t *slot = PACKED_FN == 2 ? operands + ((((j * 2 + u) * 2) * 4 + g) * 24 + o) * 16
                                                             : operands + (((j * 2 + u) * 2 + f) * 2 * 64 + lane) * 16;
                        acc[f] = dot<FORCE_ACC>(*(const v4i *)slot, nr[j], acc[f]);
                        acc[f] = dot<FORCE_ACC>(*(const v4i *)(slot + (PACKED_FN == 2 ? 4 * 24 * 16 : 64 * 16)), nr[j],
                                                acc[f]);
                    }
            }
        }
        if constexpr (IGLP_MODE >= 0)
            __builtin_amdgcn_iglp_opt(IGLP_MODE);
    };

    // output rows 24 .. 31 (fragment 1) stay zero
    if constexpr (!PACKED_FN) {
        for (int i = tid; i < LDS_BYTES / 16; i += THREADS)
            ((v4i *)smem)[i] = v4i{};
        __syncthreads();
    }
    Step even, odd;
    if constexpr (DMA_FN) {
        static_assert(PACKED_FN == 2, "Direct Fn DMA requires the compact prepacked layout");
        const auto fn_rsrc = make_rsrc(packed_fn, uint32_t(steps * STAGE_BYTES));
        auto dma_fn = [&](int s, int stage) {
            s = min(s, s_end - 1);
#pragma unroll
            for (int rd = 0; rd < FN_ROUNDS; ++rd)
                __builtin_amdgcn_raw_ptr_buffer_load_lds(
                    fn_rsrc, (lds_ptr_t)(smem + stage * STAGE_BYTES + (rd * (THREADS / 64) + wave) * 1024), 16,
                    lane * 16, __builtin_amdgcn_readfirstlane(s * STAGE_BYTES + rd * THREADS * 16 + wave * 1024), 0, 0);
        };
        // The direct global-to-LDS primitive is shared with the existing AMD
        // GEMM backend. It removes both register-held Fn prefetch arrays.
        dma_fn(s_begin, 0);
        load_step(s_begin, even);
        asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)\n\ts_barrier" ::: "memory");
        for (int s = s_begin; s < s_end; s += 2) {
            dma_fn(s + 1, 1);
            load_step(s + 1, odd);
            compute(s, even, 0);
            asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)\n\ts_barrier" ::: "memory");
            if (s + 1 >= s_end)
                break;
            dma_fn(s + 2, 0);
            load_step(s + 2, even);
            compute(s + 1, odd, 1);
            asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)\n\ts_barrier" ::: "memory");
        }
    } else {
        v4f fn_even[FN_ROUNDS], fn_odd[FN_ROUNDS];
        load_fn(s_begin, fn_even);
        store_fn(0, fn_even);
        load_step(s_begin, even);
        __syncthreads();
        // two steps per iteration on alternating registers (no copies between the prefetch and its use);
        // the barriers order LDS only, so the prefetches and stores stay in flight
        for (int s = s_begin; s < s_end; s += 2) {
            if (!TRIM_PREFETCH || s + 1 < s_end) {
                load_fn(s + 1, fn_odd);
                if constexpr (PREFETCH_INPUT)
                    load_step(s + 1, odd);
            }
            compute(s, even, 0);
            if (s + 1 < s_end)
                store_fn(1, fn_odd);
            if (!TRIM_FINAL_SYNC || s + 1 < s_end)
                asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
            if (s + 1 >= s_end)
                break;
            if (!TRIM_PREFETCH || s + 2 < s_end) {
                load_fn(s + 2, fn_even);
                if constexpr (PREFETCH_INPUT)
                    load_step(s + 2, even);
            }
            if constexpr (!PREFETCH_INPUT)
                load_step(s + 1, even);
            compute(s + 1, PREFETCH_INPUT ? odd : even, 1);
            if (s + 2 < s_end)
                store_fn(0, fn_even);
            // The caller's publication/claim join precedes any next-task LDS reuse.
            if (!TRIM_FINAL_SYNC || s + 2 < s_end)
                asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
            if constexpr (!PREFETCH_INPUT)
                if (s + 2 < s_end)
                    load_step(s + 2, even);
        }
    }

    // acc[f][e] of lane (g, r16) is the mix 16 f + 4 g + e of the wave's token r16
    if constexpr (PAIR_STATS) {
        sq = sq_pair[0] + sq_pair[1];
        u_sq = u_sq_pair[0] + u_sq_pair[1];
    }
    // Workspace stays in logical 64-row groups even for a wider producer.
    // A final 128-row tile can contain an entirely absent second group.
    const int group64 = BM == 32 ? m_block / 2 : m_block * (BM / 64) + row / 64;
    const int partial_row = BM == 32 ? (m_block % 2) * 32 + row : row % 64;
    const size_t task = (size_t)group64 * p.splits + split;
    const bool partial_valid = group64 < (p.num_tokens + 63) / 64;
    if (partial_valid) {
        float *mixes = p.mix_partials + (task * 64 + partial_row) * OUTPUTS;
        if constexpr (COHERENT_PARTIALS) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                __hip_atomic_store(mixes + 4 * g + e, acc[0][e], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                if (g < 2)
                    __hip_atomic_store(mixes + 16 + 4 * g + e, acc[1][e], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            }
        } else {
            *(v4f *)(mixes + 4 * g) = acc[0];
            if (g < 2)
                *(v4f *)(mixes + 16 + 4 * g) = acc[1];
        }
    }
    sq += __shfl_xor(sq, 16);
    sq += __shfl_xor(sq, 32);
    if (g == 0 && partial_valid) {
        if constexpr (COHERENT_PARTIALS)
            __hip_atomic_store(p.sqr_partials + task * 64 + partial_row, sq, __ATOMIC_RELAXED,
                               __HIP_MEMORY_SCOPE_AGENT);
        else
            p.sqr_partials[task * 64 + partial_row] = sq;
    }
    if constexpr (SAVE_U_SQR) {
        u_sq += __shfl_xor(u_sq, 16);
        u_sq += __shfl_xor(u_sq, 32);
        if (g == 0 && partial_valid) {
            if constexpr (COHERENT_PARTIALS)
                __hip_atomic_store(u_sqr_partials + task * 64 + partial_row, u_sq, __ATOMIC_RELAXED,
                                   __HIP_MEMORY_SCOPE_AGENT);
            else
                u_sqr_partials[task * 64 + partial_row] = u_sq;
        }
    }
#endif
}

// Native CDNA4 MegaMoE consumes plain MN-major scale rows. CUDA's
// 32x4 UTCCP permutation is not part of this backend's buffer contract.
__device__ __forceinline__ int64_t shared_sf_row(int token, int block_m) {
    (void)block_m;
    return token;
}

// The mixes of a token per quad of lanes: lane k adds the split partials k, k + 4, ... in split
// order and the quad adds its lanes in a fixed butterfly (an order set by the split count alone),
// then the RMS scaling of new_residual, the sigmoids (lane k: previous and post mix k) and the
// Sinkhorn normalization of the 4 x 4 combination (lane k: row k, columns over the quad)
template <int BATCH = 4, bool FAST_RECIPROCAL = false, bool COHERENT_PARTIALS = false>
__device__ __forceinline__ void mega_mhc_mix_impl(const MegaMhcArgs &p, int token_override = -1,
                                                  int thread_override = threadIdx.x) {
    using namespace mega_mhc;
    const auto reciprocal = [](float x) {
        if constexpr (FAST_RECIPROCAL)
            return reciprocal_positive(x);
        else
            return 1.f / x;
    };
    const int thread = (token_override < 0 ? blockIdx.x * blockDim.x : token_override * ROUTES) + thread_override;
    const int k = thread % ROUTES;
    const int token = min(thread / ROUTES, p.num_tokens - 1);
    const size_t task0 = (size_t)(token / BM) * p.splits;
    const int r = token % BM;
    const auto quad_sum = [](float v) {
        v += __int_as_float(
            __builtin_amdgcn_update_dpp(0, __float_as_int(v), 0xb1, 0xf, 0xf, false)); // quad_perm 1 0 3 2
        return v + __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x4e, 0xf, 0xf, false)); // 2 3 0 1
    };
    // the loads of four of the lane's splits ahead of their sums
    float mix[OUTPUTS] = {}, sqr = 0.f;
    for (int s0 = k; s0 < p.splits; s0 += BATCH * ROUTES) {
        v4f v[BATCH][OUTPUTS / 4];
        float sq[BATCH];
#pragma unroll
        for (int b = 0; b < BATCH; ++b) {
            const int s = min(s0 + b * ROUTES, p.splits - 1);
            const v4f *partial = (const v4f *)(p.mix_partials + ((task0 + s) * BM + r) * OUTPUTS);
#pragma unroll
            for (int q = 0; q < OUTPUTS / 4; ++q) {
                if constexpr (COHERENT_PARTIALS) {
#pragma unroll
                    for (int e = 0; e < 4; ++e)
                        v[b][q][e] = __hip_atomic_load(reinterpret_cast<const float *>(partial) + q * 4 + e,
                                                       __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                } else
                    v[b][q] = partial[q];
            }
            if constexpr (COHERENT_PARTIALS)
                sq[b] = __hip_atomic_load(p.sqr_partials + (task0 + s) * BM + r, __ATOMIC_RELAXED,
                                          __HIP_MEMORY_SCOPE_AGENT);
            else
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
        p.prev_mix[token * ROUTES + k] = reciprocal(1.f + __expf(-pre)) + p.hc_pre_eps;
        p.new_post_mix[token * ROUTES + k] = reciprocal(1.f + __expf(-post)) * p.hc_post_scale;
    }
    // one division per row or column
    const float row_max = fmaxf(fmaxf(c[0], c[1]), fmaxf(c[2], c[3]));
    float sum = 0.f;
#pragma unroll
    for (int j = 0; j < ROUTES; ++j)
        sum += c[j] = __expf(c[j] - row_max);
    float inv = reciprocal(sum);
#pragma unroll
    for (int j = 0; j < ROUTES; ++j)
        c[j] = c[j] * inv + p.sinkhorn_eps;
    const auto normalize_cols = [&] {
#pragma unroll
        for (int j = 0; j < ROUTES; ++j)
            c[j] *= reciprocal(quad_sum(c[j]) + p.sinkhorn_eps);
    };
    normalize_cols();
    for (int it = 1; it < p.num_sinkhorn_iters; ++it) {
        inv = reciprocal(c[0] + c[1] + c[2] + c[3] + p.sinkhorn_eps);
#pragma unroll
        for (int j = 0; j < ROUTES; ++j)
            c[j] *= inv;
        normalize_cols();
    }
    if (valid)
        *(v4f *)(p.new_comb_res_mix + (token * ROUTES + k) * ROUTES) = v4f{c[0], c[1], c[2], c[3]};
}

// CHUNKS = hidden / 1024: thread t owns hidden 4 (t + 256 c) .. + 3 of chunk c
template <bool SHIFTED, int CHUNKS, bool FAST_QUANT = false>
__device__ __forceinline__ void mega_mhc_norm_impl(const MegaMhcArgs &p) {
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
    auto load_input = [&](int token, v2u(&in)[INPUTS][CHUNKS]) {
#pragma unroll
        for (int j = 0; j < INPUTS; ++j)
#pragma unroll
            for (int c = 0; c < CHUNKS; ++c)
                in[j][c] =
                    token >= p.num_tokens ? v2u{}
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
        const float rms =
            rsqrtf((wave_sums[parity][0] + wave_sums[parity][1] + wave_sums[parity][2] + wave_sums[parity][3]) / H +
                   p.rmsnorm_eps);

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
                    amax = fmaxf(amax,
                                 __int_as_float(__builtin_amdgcn_update_dpp(__float_as_int(amax), __float_as_int(amax),
                                                                            decltype(ctrl)::value, 0xf, 0xf, false)));
                };
                dpp_max(std::integral_constant<int, 0xb1>{});  // quad_perm 1 0 3 2
                dpp_max(std::integral_constant<int, 0x4e>{});  // quad_perm 2 3 0 1
                dpp_max(std::integral_constant<int, 0x141>{}); // row_half_mirror
                const uint32_t bits = __float_as_uint(fmaxf(amax, 1e-4f) / 448.f);
                const int exponent = min(max((int)(bits >> 23 & 0xff) + ((bits & 0x7fffff) != 0), 1), 254);
                const float inv =
                    FAST_QUANT ? inverse_scale(exponent) : 1.f / __uint_as_float((uint32_t)exponent << 23);
                const int lo = __builtin_amdgcn_cvt_pk_fp8_f32(yv[0] * inv, yv[1] * inv, 0, false);
                const int packed = __builtin_amdgcn_cvt_pk_fp8_f32(yv[2] * inv, yv[3] * inv, lo, true);
                *(int *)(p.y_fp8 + (size_t)token * H + h) = packed;
                // the block's leader stores byte (t / 8) % 4 of the word of hidden 128 (t / 32 + 8 c) .. + 127
                if constexpr (FAST_QUANT) {
                    uint32_t word = exponent;
                    word |= uint32_t(__shfl(exponent, (lane & ~31) + 8)) << 8;
                    word |= uint32_t(__shfl(exponent, (lane & ~31) + 16)) << 16;
                    word |= uint32_t(__shfl(exponent, (lane & ~31) + 24)) << 24;
                    if (lane % 32 == 0) {
                        const int w = tid / 32 + 8 * c;
                        p.y_sf[token * p.y_sf_stride_token + w * p.y_sf_stride_word] = word;
                        if (p.y_shared_sf)
                            p.y_shared_sf[shared_sf_row(token, p.shared_sf_block_m) + w * p.y_shared_sf_stride_word] =
                                word;
                    }
                } else if (lane % 8 == 0) {
                    const int w = tid / 32 + 8 * c, b = tid / 8 % 4;
                    ((uint8_t *)(p.y_sf + token * p.y_sf_stride_token + w * p.y_sf_stride_word))[b] = (uint8_t)exponent;
                    if (p.y_shared_sf != nullptr)
                        ((uint8_t *)(p.y_shared_sf + shared_sf_row(token, p.shared_sf_block_m) +
                                     w * p.y_shared_sf_stride_word))[b] = (uint8_t)exponent;
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
