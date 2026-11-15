// AMD GPU experimental cache-domain-local persistent mHC.
// All producers/consumers of one token tile are selected by their physical
// cache-domain ID. Payload never crosses a cache domain inside this dispatch.
// The custom ISA ordering below is ONLY valid under that ownership invariant:
// vector-store completion + CTA synchronization before relaxed publication;
// coherent flag polling + vector L1 invalidation before payload consumption.
// It is NOT a general replacement for agent-wide release/acquire fences.
#pragma once
#include <deep_gemm/impls/cdna4/mega_mhc_roles.cuh>

namespace deep_gemm::cdna4::mhc_cache_local {

__device__ __forceinline__ int domain_id() {
#if defined(__HIP_DEVICE_COMPILE__)
    return __builtin_amdgcn_s_getreg(20 | (31 << 11));
#else
    return 0;
#endif
}

__device__ __forceinline__ void publish_payload() {
#if defined(__HIP_DEVICE_COMPILE__)
    asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
#endif
    __syncthreads();
}

__device__ __forceinline__ void acquire_payload() {
#if defined(__HIP_DEVICE_COMPILE__)
    asm volatile("buffer_inv sc0\n\ts_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
#endif
}

template <int H, bool FAST_MIX = false, bool NT_RESIDUAL = false, bool PACKED_POST = false, bool STREAM_ROUTES = false,
          bool DMA_FN = false, bool EXACT_SCALE_INVERSE = false, bool PREFETCH_INPUT = true, bool PAIR_STATS = false,
          bool PROFILE = false, int TILE_M = 64, bool FORCE_ACC = false, int IGLP_MODE = -1,
          bool PARALLEL_NORM_STATS = false, int FIXED_SPLITS = 0, bool DISTRIBUTED_COEFF = false,
          bool PACKED_NORM_AMAX = false, bool FN_SWIZZLE = false, bool TRIM_PREFETCH = false,
          bool TRIM_FINAL_SYNC = false>
__device__ __forceinline__ void run(MegaMhcArgs p, int *counters, const uint16_t *fn, int domains,
                                    unsigned long long *clocks = nullptr) {
#if defined(__HIP_DEVICE_COMPILE__)
    extern __shared__ __attribute__((aligned(16))) uint8_t smem[];
    if constexpr (FIXED_SPLITS > 0) {
        // Optional shape-specialized dispatch; the host must reject mismatches.
        p.hidden = H;
        p.splits = FIXED_SPLITS;
    }
    constexpr int STRIDE = 32;
    static_assert(TILE_M == 32 || TILE_M == 64 || TILE_M == 128, "Unsupported cache-local producer tile");
    constexpr int WAVES = TILE_M / 16, RATIO = TILE_M == 128 ? 2 : 1, HALVES = TILE_M == 32 ? 2 : 1;
    constexpr int NORM_JOBS = 64 / (2 * WAVES), JOBS = 1 + NORM_JOBS;
    const int tid = threadIdx.x, wave = __builtin_amdgcn_readfirstlane(tid / 64), lane = tid % 64;
    const int domain = domain_id(), groups = (p.num_tokens + 63) / 64;
    unsigned long long started = 0, mark = 0, core = 0, publish = 0, wait = 0, mix = 0, norm = 0, claims = 0;
    unsigned long long producer_jobs = 0, mix_jobs = 0, norm_jobs = 0, norm_rows = 0;
    auto tick = []() {
        asm volatile("" ::: "memory");
        return clock64();
    };
    if constexpr (PROFILE)
        if (tid == 0)
            started = tick();
    if (domain < 0 || domain >= domains)
        return; // host probes IDs before enabling this experiment
    // Narrow producer halves of a logical 64-row group must use the SAME
    // cache domain. Otherwise the special payload protocol would be invalid.
    const int tiles = (p.num_tokens + 64 * RATIO - 1) / (64 * RATIO);
    const int local_groups = tiles > domain ? (tiles - 1 - domain) / domains + 1 : 0;
    auto *ready = counters;
    auto *done = counters + groups * STRIDE;
    auto *heads = counters + groups * STRIDE * 2;
    auto *producer_head = heads + domain * STRIDE * 2;
    auto *consumer_head = producer_head + STRIDE;
    auto *exits = heads + domains * STRIDE * 2;
    auto *u_stats = p.sqr_partials + groups * p.splits * 64;
    auto claim = [&](int *head) {
        unsigned long long t = 0;
        if constexpr (PROFILE)
            if (tid == 0)
                t = tick();
        auto *slot = reinterpret_cast<int *>(smem);
        if (tid == 0)
            *slot = __hip_atomic_fetch_add(head, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        __syncthreads();
        const int ticket = __builtin_amdgcn_readfirstlane(*slot);
        __syncthreads();
        if constexpr (PROFILE)
            if (tid == 0)
                claims += tick() - t;
        return ticket;
    };
    // Exhaust the domain's producer queue before waiting on consumers. A
    // claimed producer belongs to an already-running CTA in THIS domain;
    // progress never requires an as-yet-unscheduled CTA or a different domain.
    for (int task = claim(producer_head); task < local_groups * p.splits * HALVES; task = claim(producer_head)) {
        const int group = (task / (p.splits * HALVES)) * domains + domain, split = task % p.splits;
        const int half = (task / p.splits) % HALVES;
        const int producer_tile = group * HALVES + half;
        if constexpr (PROFILE)
            if (tid == 0)
                mark = tick();
        // For this instruction family, cache-policy bit 1 is the NT hint;
        // it changes replacement priority, not the payload's visibility scope.
        // The final missing half still publishes readiness, but has no payload.
        if (producer_tile * TILE_M < p.num_tokens)
            mega_mhc_residual_impl<true, 2, true, NT_RESIDUAL ? 2 : 0, PACKED_POST, STREAM_ROUTES, DMA_FN,
                                   PREFETCH_INPUT, PAIR_STATS, TILE_M, FORCE_ACC, IGLP_MODE, DISTRIBUTED_COEFF,
                                   FN_SWIZZLE, false, false, false, TRIM_PREFETCH, TRIM_FINAL_SYNC>(p, producer_tile,
                                                                                                    split, fn, u_stats);
        if constexpr (PROFILE)
            if (tid == 0) {
                core += tick() - mark;
                mark = tick();
                ++producer_jobs;
            }
        publish_payload();
        if (tid == 0) {
#pragma unroll
            for (int sub = 0; sub < RATIO; ++sub)
                if (group * RATIO + sub < groups)
                    __hip_atomic_fetch_add(ready + (group * RATIO + sub) * STRIDE, 1, __ATOMIC_RELAXED,
                                           __HIP_MEMORY_SCOPE_AGENT);
        }
        __syncthreads();
        if constexpr (PROFILE)
            if (tid == 0)
                publish += tick() - mark;
    }

    // Both logical 64-row groups of a wide tile stay in the producer's cache
    // domain. Workspace and output layouts remain the original 64-row ABI.
    for (int job = claim(consumer_head); job < local_groups * RATIO * JOBS; job = claim(consumer_head)) {
        const int tile = (job / (RATIO * JOBS)) * domains + domain, within = job % (RATIO * JOBS);
        const int group = tile * RATIO + within / JOBS, kind = within % JOBS;
        if (group >= groups)
            continue;
        if constexpr (PROFILE)
            if (tid == 0)
                mark = tick();
        if (tid == 0) {
            while (__hip_atomic_load(ready + group * STRIDE, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) !=
                   p.splits * HALVES)
                __builtin_amdgcn_s_sleep(1);
        }
        __syncthreads();
        acquire_payload();
        if constexpr (PROFILE)
            if (tid == 0) {
                wait += tick() - mark;
                mark = tick();
            }
        if (kind == 0) {
            if (tid < 256)
                mega_mhc_mix_impl<1, FAST_MIX>(p, group * 64, tid);
            if constexpr (TILE_M == 32)
                mega_mhc_mix_impl<1, FAST_MIX>(p, group * 64 + 32, tid);
        } else {
            const int pair = group * 32 + (kind - 1) * WAVES + wave;
            if (pair * 2 < p.num_tokens)
                mhc_roles::norm_pair<H, 1, EXACT_SCALE_INVERSE, PARALLEL_NORM_STATS, PACKED_NORM_AMAX>(p, pair, 0,
                                                                                                       u_stats);
        }
        __syncthreads();
        if constexpr (PROFILE)
            if (tid == 0) {
                if (kind == 0) {
                    mix += tick() - mark;
                    ++mix_jobs;
                } else {
                    norm += tick() - mark;
                    ++norm_jobs;
                    norm_rows += max(0, min(2 * WAVES, p.num_tokens - group * 64 - (kind - 1) * 2 * WAVES));
                }
            }
        if (tid == 0 &&
            __hip_atomic_fetch_add(done + group * STRIDE, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) == NORM_JOBS) {
            __hip_atomic_store(ready + group * STRIDE, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            __hip_atomic_store(done + group * STRIDE, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
        __syncthreads();
    }
    publish_payload();
    if (tid == 0 &&
        __hip_atomic_fetch_add(exits, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) == int(gridDim.x) - 1) {
        // Every CTA has exhausted both queues. Kernel/stream completion provides
        // inter-dispatch visibility; no next call starts while these are reset.
        for (int d = 0; d < domains; ++d) {
            __hip_atomic_store(heads + d * STRIDE * 2, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            __hip_atomic_store(heads + d * STRIDE * 2 + STRIDE, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
        __hip_atomic_store(exits, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    if constexpr (PROFILE)
        if (tid == 0) {
            auto *out = clocks + blockIdx.x * 16;
            const auto total = tick() - started;
            out[0] = total;
            out[1] = core;
            out[2] = publish;
            out[3] = wait;
            out[4] = mix;
            out[5] = norm;
            out[6] = claims;
            out[7] = total - core - publish - wait - mix - norm - claims;
            out[8] = producer_jobs;
            out[9] = mix_jobs;
            out[10] = norm_jobs;
            out[11] = norm_rows;
            out[12] = 0;
            out[13] = 0;
            out[14] = 0;
            out[15] = 1;
        }
#endif
}

} // namespace deep_gemm::cdna4::mhc_cache_local
