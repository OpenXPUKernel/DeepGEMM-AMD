#pragma once
#include <deep_gemm/impls/cdna4_mega_moe_common.cuh>
#include <deep_gemm/layout/cdna4_mega_moe.cuh>
// One cooperative launch owns routing, gather, both GEMMs, shared experts,
// peer return and reduction. Producer workgroups and compute workgroups remain
// resident together; no consumer can evict the producers it is waiting for.
namespace resident_moe {
using deep_gemm::cdna4::MegaMoeKernelArgs;
using deep_gemm::cdna4::MegaMoeRowBlock;
using namespace mega_moe;
using namespace uni_symm::device;
constexpr int CounterSpacing = (32);
enum {
    PullHead = 0,
    FirstHead = CounterSpacing,
    SecondHead = 2 * CounterSpacing,
    SecondTail = 3 * CounterSpacing,
    SecondDone = 4 * CounterSpacing,
    BlockCount = 5 * CounterSpacing,
    StateSize = (6 * CounterSpacing)
};
constexpr int ControlRelayWord = 65536 / sizeof(uint32_t) - 1;

constexpr int M = 128;
static_assert(M == 64 || M == 128 || M == 256);
// Monotonic per-workgroup epochs prevent a fast peer from erasing a phase
// before a slower peer observes it. The control words are in a proven-unused,
// initially-zero symmetric-buffer tail, independent of graph-private state.
template <bool SystemAcquire = true, bool CompletedCoherentWriters = false>
__device__ __forceinline__ void grid_sync(const MegaMoeKernelArgs &q, uint32_t tag) {
    auto *flags = reinterpret_cast<uint32_t *>(q.k_ready);
    constexpr auto LocalScope = (Scope::system);

    if constexpr (CompletedCoherentWriters) {
        // Every lane completed explicit system-coherent return stores. Keep the
        // all-writer join and immutable epoch protocol; do not redundantly flush
        // unrelated cache lines at this one final publication point.
        asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)\n\ts_barrier" ::: "memory");
        if (threadIdx.x == 0)
            store_relaxed<LocalScope>(flags + blockIdx.x, tag);
    } else {
        fence_release<LocalScope>();
        __syncthreads();
        if (threadIdx.x == 0)
            store_release<LocalScope>(flags + blockIdx.x, tag);
    }
    __syncthreads();
    for (int peer = threadIdx.x; peer < int(gridDim.x); peer += blockDim.x) {
        uint32_t observed;
        do {
            observed = load_relaxed<LocalScope>(flags + peer);
            if (!observed || int32_t(observed - tag) < 0)
                __builtin_amdgcn_s_sleep(4);
        } while (!observed || int32_t(observed - tag) < 0);
    }
    __syncthreads();
    if constexpr (SystemAcquire)
        fence_acquire<LocalScope>();
    else
        fence_acquire<Scope::device>();
    __syncthreads();
}
// Metadata and post-peer phases have a single payload-writing workgroup.
// All workgroups still publish and observe the same monotonic epoch words.
// Producer 0 releases every lane's writes, including the peer-barrier counter;
// the other workgroups have no new payload to release in these phases.
template <bool SystemAcquire = true>
__device__ __forceinline__ void grid_sync_control(const MegaMoeKernelArgs &q, uint32_t tag) {
    auto *flags = reinterpret_cast<uint32_t *>(q.k_ready);
    if (blockIdx.x == 0)
        fence_release<Scope::system>();
    asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)\n\ts_barrier" ::: "memory");
    if (threadIdx.x == 0)
        store_relaxed<Scope::system>(flags + blockIdx.x, tag);
    __syncthreads();
    {
        auto *committed = flags + ControlRelayWord;
        if (blockIdx.x == 0) {
            // No participant can enter the next phase until this relay commits.
            // Exact arrival tags therefore cannot be overwritten before the leader
            // observes them, independently of the previous launch's grid size.
            for (int peer = threadIdx.x; peer < int(gridDim.x); peer += blockDim.x)
                while (load_relaxed<Scope::system>(flags + peer) != tag)
                    __builtin_amdgcn_s_sleep(4);
            __syncthreads();
            if (threadIdx.x == 0)
                store_relaxed<Scope::system>(committed, tag);
        } else if (threadIdx.x == 0) {
            while (load_relaxed<Scope::system>(committed) != tag)
                __builtin_amdgcn_s_sleep(4);
        }
        __syncthreads();
        // Retain the original payload acquire on every reader, including the
        // device-only acquire after the completed peer-return barrier.
        if constexpr (SystemAcquire)
            fence_acquire<Scope::system>();
        else
            fence_acquire<Scope::device>();
        __syncthreads();
        return;
    }
    for (int peer = threadIdx.x; peer < int(gridDim.x); peer += blockDim.x) {
        uint32_t observed;
        do {
            observed = load_relaxed<Scope::system>(flags + peer);
            if (!observed || int32_t(observed - tag) < 0)
                __builtin_amdgcn_s_sleep(4);
        } while (!observed || int32_t(observed - tag) < 0);
    }
    __syncthreads();
    if constexpr (SystemAcquire)
        fence_acquire<Scope::system>();
    else
        fence_acquire<Scope::device>();
    __syncthreads();
}

__device__ __forceinline__ uint64_t uniform_row_pointer(const void *p) {
    const uint64_t a = reinterpret_cast<uint64_t>(p);
    // readfirstlane returns signed int even when its argument is unsigned.
    // Truncate each returned word before widening; sign-extending the low
    // word would overwrite the high address bits whenever low bit 31 is set.
    const uint32_t low = __builtin_amdgcn_readfirstlane(uint32_t(a));
    const uint32_t high = __builtin_amdgcn_readfirstlane(uint32_t(a >> 32));
    return uint64_t(low) | uint64_t(high) << 32;
}

template <int H, int Batch>
__device__ __forceinline__ void gather_batched_rows(const MegaMoeKernelArgs &q, const MegaMoeRowBlock &b,
                                                    const RowTable &table, int *scratch) {
    static_assert(Batch == 2 || Batch == 4);
    const auto &p = q.moe;
    const auto &l = p.layout;
    constexpr int Words = H / 128, Packets = (H + 1023) / 1024, Waves = 4;
    const int lane = threadIdx.x % 64, wave = threadIdx.x / 64;
    const int table_words = 3 * l.num_ranks * l.num_experts_per_rank + l.num_ranks + 2 * l.num_experts_per_rank + 2;
    int *cache = scratch + ((table_words + 7) / 8 * 8);
    const uint64_t argument_address = uint64_t(__builtin_amdgcn_kernarg_segment_ptr());
    const auto *arguments =
        reinterpret_cast<const __attribute__((address_space(4))) MegaMoeKernelArgs *>(argument_address);
    if (threadIdx.x < M) {
        const int row = b.m0 + threadIdx.x;
        v4u info{}, destination{};
        if (row < b.end) {
            const int relative = row - table.starts[b.expert], experts = l.num_experts_per_rank;
            int source = 0;
            while (relative >= table.offsets[source * experts + b.expert] + table.counts[source * experts + b.expert])
                ++source;
            // Sources differ across preparation lanes. Read the immutable peer
            // offset directly from kernargs, without materializing a private view.
            const int64_t peer_offset = arguments->moe.view.offsets[source];
            const auto *local_list = region<const int>(p, l.send_lists) +
                                     int64_t(p.view.rank * experts + b.expert) * l.num_max_tokens * l.num_topk +
                                     relative - table.offsets[source * experts + b.expert];
            const int entry = *reinterpret_cast<const int *>(reinterpret_cast<int64_t>(local_list) + peer_offset);
            const int token = entry / l.num_topk, slot = entry % l.num_topk;
            const auto *local_weight = region<const float>(p, l.topk_weights) + entry;
            const uint32_t weight = uint32_t(__float_as_int(
                *reinterpret_cast<const float *>(reinterpret_cast<int64_t>(local_weight) + peer_offset)));
            const uint32_t code = (uint32_t(source) * l.num_max_tokens + token) * l.num_topk + slot;
            auto *local_output = region<uint16_t>(p, l.combine) + (int64_t(slot) * l.num_max_tokens + token) * H;
            const uint64_t address = uint64_t(reinterpret_cast<int64_t>(local_output) + peer_offset);
            info = v4u{uint32_t(source), uint32_t(token), code, weight};
            destination = v4u{uint32_t(address), uint32_t(address >> 32), 0, 0};
        }
        *reinterpret_cast<v4u *>(cache + threadIdx.x * 8) = info;
        *reinterpret_cast<v4u *>(cache + threadIdx.x * 8 + 4) = destination;
    }
    asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)\n\ts_barrier" ::: "memory");
    auto copy = [&](auto Full) {
        constexpr bool full = decltype(Full)::value;
        for (int base = 0; base < M; base += Waves * Batch) {
            v4i values[Batch][Packets];
            uint32_t scales[Batch];
            static_for<0, Batch>([&](auto J) {
                constexpr int j = decltype(J)::value;
                const int relative = base + wave + j * Waves, row = b.m0 + relative;
                if (full || row < b.end) {
                    const auto *local = (const __attribute__((address_space(3))) v4u *)(cache + relative * 8);
                    const v4u info = *local;
                    const int source = __builtin_amdgcn_readfirstlane(info[0]),
                              token = __builtin_amdgcn_readfirstlane(info[1]);
                    const auto input = make_rsrc(reinterpret_cast<const void *>(uniform_row_pointer(p.view.map(
                                                     region<const uint8_t>(p, l.x) + int64_t(token) * H, source))),
                                                 H);
                    const auto sf = make_rsrc(reinterpret_cast<const void *>(uniform_row_pointer(p.view.map(
                                                  region<const int>(p, l.x_sf) + int64_t(token) * Words, source))),
                                              Words * 4);
                    static_for<0, Packets>([&](auto P) {
                        values[j][decltype(P)::value] =
                            __builtin_amdgcn_raw_buffer_load_b128(input, lane * 16, decltype(P)::value * 1024, 16);
                    });
                    scales[j] = __builtin_amdgcn_raw_buffer_load_b32(sf, lane * 4, 0, 0);
                }
            });
            __builtin_amdgcn_s_waitcnt(waitcnt_vm(0));
            static_for<0, Batch>([&](auto J) {
                constexpr int j = decltype(J)::value;
                const int relative = base + wave + j * Waves, row = b.m0 + relative;
                if (full || row < b.end) {
                    const auto *local = (const __attribute__((address_space(3))) v4u *)(cache + relative * 8);
                    const v4u info = local[0], destination = local[1];
                    const auto output = make_rsrc(reinterpret_cast<const void *>(uniform_row_pointer(
                                                      region<uint8_t>(p, l.l1_acts) + int64_t(row) * H)),
                                                  H);
                    static_for<0, Packets>([&](auto P) {
                        buffer_store16<(17)>(__builtin_bit_cast(v4u, values[j][decltype(P)::value]), output, lane * 16,
                                             decltype(P)::value * 1024);
                    });
                    if (lane < Words) {
                        auto *out = region<int>(p, l.l1_sf) + int64_t(lane) * l.sf_pitch + row;
                        const int value = __builtin_bit_cast(int32_t, scales[j]);
                        store_relaxed<Scope::device>(out, value);
                    }
                    if (lane < l.record_ints) {
                        const uint32_t value = lane < Words        ? scales[j]
                                               : lane == Words     ? info[3]
                                               : lane == Words + 1 ? info[2]
                                                                   : 0;
                        auto *out = region<int>(p, l.l1_records) + int64_t(row) * l.record_ints + lane;
                        store_relaxed<Scope::device>(out, __builtin_bit_cast(int32_t, value));
                    }
                    if (lane == 0) {
                        const uint64_t address = uint64_t(uint32_t(destination[0])) | uint64_t(uint32_t(destination[1]))
                                                                                          << 32;
                        auto *out = region<int64_t>(p, l.output_rows) + row;
                        store_relaxed<Scope::device>(out, __builtin_bit_cast(int64_t, address));
                    }
                }
            });
        }
    };
    if (b.end - b.m0 == M)
        copy(std::true_type{});
    else
        copy(std::false_type{});
}

template <int H> __device__ __forceinline__ void gather(const MegaMoeKernelArgs &q, int item) {
    const auto b = q.blocks[item];
    const auto &p = q.moe;
    const auto &l = p.layout;
    extern __shared__ int scratch[];
    const int lane = threadIdx.x % 64, words = l.hidden / 128;
    if (b.shared) {
        for (int row = b.m0 + threadIdx.x / 64; row < b.end; row += blockDim.x / 64)
            for (int w = lane; w < words; w += 64) {
                auto *out = region<int>(p, l.shared_l1_sf) + int64_t(w) * l.shared_sf_pitch + row;
                const int value = region<const int>(p, l.x_sf)[int64_t(row) * words + w];
                store_relaxed<Scope::device>(out, value);
            }
    } else {

        const RowTable table = build_row_table(p, scratch);

        gather_batched_rows<H, 2>(q, b, table, scratch);
    }
    // Every payload byte is explicitly coherent in this policy. Drain all
    // lanes, join the workgroup, then publish one coherent flag; readers keep
    // their coherent data loads. Ordinary cached stores retain the old fence.
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    __syncthreads();
    if (threadIdx.x == 0) {
        store_relaxed<Scope::device>(q.sf_ready + item, 1);
    }
    __syncthreads();
}

__device__ __forceinline__ void acquire_row_payload() {
    // These readers use coherent payload instructions after observing the
    // publication signal. Preserve compiler ordering without invalidating
    // read-only expert weights alongside the mutable activation cache lines.
    asm volatile("" ::: "memory");
}

} // namespace resident_moe
