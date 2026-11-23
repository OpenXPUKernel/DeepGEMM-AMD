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

// Route has already joined every producer before the leader built the row
// table. With no other intervening producer, only a publication is needed.
__device__ __forceinline__ void publish_task_table(const MegaMoeKernelArgs &q, uint32_t tag) {
    auto *ready = reinterpret_cast<uint32_t *>(q.k_ready) + ControlRelayWord;
    if (blockIdx.x == 0) {
        fence_release<Scope::system>();
        __syncthreads();
        if (threadIdx.x == 0)
            store_release<Scope::system>(ready, tag);
    } else if (threadIdx.x == 0) {
        while (load_relaxed<Scope::system>(ready) != tag)
            __builtin_amdgcn_s_sleep(4);
    }
    __syncthreads();
    fence_acquire<Scope::system>();
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
__device__ __forceinline__ void gather_batched_rows(
    const MegaMoeKernelArgs &q, const MegaMoeRowBlock &b, const RowTable &table, int *scratch, bool direct_input, bool mixed_input) {
    static_assert(Batch == 2 || Batch == 4);
    const auto &p = q.moe;
    const auto &l = p.layout;
    constexpr int Words = H / 128, Packets = (H + 1023) / 1024, Waves = 4;
    const int lane = threadIdx.x % 64, wave = threadIdx.x / 64;
    const int table_words = 3 * l.num_ranks * l.num_experts_per_rank + l.num_ranks + 2 * l.num_experts_per_rank + 2;
    int *cache = scratch + ((table_words + 7) / 8 * 8);
#ifdef CDNA4_GATHER_EARLY_ADDRESS
    constexpr int PacketWords = 4;
#else
    constexpr int PacketWords = 8;
#endif
#ifdef CDNA4_GATHER_SCALE_ROWS
    int *staged_scales = cache + M * PacketWords;
#endif
    const uint64_t argument_address = uint64_t(__builtin_amdgcn_kernarg_segment_ptr());
    const auto *arguments =
        reinterpret_cast<const __attribute__((address_space(4))) MegaMoeKernelArgs *>(argument_address);
#ifdef CDNA4_MOE_BOUND_GATHER_ROWS
    const int rows = b.end - b.m0;
#else
    constexpr int rows = M;
#endif
    if (threadIdx.x < rows) {
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
#ifdef CDNA4_GATHER_EARLY_ADDRESS
            store_relaxed<Scope::device>(region<int64_t>(p, l.output_rows) + row, __builtin_bit_cast(int64_t, address));
#else
            destination = v4u{uint32_t(address), uint32_t(address >> 32), 0, 0};
#endif
        }
        *reinterpret_cast<v4u *>(cache + threadIdx.x * PacketWords) = info;
#ifndef CDNA4_GATHER_EARLY_ADDRESS
        *reinterpret_cast<v4u *>(cache + threadIdx.x * PacketWords + 4) = destination;
#endif
    }
    asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)\n\ts_barrier" ::: "memory");
    auto copy = [&](auto Full) {
        constexpr bool full = decltype(Full)::value;
        for (int base = 0; base < (full ? M : rows); base += Waves * Batch) {
            v4i values[Batch][Packets];
            uint32_t scales[Batch];
            static_for<0, Batch>([&](auto J) {
                constexpr int j = decltype(J)::value;
                const int relative = base + wave + j * Waves, row = b.m0 + relative;
                if (full || row < b.end) {
                    const auto *local = (const __attribute__((address_space(3))) v4u *)(cache + relative * PacketWords);
                    const v4u info = *local;
                    const int source = __builtin_amdgcn_readfirstlane(info[0]),
                              token = __builtin_amdgcn_readfirstlane(info[1]);
                    const auto input = make_rsrc(reinterpret_cast<const void *>(uniform_row_pointer(p.view.map(
                                                     region<const uint8_t>(p, l.x) + int64_t(token) * H, source))),
                                                 H);
                    const auto sf = make_rsrc(reinterpret_cast<const void *>(uniform_row_pointer(p.view.map(
                                                  region<const int>(p, l.x_sf) + int64_t(token) * Words, source))),
                                              Words * 4);
                    bool copy_row = !direct_input;
#ifdef CDNA4_MOE_MIXED_LOCAL_ROWS
                    copy_row &= !mixed_input || __builtin_amdgcn_readfirstlane(info[0]) != p.view.rank;
#endif
                    if (copy_row) {
                        static_for<0, Packets>([&](auto P) {
                            values[j][decltype(P)::value] =
                                __builtin_amdgcn_raw_buffer_load_b128(input, lane * 16, decltype(P)::value * 1024, 16);
                        });
                    }
                    scales[j] = __builtin_amdgcn_raw_buffer_load_b32(sf, lane * 4, 0, 0);
                }
            });
            __builtin_amdgcn_s_waitcnt(waitcnt_vm(0));
            static_for<0, Batch>([&](auto J) {
                constexpr int j = decltype(J)::value;
                const int relative = base + wave + j * Waves, row = b.m0 + relative;
                if (full || row < b.end) {
                    const auto *local = (const __attribute__((address_space(3))) v4u *)(cache + relative * PacketWords);
                    const v4u info = local[0];
#ifndef CDNA4_GATHER_EARLY_ADDRESS
                    const v4u destination = local[1];
#endif
                    const auto output = make_rsrc(reinterpret_cast<const void *>(uniform_row_pointer(
                                                      region<uint8_t>(p, l.l1_acts) + int64_t(row) * H)),
                                                  H);
                    bool copy_row = !direct_input;
#ifdef CDNA4_MOE_MIXED_LOCAL_ROWS
                    copy_row &= !mixed_input || __builtin_amdgcn_readfirstlane(info[0]) != p.view.rank;
#endif
                    if (copy_row) {
                        static_for<0, Packets>([&](auto P) {
                            buffer_store16<(17)>(
                                __builtin_bit_cast(v4u, values[j][decltype(P)::value]), output, lane * 16, decltype(P)::value * 1024);
                        });
                    }
                    if (lane < Words) {
                        const int value = __builtin_bit_cast(int32_t, scales[j]);
#ifdef CDNA4_GATHER_SCALE_ROWS
                        staged_scales[relative * (Words + 1) + lane] = value;
#else
                        auto *out = region<int>(p, l.l1_sf) + int64_t(lane) * l.sf_pitch + row;
                        store_relaxed<Scope::device>(out, value);
#endif
                    }
                    if (lane < l.record_ints) {
                        const uint32_t value = lane < Words        ? scales[j]
                                               : lane == Words     ? info[3]
                                               : lane == Words + 1 ? info[2]
                                                                   : 0;
                        auto *out = region<int>(p, l.l1_records) + int64_t(row) * l.record_ints + lane;
                        store_relaxed<Scope::device>(out, __builtin_bit_cast(int32_t, value));
                    }
#ifndef CDNA4_GATHER_EARLY_ADDRESS
                    if (lane == 0) {
                        const uint64_t address = uint64_t(uint32_t(destination[0])) | uint64_t(uint32_t(destination[1]))
                                                                                          << 32;
                        auto *out = region<int64_t>(p, l.output_rows) + row;
                        store_relaxed<Scope::device>(out, __builtin_bit_cast(int64_t, address));
                    }
#endif
                }
            });
        }
    };
    if (b.end - b.m0 == M)
        copy(std::true_type{});
    else
        copy(std::false_type{});
#ifdef CDNA4_GATHER_SCALE_ROWS
    // Remote scale loads stay contiguous by token. After all row writers
    // arrive, store each scale word across a contiguous group of local rows.
    asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
    constexpr int LaneRows = CDNA4_GATHER_SCALE_ROWS;
    for (int r = threadIdx.x % LaneRows; r < b.end - b.m0; r += LaneRows) {
        for (int word = threadIdx.x / LaneRows; word < Words; word += 256 / LaneRows) {
            auto *dst = region<int>(p, l.l1_sf) + int64_t(word) * l.sf_pitch + b.m0 + r;
            store_relaxed<Scope::device>(dst, staged_scales[r * (Words + 1) + word]);
        }
    }
#endif
}

// The resident gather reads only the counts of experts owned by this rank.
// A source sends each count to that expert's owner. Sender-side dst_rows
// are scratch for push dispatch and are unused by this receiver-owned path.
__device__ __forceinline__ void publish_receiver_counts(const MegaMoeArgs &p) {
    const auto &l = p.layout;
    const int experts = l.num_experts, local = l.num_experts_per_rank, rank = p.view.rank;
    const int tid = threadIdx.x, wave = tid / 64, lane = tid % 64;
    auto *counts = region<int>(p, l.counts);
#ifndef CDNA4_MOE_DIRECT_RECEIVER_COUNTS
    auto *send = region<int>(p, l.send_counts);
    for (int e = tid; e < experts; e += blockDim.x)
        store_relaxed<Scope::system>(p.view.map(counts + rank * experts + e, e / local), send[e]);
#endif

    compact_barrier_workgroup(
        p.view, region<uint32_t>(p, l.barrier + p.barrier_slot * 128), region<int>(p, l.barrier + p.barrier_slot * 128 + 64), tid);
    int total = 0;
    if (tid < local)
        for (int src = 0; src < l.num_ranks; ++src)
            total += load_relaxed<Scope::system>(counts + src * experts + rank * local + tid);
    const int aligned = (total + l.alignment - 1) / l.alignment * l.alignment;
    int prefix = aligned;
#pragma unroll
    for (int offset = 1; offset < 64; offset *= 2) {
        const int previous = __shfl_up(prefix, offset, 64);
        if (lane >= offset)
            prefix += previous;
    }
    __shared__ int wave_rows[4];
    if (lane == 63)
        wave_rows[wave] = prefix;
    __syncthreads();
    if (tid < local) {
        int start = prefix - aligned;
        for (int w = 0; w < wave; ++w)
            start += wave_rows[w];
        region<int>(p, l.psum_ends)[tid] = start + total;
        if (p.stats)
            p.stats[tid] += total;
    }
#ifndef CDNA4_MOE_DIRECT_RECEIVER_COUNTS
    for (int e = tid; e < experts; e += blockDim.x)
        send[e] = 0;
#endif
}

// The counts and expert starts are immutable until the next launch. Each
// gather actor retains its own copy below the per-block row packet scratch.
__device__ __forceinline__ RowTable cache_gather_table(const MegaMoeArgs &p, int *scratch) {
    const auto &l = p.layout;
    RowTable table{};
    const int experts = l.num_experts_per_rank, ranks = l.num_ranks;
    table.counts = scratch;
    table.offsets = table.counts + ranks * experts;
    table.starts = table.offsets + ranks * experts;
    const int *counts = region<const int>(p, l.counts) + p.view.rank * experts;
    for (int i = threadIdx.x; i < ranks * experts; i += blockDim.x)
        table.counts[i] = load_relaxed<Scope::system>(counts + (i / experts) * l.num_experts + i % experts);
    __syncthreads();
    for (int e = threadIdx.x; e < experts; e += blockDim.x) {
        int offset = 0;
        for (int rank = 0; rank < ranks; ++rank) {
            const int i = rank * experts + e;
            table.offsets[i] = offset;
            offset += table.counts[i];
        }
        table.starts[e] = expert_start(region<const int>(p, l.psum_ends), e, l.alignment);
    }
    __syncthreads();
    return table;
}

__device__ __forceinline__ void prepare_shared_scales(const MegaMoeArgs &p, const MegaMoeRowBlock &b) {
    const auto &l = p.layout;
    const int lane = threadIdx.x % 64, words = l.hidden / 128;
    for (int row = b.m0 + threadIdx.x / 64; row < b.end; row += blockDim.x / 64)
        for (int w = lane; w < words; w += 64) {
            auto *out = region<int>(p, l.shared_l1_sf) + int64_t(w) * l.shared_sf_pitch + row;
            const int value = region<const int>(p, l.x_sf)[int64_t(row) * words + w];
            store_relaxed<Scope::device>(out, value);
        }
}
#ifndef CDNA4_MOE_GATHER_SHARDS
#define CDNA4_MOE_GATHER_SHARDS 1
#endif
template <int H> __device__ __forceinline__ void gather(const MegaMoeKernelArgs &q, int item, int shard, const RowTable &cached) {
    auto b = q.blocks[item];
#ifdef CDNA4_MOE_EAGER_LOCAL
    if (!b.shared && load_relaxed<Scope::device>(q.jobs + item) > 0)
        return;
#endif
    if constexpr (CDNA4_MOE_GATHER_SHARDS > 1) {
        const int rows = (b.end - b.m0 + CDNA4_MOE_GATHER_SHARDS - 1) / CDNA4_MOE_GATHER_SHARDS;
        b.m0 = min(b.end, b.m0 + rows * shard);
        b.end = min(b.end, b.m0 + rows);
    }
    const auto &p = q.moe;
    const auto &l = p.layout;
    extern __shared__ int scratch[];
    if (b.shared) {
        prepare_shared_scales(p, b);
    } else {

#ifdef CDNA4_MOE_CACHE_GATHER_TABLE
        const RowTable &table = cached;
#else
        const RowTable table = build_row_table(p, scratch);
#endif

        bool direct = false, mixed = false;
#ifdef CDNA4_MOE_LOCAL_INPUT
        const auto whole = q.blocks[item];
        const int bucket = p.view.rank * l.num_experts_per_rank + b.expert;
        const int start = table.starts[b.expert] + table.offsets[bucket];
        direct = whole.m0 >= start && whole.end <= start + table.counts[bucket];
#ifdef CDNA4_MOE_MIXED_LOCAL_ROWS
        const int local_count = min(whole.end, start + table.counts[bucket]) - max(whole.m0, start);
        mixed = !direct && local_count >= CDNA4_MOE_MIXED_LOCAL_ROWS;
#endif
        if (shard == 0 && threadIdx.x == 0)
            store_relaxed<Scope::device>(q.jobs + item, direct ? whole.m0 - start + 1 : (mixed ? -1 : 0));
#endif
        gather_batched_rows<H, 2>(q, b, table, scratch, direct, mixed);
    }
    // Every payload byte is explicitly coherent in this policy. Drain all
    // lanes, join the workgroup, then publish one coherent flag; readers keep
    // their coherent data loads. Ordinary cached stores retain the old fence.
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    __syncthreads();
    if (threadIdx.x == 0) {
        if constexpr (CDNA4_MOE_GATHER_SHARDS > 1)
            atomic_add<Scope::device>(q.sf_ready + item, 1);
        else
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
