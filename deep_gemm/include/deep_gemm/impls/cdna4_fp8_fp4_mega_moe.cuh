#pragma once
#include <deep_gemm/impls/cdna4_mega_moe_epilogue.cuh>
#include <deep_gemm/impls/cdna4_mega_moe_gemm.cuh>
#include <deep_gemm/layout/cdna4_mega_moe.cuh>
#include <deep_gemm/scheduler/cdna4_mega_moe.cuh>

namespace resident_moe {
__device__ __forceinline__ void record_phase(const MegaMoeKernelArgs &q, int phase) {
#if defined(DG_MEGA_MOE_PHASE_TIMESTAMPS)
    if (blockIdx.x == 0 && threadIdx.x == 0)
        reinterpret_cast<volatile uint64_t *>(q.k_ready)[256 + phase] = __builtin_readcyclecounter();
#endif
}
template <int K, int N, bool First, bool Shared, bool Paired = false, int PairStride = 0>
__device__ __forceinline__ void tile(const MegaMoeKernelArgs &q, const MegaMoeRowBlock &b, int column,
                                     const int *row_weights = nullptr) {
    static_assert(!Paired || (!Shared && PairStride > 0));
    // Reload only this stage from read-only kernel arguments. Keeping a mutable
    // copy of the complete launch struct forces private-memory materialization.
    uint64_t address = uint64_t(__builtin_amdgcn_kernarg_segment_ptr());
    asm volatile("s_mov_b64 %0, %0" : "+s"(address)::"memory");
    const auto *args = reinterpret_cast<const __attribute__((address_space(4))) MegaMoeKernelArgs *>(address);
    auto a = First ? (Shared ? args->shared_l1 : args->l1) : (Shared ? args->shared_l2 : args->l2);
    if constexpr (Shared && First)
        RESIDENT_S1(a);
    else if constexpr (Shared)
        RESIDENT_S2(a);
    else if constexpr (First)
        RESIDENT_L1(a);
    else
        RESIDENT_L2(a);
    extern __shared__ __attribute__((aligned(1024))) uint8_t smem[];
    constexpr int BN = Paired ? 512 : 256, Pitch = 2 * BN + 16;

    pipeline_native_tile<K, N, First, !Shared, !Shared && CDNA4_MOE_WEIGHT_PANELS, BN, Paired ? PairStride : 0>(
        a, b.expert, b.m0, b.end, column);

    if constexpr (First) {
        if constexpr (Paired) {
            static_for<0, 2>([&](auto Half) {
                constexpr int half = decltype(Half)::value;
#pragma unroll 1
                for (int round = 0; round < M / 32; ++round)
                    if (threadIdx.x < 256) {
                        const int r = threadIdx.x / 8, extra = 16 * round + (r / 16) * (M / 2 - 16);
                        pipeline_swiglu_epilogue<false, M / 2, (1), 256, (1)>(
                            a, smem + extra * Pitch + half * 512, Pitch, b.m0, b.end, column + half * PairStride * 256,
                            round, threadIdx.x, row_weights);
                    }
            });
        } else {
#pragma unroll 1
            for (int round = 0; round < M / (32); ++round)
                if (threadIdx.x < 256) {
                    const int r = threadIdx.x / 8, extra = 16 * round + (r / 16) * (M / 2 - 16);
                    pipeline_swiglu_epilogue<false, M / 2, (1), 256, (1)>(a, smem + extra * 528, 528, b.m0, b.end,
                                                                          column, round, threadIdx.x, row_weights);
                }
        }
    } else if constexpr (!First && !Shared) {
        constexpr int ColThreads = BN / 8, RowStride = 256 / ColThreads;
        const int row0 = threadIdx.x / ColThreads, col = (threadIdx.x % ColThreads) * 8;
        {
            constexpr int Batch = (2);
            constexpr int Chunks = (M + RowStride - 1) / RowStride;
            const int logical_col = Paired ? col % 256 + (col / 256) * PairStride * 256 : col;
#pragma unroll 1
            for (int base = 0; base < Chunks; base += Batch) {
                int64_t addresses[Batch];
                v4u values[Batch];
                static_for<0, Batch>([&](auto J) {
                    constexpr int j = decltype(J)::value;
                    const int relative = row0 + (base + j) * RowStride, row = b.m0 + relative;
                    if (base + j < Chunks && row < b.end) {
                        addresses[j] = load_relaxed<Scope::device>(a.moe_output_rows + row);
                        values[j] = *reinterpret_cast<const v4u *>(smem + relative * Pitch + col * 2);
                    }
                });
                // Keep pointer reads ahead of this small store batch. The final
                // per-tile VMEM drain and writer/peer joins remain unchanged.
                __builtin_amdgcn_s_waitcnt(waitcnt_vm(0));
                static_for<0, Batch>([&](auto J) {
                    constexpr int j = decltype(J)::value;
                    const int row = b.m0 + row0 + (base + j) * RowStride;
                    if (base + j < Chunks && row < b.end) {
                        const uint64_t pointer = reinterpret_cast<uint64_t>(reinterpret_cast<uint16_t *>(addresses[j]) +
                                                                            column + logical_col);
                        const v4u value = values[j];
                        asm volatile("global_store_dwordx4 %0, %1, off sc0 sc1\n\ts_nop 0" ::"v"(pointer), "v"(value)
                                     : "memory");
                    }
                });
            }
        }
    }
    asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)\n\ts_barrier" ::: "memory");
}

template <int K, int N, bool First, int Groups>
__device__ __forceinline__ void row_columns(const MegaMoeKernelArgs &q, const MegaMoeRowBlock &b, int item, int part,
                                            int columns) {
    const int *row_weights = nullptr;
    if constexpr (First) {
        // Refill after this row's gather publication on every frame/task. This
        // separate static LDS region must not alias recyclable GEMM operands.
        // Each column reuses the same unchanged FP32 routing-weight bit pattern;
        // pointwise readers need only LDS readiness, not another VMEM load/wait.
        __shared__ int staged_weights[M];
        if (threadIdx.x < M) {
            const int row = b.m0 + threadIdx.x;
            int word = __float_as_int(1.f);
            if (!b.shared && row < b.end)
                word = load_relaxed<Scope::device>(region<const int>(q.moe, q.moe.layout.l1_records) +
                                                   int64_t(row) * q.moe.layout.record_ints + q.moe.layout.hidden / 128);
            staged_weights[threadIdx.x] = word;
        }
        __syncthreads();
        row_weights = staged_weights;
    }

    {

        {
            if (!b.shared) {
                for (int column = part; column < columns; column += 2 * Groups) {
                    const bool paired = column + Groups < columns;
                    if (paired)
                        tile<K, N, First, false, true, Groups>(q, b, column * 256, row_weights);
                    else
                        tile<K, N, First, false>(q, b, column * 256, row_weights);
                }
                return;
            }
        }
        for (int column = part; column < columns; column += Groups) {

            if (b.shared)
                tile<K, N, First, true>(q, b, column * (256), row_weights);
            else
                tile<K, N, First, false>(q, b, column * (256), row_weights);
        }
    }
}

template <int Threads, int Batch>
__device__ __forceinline__ void route_prefetched_expert(const MegaMoeArgs &p, int expert) {
    static_assert(Batch == 2 || Batch == 4 || Batch == 8);
    const auto &l = p.layout;
    const int tid = threadIdx.x, lane = tid % 64, wave = tid / 64;
    const int entries = p.num_tokens * l.num_topk;
    const auto *indices = region<const int64_t>(p, l.topk_idx);
    __shared__ int wave_counts[Threads / 64];
    int total = 0;
    for (int first = 0; first < entries; first += Threads * Batch) {
        // Read ahead without modifying entry order or retaining state across
        // frames. The per-tile wave count joins remain identical to stable route.
        int64_t keys[Batch];
        static_for<0, Batch>([&](auto J) {
            constexpr int j = decltype(J)::value;
            const int entry = first + j * Threads + tid;
            keys[j] = entry < entries ? indices[entry] : int64_t(-1);
        });
        static_for<0, Batch>([&](auto J) {
            constexpr int j = decltype(J)::value;
            const int entry = first + j * Threads + tid;
            const bool selected = keys[j] == expert;
            const uint64_t mask = __ballot(selected);
            const uint64_t preceding = lane == 0 ? 0ull : (uint64_t(1) << lane) - 1ull;
            if (lane == 0)
                wave_counts[wave] = __popcll(mask);
            __syncthreads();
            int before_wave = 0, count = 0;
            for (int w = 0; w < Threads / 64; ++w) {
                before_wave += w < wave ? wave_counts[w] : 0;
                count += wave_counts[w];
            }
            if (selected) {
                const int position = total + before_wave + __popcll(mask & preceding);
                region<int>(p, l.positions)[entry] = position;
                if (p.ordered_push)
                    region<int>(p, l.send_lists)[int64_t(expert) * l.num_max_tokens * l.num_topk + position] = entry;
            }
            __syncthreads();
            total += count;
        });
    }
    if (tid == 0)
        region<int>(p, l.send_counts)[expert] = total;
}

template <int H, int TopK, int ColumnGroups>
__device__ __forceinline__ void run_gather_actor(const MegaMoeKernelArgs &q, int *job, uint32_t epoch) {
    const auto &p = q.moe;
    // These actors read inputs and route lists produced by peer GPUs. The
    // local grid joins do not replace the required remote-input acquire.

    for (;;) {
        if (threadIdx.x == 0)
            job[0] = atomic_add<Scope::device>(q.state + PullHead, 1);
        __syncthreads();
        const int index = __builtin_amdgcn_readfirstlane(job[0]);
        if (index >= q.state[BlockCount])
            break;

        gather<H>(q, index);
    }
    // In the metadata-first policy, make row maps available before this
    // actor starts sending its source packets. Matrix consumers retain the
    // per-packet ready waits, so early metadata is not early payload access.
}

template <int H, int I, int TopK, int ColumnGroups = 6>
__device__ __forceinline__ void run(const MegaMoeKernelArgs &q) {
    const auto &p = q.moe;
    record_phase(q, 0);

    const uint32_t epoch = load_relaxed<Scope::system>(region<const uint32_t>(p, p.layout.barrier)) * (8u);
    if (blockIdx.x == 0 && threadIdx.x == 0)
        store_relaxed<Scope::system>(reinterpret_cast<uint32_t *>(q.k_ready) + ControlRelayWord, uint32_t(0));
    // The unchanged first all-writer join completes this reset before any
    // control relay is consumed. Re-enabling a relay after arbitrary intervening
    // launches cannot observe its stale commit, including an epoch wrap.

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < StateSize + 2 * q.max_blocks + 2 * q.max_blocks * q.n2;
         i += gridDim.x * blockDim.x)
        q.state[i] = 0;

    for (int e = blockIdx.x; e < p.layout.num_experts; e += gridDim.x) {
        route_prefetched_expert<256, 8>(p, e);
        __syncthreads();
    }
    // Route stores on every CU must precede the system release in the leader.

    // Only the metadata producer needs route counts at this point. All other
    // workgroups wait at the task-table publication below, avoiding an
    // all-to-all scan of the same arrival flags.
    fence_release<Scope::system>();
    __syncthreads();
    auto *route_flags = reinterpret_cast<uint32_t *>(q.k_ready);
    if (threadIdx.x == 0)
        store_release<Scope::system>(route_flags + blockIdx.x, epoch + 1u);
    __syncthreads();
    if (blockIdx.x == 0) {
        for (int peer = threadIdx.x; peer < int(gridDim.x); peer += blockDim.x) {
            uint32_t observed;
            do {
                observed = load_relaxed<Scope::system>(route_flags + peer);
                if (!observed || int32_t(observed - (epoch + 1u)) < 0)
                    __builtin_amdgcn_s_sleep(4);
            } while (!observed || int32_t(observed - (epoch + 1u)) < 0);
        }
        __syncthreads();
        fence_acquire<Scope::system>();
        __syncthreads();
    }
    record_phase(q, 1);

    if (blockIdx.x == 0) {
        mega_moe_barrier_impl<true>(p);
        // Only this producer consumes the count metadata before publishing
        // the task table. Join its writers locally; readers wait at tag 3.
        fence_release<Scope::system>();
        asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)\n\ts_barrier" ::: "memory");
        fence_acquire<Scope::system>();
    }
    record_phase(q, 2);

    if (blockIdx.x == 0) {
        const int shared_blocks = (q.shared_tokens + M - 1) / M;
        for (int i = threadIdx.x; i < shared_blocks; i += blockDim.x)
            q.blocks[i] = {0, i * M, min((i + 1) * M, q.shared_tokens), 1};
        const auto *ends = region<const int>(p, p.layout.psum_ends);
        if (p.layout.num_experts_per_rank <= 64) {
            if (threadIdx.x < 64) {
                const int expert = threadIdx.x;
                const bool valid = expert < p.layout.num_experts_per_rank;
                const int start = valid ? expert_start(ends, expert, p.layout.alignment) : 0;
                const int end = valid ? ends[expert] : 0;
                const int count = (end - start + M - 1) / M;
                int inclusive = count;
#pragma unroll
                for (int offset = 1; offset < 64; offset *= 2) {
                    const int previous = __shfl_up(inclusive, offset, 64);
                    if (expert >= offset)
                        inclusive += previous;
                }
                const int base = shared_blocks + inclusive - count;
                for (int i = 0; i < count; ++i)
                    q.blocks[base + i] = {expert, start + i * M, min(start + (i + 1) * M, end), 0};
                const int total = __shfl(inclusive, 63, 64);
                if (expert == 0)
                    q.state[BlockCount] = shared_blocks + total;
            }
        } else if (threadIdx.x == 0) {
            int count = shared_blocks;
            for (int expert = 0; expert < p.layout.num_experts_per_rank; ++expert) {
                const int start = expert_start(ends, expert, p.layout.alignment), end = ends[expert];
                for (int row = start; row < end; row += M)
                    q.blocks[count++] = {expert, row, min(row + M, end), 0};
            }
            q.state[BlockCount] = count;
        }
    }
    grid_sync_control(q, epoch + 3u);
    record_phase(q, 3);

    __shared__ int job[4];
    if (blockIdx.x < q.dispatch_blocks)
        run_gather_actor<H, TopK, ColumnGroups>(q, job, epoch);
    {
        // Coarse M-tile tasks keep both GEMMs on the same workgroup. Different
        // row tiles still overlap gather, first GEMM, second GEMM and peer return,
        // but avoid a global atomic queue operation for every 256-column slice.

        for (;;) {
            if (threadIdx.x == 0) {
                {
                    job[0] = atomic_add<Scope::device>(q.state + FirstHead, 1);
                }
            }
            __syncthreads();
            int item, part;
            {
                const int ticket = __builtin_amdgcn_readfirstlane(job[0]);

                {
                    item = ticket < 0 ? -1 : (ticket / ColumnGroups);
                    part = (ticket % ColumnGroups);
                }
            }
            if (item < 0 || item >= q.state[BlockCount])
                break;

            if (threadIdx.x == 0)
                while (!load_relaxed<Scope::device>(q.sf_ready + item))
                    __builtin_amdgcn_s_sleep(16);
            __syncthreads();
            acquire_row_payload();
            __syncthreads();

            const auto b = q.blocks[item];

            const int first_columns = q.n1 * (1);
            row_columns<H, 2 * I, true, ColumnGroups>(q, b, item, part, first_columns);
            // Each native tile already drains every coherent payload writer and
            // joins its waves. No ordinary payload store is allowed in this policy.

            __syncthreads();
            if constexpr (ColumnGroups > 1) {
                if (threadIdx.x == 0) {
                    atomic_add<Scope::device>(q.l1_done + item, 1);
                    while (load_relaxed<Scope::device>(q.l1_done + item) < ColumnGroups)
                        __builtin_amdgcn_s_sleep(4);
                }
                __syncthreads();
                acquire_row_payload();
                __syncthreads();
            }

            const int second_columns = q.n2 * (1);
            row_columns<I, H, false, ColumnGroups>(q, b, item, part, second_columns);

            // In the deferred policy no consumer observes a row-level completion
            // signal. Every writer's final completion path still performs a system
            // release before publication; the leader waits for *all* such writers
            // before the peer barrier and reduction. Never use this policy with an
            // asynchronous row consumer or a dedicated local-to-peer return actor.
            __syncthreads();
        }
    }
    // Every peer-return writer releases its stores before the local join. Only
    // after that join may the leader publish cross-rank completion; reduction
    // retains coherent system-scope payload reads after the peer barrier.

    constexpr unsigned final_offset = 0;

    // All coherent result stores must complete before this workgroup arrives.
    // The last arriving workgroup performs the peer barrier; other workgroups
    // wait on one committed epoch instead of scanning every writer's flag.
    asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)\n\ts_barrier" ::: "memory");
    if (threadIdx.x == 0)
        job[0] = atomic_add_release<Scope::system>(q.state + SecondDone, 1);
    __syncthreads();
    auto *committed = reinterpret_cast<uint32_t *>(q.k_ready) + ControlRelayWord;
    const uint32_t completed_epoch = epoch + 5u + final_offset;
    if (job[0] == int(gridDim.x) - 1) {
        fence_acquire<Scope::system>();
        __syncthreads();
        mega_moe_barrier_impl<false>(p);
        fence_release<Scope::system>();
        __syncthreads();
        if (threadIdx.x == 0)
            store_release<Scope::system>(committed, completed_epoch);
    } else if (threadIdx.x == 0) {
        while (load_relaxed<Scope::system>(committed) != completed_epoch)
            __builtin_amdgcn_s_sleep(4);
    }
    __syncthreads();
    fence_acquire<Scope::system>();
    __syncthreads();
    // These two diagnostic markers now both follow the combined completion.
    record_phase(q, 4);
    record_phase(q, 5);

    moe_pointwise::reduce<H, TopK>(p);
    record_phase(q, 6);

    // Prevent an early-finishing rank from reusing inputs while peers are still
    // reading this call. The next launch's counts barrier supplies this ordering.
}
} // namespace resident_moe
