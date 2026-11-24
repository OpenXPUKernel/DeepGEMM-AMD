#pragma once
#include <deep_gemm/impls/cdna4_mega_moe_epilogue.cuh>
#include <deep_gemm/impls/cdna4_mega_moe_gemm.cuh>
#include <deep_gemm/layout/cdna4_mega_moe.cuh>
#include <deep_gemm/scheduler/cdna4_mega_moe.cuh>

namespace resident_moe {
// Only the first lane of the same resident workgroup timestamps a layer.
// The per-layer ring is allocated before graph capture and reset outside it.
__device__ __forceinline__ void online_tick(const MegaMoeKernelArgs &q, int field, bool elected_writer = false) {
#ifdef CDNA4_ONLINE_MOE_TIMING
    if ((!blockIdx.x || elected_writer) && !threadIdx.x) {
        auto *ring = reinterpret_cast<volatile uint64_t *>(q.trace);
        const int64_t position =
            q.position64 ? *static_cast<const int64_t *>(q.first_position) : *static_cast<const int32_t *>(q.first_position);
#ifdef CDNA4_TRACE_MIN_POSITION
        // The same captured graph can serve later short requests. Filter at
        // recording time so those invocations cannot evict a measured frame.
        if (position < CDNA4_TRACE_MIN_POSITION || position > CDNA4_TRACE_MAX_POSITION)
            return;
#endif
#ifdef CDNA4_TRACE_PEER_JOIN
        auto *sequence = reinterpret_cast<unsigned long long *>(q.trace);
        const uint64_t seq = field == 2 ? 1 + __hip_atomic_fetch_add(sequence, 1ull, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT)
                                        : __hip_atomic_load(sequence, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
#else
        uint64_t seq = ring[0];
        if (field == 2)
            ring[0] = ++seq;
#endif
        const int base = 1 + int((seq - 1) & 255) * 9;
        ring[base + field] = __builtin_amdgcn_s_memrealtime();
        if (field == 2) {
            ring[base] = seq;
            ring[base + 1] = uint64_t(position);
        }
        if (field == 7)
            ring[base + 8] = q.state[BlockCount];
    }
#endif
}
__device__ __forceinline__ void record_phase(const MegaMoeKernelArgs &q, int phase) {
#if defined(DG_MEGA_MOE_PHASE_TIMESTAMPS)
    if (blockIdx.x == 0 && threadIdx.x == 0)
        reinterpret_cast<volatile uint64_t *>(q.k_ready)[256 + phase] = __builtin_readcyclecounter();
#endif
}
// With between one and two L1 panels per worker, the short tasks can
// produce a contiguous K prefix for L2 before the paired tasks finish.
template <int Columns, int Groups> constexpr int early_k_panels() {
    constexpr int singles = 2 * Groups - Columns;
    return Columns > Groups && singles >= 3 ? singles : 0;
}
#ifdef CDNA4_UP_WEIGHT_AHEAD
// Match the packed FP4 matrix operand ring exactly. Only immutable weights and
// exponents are touched here; source indices, weights and activations still
// require the ordinary gather publication before the first projection.
template <int K, int Stride>
__device__ __forceinline__ void prefetch_up_weights(const MegaMoeKernelArgs &q, const MegaMoeRowBlock &b, int part) {
    constexpr int BN = Stride ? 512 : 256, Span = Stride ? (Stride + 1) * 256 : 256, Step = 128 * 128 + BN * 64 + 1024 + BN * 4;
    constexpr int BO = 128 * 128, SBO = BO + BN * 64 + 1024;
    const auto &a = q.l1;
    const int tid = threadIdx.x, wave = tid / 64, lane = tid % 64, br = 16 * wave + lane / 4;
    const int n0 = part * 256, bv = br * 64 + ((lane % 4) ^ ((br >> 2) & 3)) * 16;
    auto rb = make_rsrc(a.B + int64_t(b.expert) * a.b_group_stride + int64_t(n0) * (K / 2), Span * (K / 2));
    auto rs = make_rsrc(a.sfb + int64_t(b.expert) * a.sfb_group_stride + n0, ((K / 128 - 1) * a.ldb_sf + Span) * 4);
    extern __shared__ __attribute__((aligned(1024))) uint8_t sm[];
#ifdef CDNA4_UP_WEIGHT_CACHE
    constexpr int policy = CDNA4_UP_WEIGHT_CACHE;
#else
    constexpr int policy = 0;
#endif
    static_for<0, CDNA4_UP_WEIGHT_AHEAD == 2 ? 3 : 1>([&](auto Stage) {
        constexpr int k = decltype(Stage)::value;
        auto *st = sm + k * Step;
        static_for<0, BN / 64>([&](auto Part) {
            constexpr int j = decltype(Part)::value;
            __builtin_amdgcn_raw_ptr_buffer_load_lds(rb, (lds_ptr_t)(st + BO + (j * 4 + wave) * 1024), 16, bv,
                (j / 4) * Stride * 256 * (K / 2) + (j % 4) * 4096 + k * 256 * 64, 0, policy);
        });
        static_for<0, BN / 256>([&](auto Half) {
            constexpr int h = decltype(Half)::value;
            __builtin_amdgcn_raw_ptr_buffer_load_lds(
                rs, (lds_ptr_t)(st + SBO + (wave + h * 4) * 256), 4, tid * 4, k * a.ldb_sf * 4 + h * 256 * Stride * 4, 0, 0);
        });
    });
}
#endif
template <int K, int N, bool First, bool Shared, bool Paired = false, int PairStride = 0, int ReadyGroups = 0>
__device__ __forceinline__ void tile(const MegaMoeKernelArgs &q, const MegaMoeRowBlock &b, int column, const int *row_weights = nullptr,
    const uint32_t *local_rows = nullptr, const int *completion = nullptr, bool up_ahead = false) {
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

#ifdef CDNA4_MOE_STAGE_RETURN
    // This disjoint static LDS allocation survives both the operand ring and
    // its output alias. The matrix epilogue drains VMEM and joins every wave
    // before the first return pointer can be consumed. Row-block storage in
    // the symmetric arena is padded to the full M rows.
    __shared__ __attribute__((aligned(16))) int64_t return_rows[M];
    if constexpr (!First && !Shared) {
        if (threadIdx.x < M / 2) {
            auto res = make_rsrc(a.moe_output_rows + b.m0, M * sizeof(int64_t));
            __builtin_amdgcn_raw_ptr_buffer_load_lds(res, (lds_ptr_t)return_rows, 16, threadIdx.x * 16, 0, 0, 16);
        }
    }
#endif
    const int32_t *original_sf = nullptr;
    uint32_t input_bytes = 0;
#ifdef CDNA4_MOE_LOCAL_INPUT
    if constexpr (First && !Shared) {
        if (local_rows) {
            a.A = region<const uint8_t>(q.moe, q.moe.layout.x);
#ifdef CDNA4_MOE_EAGER_LOCAL
            original_sf = region<const int32_t>(q.moe, q.moe.layout.x_sf);
#endif
#ifdef CDNA4_MOE_MIXED_LOCAL_ROWS
            input_bytes = uint32_t(q.moe.layout.l1_records - q.moe.layout.x);
#else
            input_bytes = uint32_t(q.moe.layout.num_max_tokens) * K;
#endif
        }
    }
#endif
#ifdef CDNA4_SHARED_DURING_ROUTE
    if constexpr (First && Shared) {
        original_sf = region<const int32_t>(q.moe, q.moe.layout.x_sf);
        input_bytes = uint32_t(q.moe.layout.num_max_tokens) * K;
    }
#endif
    auto matrix = [&](auto cap) {
        pipeline_native_tile<K, N, First, !Shared, !Shared && CDNA4_MOE_WEIGHT_PANELS, BN, Paired ? PairStride : 0, ReadyGroups,
            decltype(cap)::value>(
            a, b.expert, b.m0, b.end, column, local_rows, input_bytes, completion, original_sf, row_weights, up_ahead);
    };
    using Full = std::integral_constant<int, 128>;
#ifdef CDNA4_ACTIVE_SLABS
    if constexpr (!Shared && (CDNA4_ACTIVE_SLAB_PROJECTIONS & (First ? 1 : 2))) {
        if constexpr (CDNA4_ACTIVE_SLABS <= 2) {
            matrix(std::integral_constant<int, 0>{});
        } else {
#if defined(CDNA4_USE_SHORT_MFMA16) && CDNA4_SHORT_MFMA16 == 5
            if (b.end - b.m0 <= 16)
                matrix(std::integral_constant<int, 16>{});
            else
#endif
            if (CDNA4_ACTIVE_SLABS == 4 && b.end - b.m0 <= 32)
                matrix(std::integral_constant<int, 32>{});
            else if (b.end - b.m0 <= 64)
                matrix(std::integral_constant<int, 64>{});
            else
                matrix(Full{});
        }
    } else
#endif
        matrix(Full{});

#ifdef CDNA4_MOE_PACKED_EPILOGUE
    if constexpr (First && !Shared && (CDNA4_MOE_PACKED_EPILOGUE & (Paired ? 2 : 1))) {
        // Packed activations and complete exponent words were written from the
        // matrix body. The final VMEM drain and workgroup join remain below.
    } else
#endif
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
#ifdef CDNA4_MOE_STAGE_RETURN
                        addresses[j] = return_rows[relative];
#else
                        addresses[j] = load_relaxed<Scope::device>(a.moe_output_rows + row);
#endif
                        values[j] = *reinterpret_cast<const v4u *>(smem + relative * Pitch + col * 2);
                    }
                });
                // Keep pointer reads ahead of this small store batch. The final
                // per-tile VMEM drain and writer/peer joins remain unchanged.
#ifndef CDNA4_MOE_STAGE_RETURN
                __builtin_amdgcn_s_waitcnt(waitcnt_vm(0));
#endif
                static_for<0, Batch>([&](auto J) {
                    constexpr int j = decltype(J)::value;
                    const int row = b.m0 + row0 + (base + j) * RowStride;
                    if (base + j < Chunks && row < b.end) {
                        const uint64_t pointer = reinterpret_cast<uint64_t>(reinterpret_cast<uint16_t *>(addresses[j]) +
                                                                            column + logical_col);
                        const v4u value = values[j];
#if defined(CDNA4_MOE_STAGE_RETURN) && CDNA4_MOE_STAGE_RETURN == 2
                        const uint64_t local_begin = reinterpret_cast<uint64_t>(region<uint8_t>(q.moe, q.moe.layout.combine));
                        const uint64_t extent = uint64_t(q.moe.layout.num_topk) * q.moe.layout.num_max_tokens * N * 2;
                        if (uint64_t(addresses[j]) - local_begin < extent)
                            asm volatile("global_store_dwordx4 %0, %1, off sc1\n\ts_nop 0" ::"v"(pointer), "v"(value) : "memory");
                        else
#endif
                            asm volatile("global_store_dwordx4 %0, %1, off sc0 sc1\n\ts_nop 0" ::"v"(pointer), "v"(value) : "memory");
                    }
                });
            }
        }
    }
    asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)\n\ts_barrier" ::: "memory");
}

template <int K, int N, bool First, int Groups>
__device__ __forceinline__ void row_columns(const MegaMoeKernelArgs &q, const MegaMoeRowBlock &b, int item, int part, int columns,
    const int *completion = nullptr, int tail_groups = 0, bool up_ahead = false) {
    const int *row_weights = nullptr;
    const uint32_t *local_rows = nullptr;
    if constexpr (First) {
        // Refill after this row's gather publication on every frame/task. This
        // separate static LDS region must not alias recyclable GEMM operands.
        // Each column reuses the same unchanged FP32 routing-weight bit pattern;
        // pointwise readers need only LDS readiness, not another VMEM load/wait.
        __shared__ int staged_weights[M];
        int direct_weight = __float_as_int(1.f);
#ifdef CDNA4_MOE_LOCAL_INPUT
        __shared__ uint32_t source_rows[M];
        const int offset = b.shared ? 0 : load_relaxed<Scope::device>(q.jobs + item);
#ifdef CDNA4_MOE_MIXED_LOCAL_ROWS
        if (offset < 0) {
            const auto &l = q.moe.layout;
            if (threadIdx.x < M) {
                const int row = b.m0 + threadIdx.x;
                uint32_t bytes = uint32_t(l.l1_records - l.x);
                if (row < b.end) {
                    const auto *record = region<const uint32_t>(q.moe, l.l1_records) + int64_t(row) * l.record_ints;
                    const uint32_t code = load_relaxed<Scope::device>(record + K / 128 + 1);
                    const uint32_t source_token = code / l.num_topk;
                    const uint32_t local_begin = uint32_t(q.moe.view.rank) * l.num_max_tokens;
                    const uint32_t token = source_token - local_begin;
                    bytes = token < uint32_t(l.num_max_tokens) ? token * K : uint32_t(l.l1_acts - l.x) + uint32_t(row) * K;
                }
                source_rows[threadIdx.x] = bytes;
            }
            local_rows = source_rows;
        } else
#endif
            if (offset > 0) {
            const auto &l = q.moe.layout;
            const int expert = q.moe.view.rank * l.num_experts_per_rank + b.expert;
            const int *list = region<const int>(q.moe, l.send_lists) + int64_t(expert) * l.num_max_tokens * l.num_topk + offset - 1;
#ifdef CDNA4_MOE_EAGER_LOCAL
            if (threadIdx.x < M) {
                const int row = b.m0 + threadIdx.x;
                uint32_t bytes = uint32_t(l.num_max_tokens) * K;
                if (row < b.end) {
                    const int entry = list[threadIdx.x], token = entry / l.num_topk, slot = entry % l.num_topk;
                    bytes = uint32_t(token) * K;
                    direct_weight = load_relaxed<Scope::device>(region<const int>(q.moe, l.topk_weights) + entry);
                    // Part zero is in every published early K prefix. Its tile
                    // drains all row-pointer writers before the L2 can consume them.
                    if (part == 0) {
                        auto *dst = region<uint16_t>(q.moe, l.combine) + (int64_t(slot) * l.num_max_tokens + token) * K;
                        store_relaxed<Scope::device>(region<int64_t>(q.moe, l.output_rows) + row, reinterpret_cast<int64_t>(dst));
                    }
                }
                source_rows[threadIdx.x] = bytes;
            }
#else
            if (threadIdx.x < M)
                source_rows[threadIdx.x] = b.m0 + threadIdx.x < b.end ? uint32_t(list[threadIdx.x] / l.num_topk) * K :
#ifdef CDNA4_MOE_MIXED_LOCAL_ROWS
                                                                      uint32_t(l.l1_records - l.x);
#else
                                                                      uint32_t(l.num_max_tokens) * K;
#endif
#endif
            local_rows = source_rows;
        }
#endif
        if (threadIdx.x < M) {
            const int row = b.m0 + threadIdx.x;
            int word = __float_as_int(1.f);
#ifdef CDNA4_MOE_EAGER_LOCAL
            if (local_rows)
                word = direct_weight;
            else
#endif
                if (!b.shared && row < b.end)
                word = load_relaxed<Scope::device>(region<const int>(q.moe, q.moe.layout.l1_records) +
                                                   int64_t(row) * q.moe.layout.record_ints + q.moe.layout.hidden / 128);
            staged_weights[threadIdx.x] = word;
        }
        __syncthreads();
        row_weights = staged_weights;
    }
#ifdef CDNA4_TAIL_GROUPS
    if (tail_groups) {
        for (int col = part; col < columns; col += tail_groups) {
            if (b.shared)
                tile<K, N, First, true>(q, b, col * 256, row_weights, local_rows);
            else
                tile<K, N, First, false>(q, b, col * 256, row_weights, local_rows);
        }
        return;
    }
#endif

    {

        {
            if (!b.shared) {
#ifdef CDNA4_MOE_EARLY_K
                if constexpr (First && early_k_panels<N / 256, Groups>() != 0) {
                    constexpr int singles = early_k_panels<N / 256, Groups>();
                    constexpr int stride = N / 256 - Groups;
                    if (part < singles)
                        tile<K, N, true, false>(q, b, part * 256, row_weights, local_rows, nullptr, up_ahead);
                    else
                        tile<K, N, true, false, true, stride>(q, b, part * 256, row_weights, local_rows, nullptr, up_ahead);
                    return;
                }
#endif
#if defined(CDNA4_BALANCE_DOWN) && defined(CDNA4_MOE_EARLY_K)
                // The first short_up workers completed one up panel. Transfer
                // only second down panels from workers that computed two up
                // panels. Each logical output column still has a single writer.
                constexpr int short_up = early_k_panels<2 * K / 256, Groups>();
                constexpr int paired_down = N / 256 - Groups;
                if constexpr (!First && short_up > 0 && paired_down > short_up && paired_down <= Groups) {
                    constexpr int eligible = paired_down - short_up;
                    constexpr int possible = eligible < short_up ? eligible : short_up;
                    constexpr int moved = CDNA4_BALANCE_DOWN == 1   ? (possible + 1) / 2
                                          : CDNA4_BALANCE_DOWN == 4 ? possible / 2 * 2
                                                                    : possible;
                    constexpr int receivers = CDNA4_BALANCE_DOWN == 4 ? moved / 2 : moved;
                    if (completion && part >= short_up && part < short_up + moved) {
                        tile<K, N, false, false, false, 0, Groups>(q, b, part * 256, row_weights, local_rows, completion);
                        return;
                    }
                    if (completion && part < receivers) {
                        const int extra = Groups + short_up + (CDNA4_BALANCE_DOWN == 4 ? 2 * part : part);
                        auto compute_extra = [&]() {
                            if constexpr (CDNA4_BALANCE_DOWN == 4)
                                tile<K, N, false, false, true, 1, Groups>(q, b, extra * 256, row_weights, local_rows, completion);
                            else
                                tile<K, N, false, false, false, 0, Groups>(q, b, extra * 256, row_weights, local_rows, completion);
                        };
                        if constexpr (CDNA4_BALANCE_DOWN == 3)
                            compute_extra();
                        tile<K, N, false, false, true, Groups, Groups>(q, b, part * 256, row_weights, local_rows, completion);
                        if constexpr (CDNA4_BALANCE_DOWN != 3)
                            compute_extra();
                        return;
                    }
                }
#endif
                for (int column = part; column < columns; column += 2 * Groups) {
                    const bool paired = column + Groups < columns;
                    if (paired)
                        tile<K, N, First, false, true, Groups, First ? 0 : Groups>(q, b, column * 256, row_weights, local_rows, completion);
                    else
                        tile<K, N, First, false, false, 0, First ? 0 : Groups>(q, b, column * 256, row_weights, local_rows, completion);
                }
                return;
            }
        }
        for (int column = part; column < columns; column += Groups) {

            if (b.shared)
                tile<K, N, First, true>(q, b, column * (256), row_weights, local_rows);
            else
                tile<K, N, First, false>(q, b, column * (256), row_weights, local_rows);
        }
    }
}

__device__ __forceinline__ void publish_route_count(const MegaMoeArgs &p, int expert, int count) {
    const auto &l = p.layout;
#ifdef CDNA4_MOE_DIRECT_RECEIVER_COUNTS
    // The route-writer system release and grid join precede the rank barrier.
    // Both the list and its count are therefore visible before a peer gathers.
    auto *slot = region<int>(p, l.counts) + p.view.rank * l.num_experts + expert;
    store_relaxed<Scope::system>(p.view.map(slot, expert / l.num_experts_per_rank), count);
    region<int>(p, l.send_counts)[expert] = 0;
#else
    region<int>(p, l.send_counts)[expert] = count;
#endif
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
#ifdef CDNA4_MOE_ROUTE_WAVE_RANGES
    // Each wave owns a contiguous range, so a single workgroup prefix
    // suffices for every ordered list position in this batch.
    constexpr int Waves = Threads / 64;
    const uint64_t earlier = lane ? (uint64_t(1) << lane) - 1 : 0;
    for (int first = 0; first < entries; first += Threads * Batch) {
        const int wave_begin = first + wave * 64 * Batch;
        uint64_t masks[Batch];
        int wave_total = 0;
        static_for<0, Batch>([&](auto J) {
            constexpr int j = decltype(J)::value;
            const int entry = wave_begin + j * 64 + lane;
            masks[j] = __ballot(entry < entries && indices[entry] == expert);
            wave_total += __popcll(masks[j]);
        });
        if (lane == 0)
            wave_counts[wave] = wave_total;
        __syncthreads();
        int position = total;
        for (int w = 0; w < Waves; ++w) {
            const int count = wave_counts[w];
            if (w < wave)
                position += count;
            total += count;
        }
        static_for<0, Batch>([&](auto J) {
            constexpr int j = decltype(J)::value;
            const uint64_t mask = masks[j];
            if ((mask >> lane) & 1u) {
                const int entry = wave_begin + j * 64 + lane;
                const int index = position + __popcll(mask & earlier);
                region<int>(p, l.positions)[entry] = index;
                if (p.ordered_push)
                    region<int>(p, l.send_lists)[int64_t(expert) * l.num_max_tokens * l.num_topk + index] = entry;
            }
            position += __popcll(mask);
        });
        __syncthreads();
    }
#else
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
#endif
    if (tid == 0)
        publish_route_count(p, expert, total);
}

#ifdef CDNA4_MOE_ROUTE_BITMAP_GROUP
// Each source slot owns one bit. Word order and bit order together give the
// same stable positions as scanning the flattened top-k tensor for each expert.
template <int Group> __device__ __forceinline__ void route_bitmap_group(const MegaMoeArgs &p, int first) {
    const auto &l = p.layout;
    const int entries = p.num_tokens * l.num_topk, words = (entries + 31) / 32;
    const int tid = threadIdx.x, lane = tid % 64, wave = tid / 64;
    const auto *ids = region<const int64_t>(p, l.topk_idx);
    extern __shared__ __attribute__((aligned(1024))) uint8_t smem[];
    auto *mask = reinterpret_cast<unsigned *>(smem);
#ifdef CDNA4_MOE_ROUTE_PLANES
    static_assert(Group == 1 || Group == 2 || Group == 4 || Group == 8);
    auto index_mask = [&](int expert, int word) { return CDNA4_MOE_ROUTE_PLANES == 1 ? expert * words + word : word * Group + expert; };
    // A wave owns both 32-bit words of a consecutive 64-slot range.
    // A group ballot plus the low ID bit planes give every expert's exact
    // membership. No two waves write the same word, including the last one.
    for (int base = 64 * wave; base < entries; base += int(blockDim.x)) {
        const int entry = base + lane;
        const int64_t id = entry < entries ? ids[entry] : -1;
        const int e = lane % Group, half = lane / Group;
        uint64_t hits = 0;
#if CDNA4_MOE_ROUTE_PLANES == 4
#pragma unroll
        for (int k = 0; k < Group; ++k) {
            const uint64_t ballot = __ballot(id == first + k && first + k < l.num_experts);
            if (e == k)
                hits = ballot;
        }
#else
        hits = __ballot(id >= first && id < int64_t(min(first + Group, l.num_experts)));
#if CDNA4_MOE_ROUTE_PLANES == 3
        if (hits) {
#endif
#pragma unroll
            for (int bit = 1; bit < Group; bit *= 2) {
                const uint64_t plane = __ballot((id & bit) != 0);
                hits &= (e & bit) ? plane : ~plane;
            }
#if CDNA4_MOE_ROUTE_PLANES == 3
        }
#endif
#endif
        const int word = base / 32 + half;
        if (half < 2 && word < words)
            mask[index_mask(e, word)] = uint32_t(hits >> (32 * half));
    }
#else
    auto index_mask = [&](int expert, int word) { return expert * words + word; };
    for (int i = tid; i < Group * words; i += blockDim.x)
        mask[i] = 0;
    __syncthreads();
    for (int entry = tid; entry < entries; entry += blockDim.x) {
        const int64_t expert = ids[entry];
        if (expert >= first && expert < min(first + Group, l.num_experts))
            atomicOr(mask + int(expert - first) * words + entry / 32, 1u << (entry % 32));
    }
#endif
    __syncthreads();
    for (int offset = wave; offset < Group && first + offset < l.num_experts; offset += 4) {
        const int expert = first + offset;
        int total = 0;
        for (int base = 0; base < words; base += 64) {
            const int word = base + lane;
            uint32_t bits = word < words ? mask[index_mask(offset, word)] : 0;
            const int count = __popc(bits);
            int prefix = count;
#pragma unroll
            for (int k = 1; k < 64; k *= 2) {
                const int x = __shfl_up(prefix, k, 64);
                if (lane >= k)
                    prefix += x;
            }
            int position = total + prefix - count;
            total += __shfl(prefix, 63, 64);
            while (bits) {
                const int bit = __ffs(bits) - 1, entry = word * 32 + bit;
                region<int>(p, l.positions)[entry] = position;
                if (p.ordered_push)
                    region<int>(p, l.send_lists)[int64_t(expert) * l.num_max_tokens * l.num_topk + position] = entry;
                ++position;
                bits &= bits - 1;
            }
        }
        if (lane == 0)
            publish_route_count(p, expert, total);
    }
    __syncthreads();
}
#endif

// Each wave owns one entire physical expert list. Stable positions follow
// from the wave ballot, with no inter-wave prefix or workgroup rendezvous.
__device__ __forceinline__ void route_wave_expert(const MegaMoeArgs &p, int expert) {
    const auto &l = p.layout;
    const int lane = threadIdx.x % 64;
    const auto *indices = region<const int64_t>(p, l.topk_idx);
    const int entries = p.num_tokens * l.num_topk;
    const uint64_t earlier = lane ? (uint64_t(1) << lane) - 1 : 0;
    int total = 0;
    for (int base = 0; base < entries; base += 512) {
        uint64_t masks[8];
        static_for<0, 8>([&](auto J) {
            const int entry = base + decltype(J)::value * 64 + lane;
            masks[decltype(J)::value] = __ballot(entry < entries && indices[entry] == expert);
        });
        static_for<0, 8>([&](auto J) {
            constexpr int j = decltype(J)::value;
            const uint64_t mask = masks[j];
            if ((mask >> lane) & 1u) {
                const int entry = base + j * 64 + lane;
                const int position = total + __popcll(mask & earlier);
                region<int>(p, l.positions)[entry] = position;
                if (p.ordered_push)
                    region<int>(p, l.send_lists)[int64_t(expert) * l.num_max_tokens * l.num_topk + position] = entry;
            }
            total += __popcll(mask);
        });
    }
    if (lane == 0)
        publish_route_count(p, expert, total);
}

template <int H, int TopK, int ColumnGroups>
__device__ __forceinline__ void run_gather_actor(const MegaMoeKernelArgs &q, int *job, uint32_t epoch, int first_block) {
    const auto &p = q.moe;
    // These actors read inputs and route lists produced by peer GPUs. The
    // local grid joins do not replace the required remote-input acquire.

#ifdef CDNA4_MOE_CACHE_GATHER_TABLE
    extern __shared__ int scratch[];
    const RowTable table = cache_gather_table(p, scratch);
#else
    const RowTable table{};
#endif
    for (;;) {
        if (threadIdx.x == 0)
            job[0] = atomic_add<Scope::device>(q.state + PullHead, 1);
        __syncthreads();
        const int index = __builtin_amdgcn_readfirstlane(job[0]);
        if (first_block + index / CDNA4_MOE_GATHER_SHARDS >= q.state[BlockCount])
            break;

        gather<H>(q, first_block + index / CDNA4_MOE_GATHER_SHARDS, index % CDNA4_MOE_GATHER_SHARDS, table);
    }
    // In the metadata-first policy, make row maps available before this
    // actor starts sending its source packets. Matrix consumers retain the
    // per-packet ready waits, so early metadata is not early payload access.
}

template <int H, int I, int TopK, int ColumnGroups = 6>
__device__ __forceinline__ void run(const MegaMoeKernelArgs &q) {
    const auto &p = q.moe;
    record_phase(q, 0);
    online_tick(q, 2);
#ifdef DG_MEGA_MOE_PHASE_TIMESTAMPS
    uint64_t elapsed[9]{};
    uint64_t last_tick = __builtin_readcyclecounter();
    auto mark = [&](int phase) {
        const uint64_t now = __builtin_readcyclecounter();
        elapsed[phase] += now - last_tick;
        last_tick = now;
    };
#else
    auto mark = [](int) {};
#endif

#if defined(CDNA4_ROUTE_JOIN) && CDNA4_ROUTE_JOIN >= 2
    constexpr uint32_t epoch = 0;
    const int writers = min(int(gridDim.x), (p.layout.num_experts + CDNA4_MOE_ROUTE_BITMAP_GROUP - 1) / CDNA4_MOE_ROUTE_BITMAP_GROUP);
    // The next launch may restore the persistent epoch protocol after wraps.
    // Leave its reusable arrival slots in their initial state.
    if (blockIdx.x == 0 && threadIdx.x < gridDim.x)
        store_relaxed<Scope::system>(reinterpret_cast<uint32_t *>(q.k_ready) + threadIdx.x, uint32_t(0));
#else
    const uint32_t epoch = load_relaxed<Scope::system>(region<const uint32_t>(p, p.layout.barrier)) * (8u);
#endif
    if (blockIdx.x == 0 && threadIdx.x == 0)
        store_relaxed<Scope::system>(reinterpret_cast<uint32_t *>(q.k_ready) + ControlRelayWord, uint32_t(0));
    // The unchanged first all-writer join completes this reset before any
    // control relay is consumed. Re-enabling a relay after arbitrary intervening
    // launches cannot observe its stale commit, including an epoch wrap.

#if defined(CDNA4_ROUTE_JOIN)
#if CDNA4_ROUTE_JOIN == 1
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < StateSize + 5 * q.max_blocks; i += gridDim.x * blockDim.x)
        q.state[i] = 0;
#endif
#else
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < StateSize + 2 * q.max_blocks + 2 * q.max_blocks * q.n2;
         i += gridDim.x * blockDim.x)
        q.state[i] = 0;
#endif

#ifdef CDNA4_MOE_ROUTE_BITMAP_GROUP
    for (int group = blockIdx.x * CDNA4_MOE_ROUTE_BITMAP_GROUP; group < p.layout.num_experts;
        group += gridDim.x * CDNA4_MOE_ROUTE_BITMAP_GROUP)
        route_bitmap_group<CDNA4_MOE_ROUTE_BITMAP_GROUP>(p, group);
#elif defined(CDNA4_MOE_WAVE_PER_EXPERT)
    for (int e = blockIdx.x * 4 + threadIdx.x / 64; e < p.layout.num_experts; e += gridDim.x * 4)
        route_wave_expert(p, e);
    __syncthreads();
#else
    for (int e = blockIdx.x; e < p.layout.num_experts; e += gridDim.x) {
        route_prefetched_expert<256, 8>(p, e);
        __syncthreads();
    }
#endif
    // Block zero exchanges routed counts. The other resident workgroups can
    // already project local shared-expert inputs, which do not depend on routing.
    int early_blocks = 0;
#ifdef CDNA4_MOE_EARLY_SHARED_L1
    const int shared_rows = (q.shared_tokens + M - 1) / M;
    if (shared_rows <= (int(gridDim.x) - 1) / ColumnGroups)
        early_blocks = shared_rows;
    const int shared_ticket = int(blockIdx.x) - 1;
    const bool assigned = shared_ticket >= 0 && shared_ticket < early_blocks * ColumnGroups;
    const int shared_item = assigned ? shared_ticket / ColumnGroups : 0;
    const int shared_part = assigned ? shared_ticket % ColumnGroups : 0;
    const MegaMoeRowBlock shared{0, shared_item * M, min((shared_item + 1) * M, q.shared_tokens), 1};
    if (assigned && shared_part == 0)
        prepare_shared_scales(p, shared);
#endif
    // Route stores on every CU must precede the system release in the leader.

    // Only the metadata producer needs route counts at this point. All other
    // workgroups wait at the task-table publication below, avoiding an
    // all-to-all scan of the same arrival flags.
#if defined(CDNA4_ROUTE_JOIN) && CDNA4_ROUTE_JOIN >= 2
    if (blockIdx.x < writers) {
        // Release every lane's immutable lists and remote counts before
        // publishing completion of this writer to the metadata workgroup.
        fence_release<Scope::system>();
        __syncthreads();
        if (threadIdx.x == 0) {
#if CDNA4_ROUTE_JOIN == 3
            store_release<Scope::system>(q.job_ready + q.max_blocks + blockIdx.x, 1);
#else
            atomic_add_release<Scope::system>(q.state + SecondHead, 1);
#endif
        }
    }
#ifdef CDNA4_SHARED_DURING_ROUTE
    constexpr int cols = CDNA4_SHARED_DURING_ROUTE;
    const int dense_rows = (q.shared_tokens + M - 1) / M, first_dense = max(writers, q.dispatch_blocks);
    const bool overlap = dense_rows > 0 && dense_rows <= 4 && dense_rows * cols + first_dense <= int(gridDim.x);
    const int dense_ticket = int(blockIdx.x) - first_dense;
    if (overlap && dense_ticket >= 0 && dense_ticket < dense_rows * cols) {
        // These workgroups have no routing stores. Remote count publication,
        // task construction, and subsequent gathering proceed independently.
        const int item = dense_ticket / cols, part = dense_ticket % cols;
        const MegaMoeRowBlock b{0, item * M, min(q.shared_tokens, (item + 1) * M), 1};
        for (int column = part; column < q.n1; column += cols)
            tile<H, 2 * I, true, true>(q, b, column * 256);
        __syncthreads();
        auto *count = q.state + BlockCount + 8 + item;
        if (threadIdx.x == 0) {
            atomic_add_release<Scope::device>(count, 1);
            while (load_relaxed<Scope::device>(count) < cols)
                __builtin_amdgcn_s_sleep(4);
        }
        __syncthreads();
        acquire_row_payload();
        __syncthreads();
        for (int column = part; column < q.n2; column += cols)
            tile<I, H, false, true>(q, b, column * 256);
    }
#endif
    if (blockIdx.x == 0) {
#if CDNA4_ROUTE_JOIN == 3
        for (int w = threadIdx.x; w < writers; w += blockDim.x)
            while (!load_relaxed<Scope::system>(q.job_ready + q.max_blocks + w))
                __builtin_amdgcn_s_sleep(4);
#else
        if (threadIdx.x == 0)
            while (load_relaxed<Scope::system>(q.state + SecondHead) != writers)
                __builtin_amdgcn_s_sleep(4);
#endif
        __syncthreads();
        fence_acquire<Scope::system>();
        __syncthreads();
    }
#else
    fence_release<Scope::system>();
    __syncthreads();
    auto *route_flags = reinterpret_cast<uint32_t *>(q.k_ready);
    if (threadIdx.x == 0)
        store_release<Scope::system>(route_flags + blockIdx.x, epoch + 1u);
    __syncthreads();
#ifdef CDNA4_MOE_EARLY_SHARED_L1
    if (assigned) {
        const int producer = 1 + shared_item * ColumnGroups;
        if (threadIdx.x == 0) {
            uint32_t observed;
            do {
                observed = load_relaxed<Scope::system>(route_flags + producer);
                if (!observed || int32_t(observed - (epoch + 1u)) < 0)
                    __builtin_amdgcn_s_sleep(4);
            } while (!observed || int32_t(observed - (epoch + 1u)) < 0);
        }
        __syncthreads();
        fence_acquire<Scope::system>();
        __syncthreads();
        row_columns<H, 2 * I, true, ColumnGroups>(q, shared, shared_item, shared_part, q.n1);
        // The existing grid join now publishes L1 payloads from these groups.
        fence_release<Scope::system>();
        __syncthreads();
    }
#endif
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
#endif
    record_phase(q, 1);
#ifndef CDNA4_TRACE_PEER_JOIN
    online_tick(q, 3);
#endif

    if (blockIdx.x == 0) {
#ifdef CDNA4_MOE_RECEIVER_COUNTS
        if (p.layout.num_experts_per_rank <= blockDim.x)
            publish_receiver_counts(p);
        else
#endif
            mega_moe_barrier_impl<true>(p);
        // Only this producer consumes the count metadata before publishing
        // the task table. Join its writers locally; readers wait at tag 3.
        fence_release<Scope::system>();
        asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)\n\ts_barrier" ::: "memory");
        fence_acquire<Scope::system>();
    }
    record_phase(q, 2);
#ifndef CDNA4_TRACE_PEER_JOIN
    online_tick(q, 4);
#endif

    if (blockIdx.x == 0) {
#ifdef CDNA4_SHARED_DURING_ROUTE
        const int shared_blocks = overlap ? 0 : dense_rows;
#else
        const int shared_blocks = (q.shared_tokens + M - 1) / M;
#endif
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
        }
#ifdef CDNA4_MOE_PARALLEL_BLOCK_TABLE
        else if (p.layout.num_experts_per_rank <= 256) {
            __shared__ int totals[4];
            const int expert = threadIdx.x, lane = expert % 64, wave = expert / 64;
            const bool valid = expert < p.layout.num_experts_per_rank;
            const int start = valid ? expert_start(ends, expert, p.layout.alignment) : 0;
            const int end = valid ? ends[expert] : 0;
            const int count = (end - start + M - 1) / M;
            int prefix = count;
#pragma unroll
            for (int offset = 1; offset < 64; offset *= 2) {
                const int previous = __shfl_up(prefix, offset, 64);
                if (lane >= offset)
                    prefix += previous;
            }
            if (lane == 63)
                totals[wave] = prefix;
            __syncthreads();
            int base = shared_blocks + prefix - count;
            for (int w = 0; w < wave; ++w)
                base += totals[w];
            for (int i = 0; i < count; ++i)
                q.blocks[base + i] = {expert, start + i * M, min(start + (i + 1) * M, end), 0};
            if (expert == 0)
                q.state[BlockCount] = shared_blocks + totals[0] + totals[1] + totals[2] + totals[3];
        }
#endif
        else if (threadIdx.x == 0) {
            int count = shared_blocks;
            for (int expert = 0; expert < p.layout.num_experts_per_rank; ++expert) {
                const int start = expert_start(ends, expert, p.layout.alignment), end = ends[expert];
                for (int row = start; row < end; row += M)
                    q.blocks[count++] = {expert, row, min(row + M, end), 0};
            }
            q.state[BlockCount] = count;
        }
#ifdef CDNA4_MOE_EAGER_LOCAL
        // After count exchange, each complete block lying inside the local
        // source interval can run immediately. Its route list is immutable.
        __syncthreads();
        const int total = q.state[BlockCount], local = p.view.rank, experts = p.layout.num_experts_per_rank;
        const int *counts = region<const int>(p, p.layout.counts) + local * experts;
        for (int item = shared_blocks + threadIdx.x; item < total; item += blockDim.x) {
            const auto b = q.blocks[item];
            int begin = expert_start(ends, b.expert, p.layout.alignment);
            for (int source = 0; source < local; ++source)
                begin += load_relaxed<Scope::system>(counts + source * p.layout.num_experts + b.expert);
            const int n = load_relaxed<Scope::system>(counts + local * p.layout.num_experts + b.expert);
            if (b.m0 >= begin && b.end <= begin + n) {
                q.jobs[item] = b.m0 - begin + 1;
                q.sf_ready[item] = CDNA4_MOE_GATHER_SHARDS;
            }
        }
#if CDNA4_MOE_EAGER_LOCAL == 2
        __syncthreads();
        auto *order = q.jobs + q.max_blocks;
        if (total <= blockDim.x) {
            __shared__ int groups[4];
            const int t = threadIdx.x, lane = t % 64, wave = t / 64;
            const bool ready = t >= shared_blocks && t < total && q.jobs[t] > 0;
            const uint64_t mask = __ballot(ready);
            if (lane == 0)
                groups[wave] = __popcll(mask);
            __syncthreads();
            int earlier = __popcll(mask & ((uint64_t(1) << lane) - 1)), all = 0;
            for (int w = 0; w < 4; ++w) {
                if (w < wave)
                    earlier += groups[w];
                all += groups[w];
            }
            if (t < total)
                order[ready ? earlier : all + t - earlier] = t;
        } else
            for (int item = threadIdx.x; item < total; item += blockDim.x)
                order[item] = item;
#endif
        __syncthreads();
#endif
    }
#if defined(CDNA4_ROUTE_JOIN) && CDNA4_ROUTE_JOIN >= 2
    if (blockIdx.x == 0) {
        fence_release<Scope::system>();
        __syncthreads();
        if (threadIdx.x == 0)
            store_release<Scope::system>(q.state + SecondTail, 1);
    } else if (threadIdx.x == 0) {
        while (!load_relaxed<Scope::system>(q.state + SecondTail))
            __builtin_amdgcn_s_sleep(4);
    }
    __syncthreads();
    fence_acquire<Scope::system>();
    __syncthreads();
#else
#ifdef CDNA4_MOE_PUBLISH_TASK_TABLE
    if (!early_blocks)
        publish_task_table(q, epoch + 3u);
    else
#endif
        grid_sync_control(q, epoch + 3u);
#endif
    record_phase(q, 3);
#ifdef CDNA4_TRACE_PEER_JOIN
    online_tick(q, 3);
#else
    online_tick(q, 5);
#endif
    mark(0);

    __shared__ int job[4];
    if (blockIdx.x < q.dispatch_blocks)
        run_gather_actor<H, TopK, ColumnGroups>(q, job, epoch, early_blocks);
    mark(1);
    {
        // Coarse M-tile tasks keep both GEMMs on the same workgroup. Different
        // row tiles still overlap gather, first GEMM, second GEMM and peer return,
        // but avoid a global atomic queue operation for every 256-column slice.
#ifdef CDNA4_TAIL_GROUPS
        const int count = q.state[BlockCount], workers = int(gridDim.x) - q.dispatch_blocks;
        const int normal_rows = max(1, workers / ColumnGroups), remainder = count % normal_rows;
        const int tail_width = CDNA4_ADAPTIVE_TAIL ? min(CDNA4_TAIL_GROUPS, workers / max(remainder, 1)) : CDNA4_TAIL_GROUPS;
        const bool expand =
            ColumnGroups > 1 && early_blocks == 0 && remainder > 0 && tail_width >= ColumnGroups + 2 && remainder * tail_width <= workers;
        const int original_prefix = expand ? count - remainder : count;
        const int prefix_tickets = original_prefix * ColumnGroups;
#endif
#ifdef CDNA4_MOE_RESERVED_TICKETS
        // Non-gather workers own the initial ticket range directly. Once
        // used, these workers and completed gather workers share the original
        // dynamic queue for every later ticket. Early shared work keeps its
        // original scheduling contract.
        const int compute_workers = int(gridDim.x) - q.dispatch_blocks;
        constexpr int policy = CDNA4_MOE_RESERVED_TICKETS;
        const int width = policy == 1 ? compute_workers : (compute_workers / ((policy == 4 ? 2 : 1) * ColumnGroups)) * ColumnGroups;
        const int reserved = early_blocks ? 0 : width;
        const int worker = int(blockIdx.x) - q.dispatch_blocks;
        bool initial = worker >= 0 && worker < reserved;
#endif
        for (;;) {
            int ticket;
#ifdef CDNA4_MOE_RESERVED_TICKETS
            if (initial) {
                if constexpr (policy == 3) {
                    const int rows = reserved / ColumnGroups;
                    ticket = (worker % rows) * ColumnGroups + worker / rows;
                } else
                    ticket = worker;
                initial = false;
            } else {
                if (threadIdx.x == 0)
                    job[0] = reserved + atomic_add<Scope::device>(q.state + FirstHead, 1);
                __syncthreads();
                ticket = __builtin_amdgcn_readfirstlane(job[0]);
            }
#else
            if (threadIdx.x == 0)
                job[0] = atomic_add<Scope::device>(q.state + FirstHead, 1);
            __syncthreads();
            ticket = __builtin_amdgcn_readfirstlane(job[0]);
#endif
            int item, part, tail_groups = 0;
            {

                {
#ifdef CDNA4_TAIL_GROUPS
                    if (expand && ticket >= prefix_tickets) {
                        tail_groups = tail_width;
                        const int remaining = ticket - prefix_tickets;
                        item = original_prefix + remaining / tail_width;
                        part = remaining % tail_width;
                    } else
#endif
                    {
                        item = ticket < 0 ? -1 : (ticket / ColumnGroups);
                        part = (ticket % ColumnGroups);
                    }
                }
            }
            mark(2);
            if (item < 0 || item >= q.state[BlockCount])
                break;

#if defined(CDNA4_MOE_EAGER_LOCAL) && CDNA4_MOE_EAGER_LOCAL == 2
            item = load_relaxed<Scope::device>(q.jobs + q.max_blocks + item);
#endif
            const auto b = q.blocks[item];
            const int *completion = nullptr;
#if defined(CDNA4_MOE_EARLY_K) && CDNA4_MOE_EARLY_K == 2
            if constexpr (early_k_panels<2 * I / 256, ColumnGroups>() != 0)
                if (!b.shared && !tail_groups) {
#if defined(CDNA4_PROGRESSIVE_K) && CDNA4_PROGRESSIVE_K <= 2
                    completion = q.job_ready + item;
#else
                    completion = q.l1_done + item;
#endif
                }
#endif
#if defined(CDNA4_USE_SHORT_MFMA16) && CDNA4_SHORT_MFMA16 == 5
            // Tiny down tiles read complete K stages. All column owners of
            // this item use the ordinary release/join before starting them.
            if (!b.shared && b.end - b.m0 <= 32)
                completion = nullptr;
#endif
            if (item >= early_blocks) {
                bool up_ahead = false;
#if defined(CDNA4_UP_WEIGHT_AHEAD) && defined(CDNA4_MOE_EARLY_K)
                constexpr int singles = early_k_panels<2 * I / 256, ColumnGroups>();
                if constexpr (singles > 0) {
                    if (!b.shared && !tail_groups) {
                        const bool paired = part >= singles;
                        up_ahead = (CDNA4_UP_WEIGHT_AHEAD <= 2 || (CDNA4_UP_WEIGHT_AHEAD == 3 && paired) ||
                                    (CDNA4_UP_WEIGHT_AHEAD == 4 && !paired));
                        if (up_ahead) {
                            if (paired)
                                prefetch_up_weights<H, 2 * I / 256 - ColumnGroups>(q, b, part);
                            else
                                prefetch_up_weights<H, 0>(q, b, part);
                        }
                    }
                }
#endif
                if (threadIdx.x == 0)
                    while (load_relaxed<Scope::device>(q.sf_ready + item) < CDNA4_MOE_GATHER_SHARDS)
                        __builtin_amdgcn_s_sleep(16);
                __syncthreads();
                acquire_row_payload();
                __syncthreads();

                mark(3);

                const int first_columns = q.n1 * (1);
                row_columns<H, 2 * I, true, ColumnGroups>(q, b, item, part, first_columns, nullptr, tail_groups, up_ahead);
                // Each native tile already drains every coherent payload writer and
                // joins its waves. No ordinary payload store is allowed in this policy.

                __syncthreads();
                mark(4);
                if constexpr (ColumnGroups > 1) {
                    if (threadIdx.x == 0) {
#if defined(CDNA4_PROGRESSIVE_K) && CDNA4_PROGRESSIVE_K <= 2
                        if (completion) {
                            constexpr int singles = early_k_panels<2 * I / 256, ColumnGroups>();
                            constexpr int stride = 2 * I / 256 - ColumnGroups;
                            unsigned mask = 1u << part;
                            if (part >= singles)
                                mask |= 1u << (part + stride);
                            // Every producing wave drained its activation and
                            // scale stores in tile() before this publication.
                            __hip_atomic_fetch_or(
                                reinterpret_cast<unsigned *>(q.job_ready + item), mask, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
                        } else {
                            atomic_add<Scope::device>(q.l1_done + item, 1);
                            while (load_relaxed<Scope::device>(q.l1_done + item) < (tail_groups ? tail_groups : ColumnGroups))
                                __builtin_amdgcn_s_sleep(4);
                        }
#else
                        atomic_add<Scope::device>(q.l1_done + item, 1);
#if defined(CDNA4_MOE_EARLY_K) && CDNA4_MOE_EARLY_K == 2
                        if (completion) {
                            constexpr int prefix = early_k_panels<2 * I / 256, ColumnGroups>();
                            if (part < prefix)
                                atomic_add<Scope::device>(q.job_ready + item, 1);
                            while (load_relaxed<Scope::device>(q.job_ready + item) < prefix)
                                __builtin_amdgcn_s_sleep(4);
                        } else
#endif
                            while (load_relaxed<Scope::device>(q.l1_done + item) < (tail_groups ? tail_groups : ColumnGroups))
                                __builtin_amdgcn_s_sleep(4);
#endif
                    }
                    __syncthreads();
                    acquire_row_payload();
                    __syncthreads();
                }
            }
            mark(5);
            const int second_columns = q.n2 * (1);
            row_columns<I, H, false, ColumnGroups>(q, b, item, part, second_columns, completion, tail_groups);

            // In the deferred policy no consumer observes a row-level completion
            // signal. Every writer's final completion path still performs a system
            // release before publication; the leader waits for *all* such writers
            // before the peer barrier and reduction. Never use this policy with an
            // asynchronous row consumer or a dedicated local-to-peer return actor.
            __syncthreads();
            mark(6);
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
#if defined(CDNA4_ROUTE_JOIN) && CDNA4_ROUTE_JOIN >= 2
    auto *committed = reinterpret_cast<uint32_t *>(q.state + BlockCount + 1);
#else
    auto *committed = reinterpret_cast<uint32_t *>(q.k_ready) + ControlRelayWord;
#endif
    const uint32_t completed_epoch = epoch + 5u + final_offset;
    if (job[0] == int(gridDim.x) - 1) {
        fence_acquire<Scope::system>();
        __syncthreads();
#ifdef CDNA4_TRACE_PEER_JOIN
        // This elected workgroup holds the publication of all local writers.
        online_tick(q, 4, true);
#endif
        mega_moe_barrier_impl<false>(p);
#ifdef CDNA4_TRACE_PEER_JOIN
        online_tick(q, 5, true);
#endif
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
    online_tick(q, 6);
    mark(7);

    moe_pointwise::reduce<H, TopK>(p);
    record_phase(q, 6);
    online_tick(q, 7);
    mark(8);
#ifdef DG_MEGA_MOE_PHASE_TIMESTAMPS
    // The control tail reserves 64 KiB. Slots 512..2815 are disjoint
    // from grid flags, block-zero phase stamps and the final relay.
    if (threadIdx.x == 0)
        for (int phase = 0; phase < 9; ++phase)
            q.k_ready[512 + blockIdx.x * 9 + phase] = elapsed[phase];
#endif

    // Prevent an early-finishing rank from reusing inputs while peers are still
    // reading this call. The next launch's counts barrier supplies this ordering.
}
} // namespace resident_moe
