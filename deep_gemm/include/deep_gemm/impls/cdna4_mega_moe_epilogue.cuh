#pragma once
#include <deep_gemm/impls/cdna4/device_common.h>
#include <deep_gemm/impls/cdna4/gemm_args.h>
#include <deep_gemm/impls/cdna4_mega_moe_memory.cuh>

// A staged 32 x 256 BF16 gate/up tile becomes 32 x 128 activations. Eight lanes per row,
// sixteen outputs per lane; adjacent lanes jointly reduce a 32-element quantization group.
// The BF16 stage preserves exactly the standalone GEMM -> SwiGLU rounding boundary.
template <bool BF16, int WM_ROWS = 128, bool COHERENT = false, int TileN = 256, bool StagedRowWeights = false,
          int TileThreads = 256>
__device__ __forceinline__ void pipeline_swiglu_epilogue(const deep_gemm::cdna4::GemmArgs &p, const uint8_t *lds,
                                                         int pitch, int m0, int mend, int n0, int round, int tid,
                                                         const int *row_weights = nullptr) {
    static_assert(TileN == 128 || TileN == 256);
    constexpr int Lanes = TileN / 32;
    const int r = tid / Lanes, c = tid % Lanes;
    const int row =
        TileN == 128 ? m0 + (TileThreads / Lanes) * round + r : m0 + (r / 16) * WM_ROWS + 16 * round + r % 16;
    const uint16_t *src = reinterpret_cast<const uint16_t *>(lds + r * pitch) + 32 * c;
    float weight = row < mend ? 1.f : 0.f;
    if (row < mend && p.moe_records) {
        const auto *ptr = p.moe_records + int64_t(row) * p.moe_record_stride + p.moe_weight_word;
        if constexpr (StagedRowWeights) {
            const auto *local = (const __attribute__((address_space(3))) int *)row_weights;
            weight = __int_as_float(local[row - m0]);
        } else if constexpr (COHERENT)
            weight = __int_as_float(__hip_atomic_load(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM));
        else
            weight = __int_as_float(*ptr);
    }
    float clamp, unused;
    unpack_bf16x2(pack_bf16x2(p.moe_clamp, 0.f), clamp, unused);
    float values[16], amax = 0.f;
#pragma unroll
    for (int block = 0; block < 2; ++block) {
        const v4u gate = *reinterpret_cast<const v4u *>(src + 16 * block);
        const v4u up = *reinterpret_cast<const v4u *>(src + 16 * block + 8);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float g[2], u[2];
            unpack_bf16x2(gate[j], g[0], g[1]);
            unpack_bf16x2(up[j], u[0], u[1]);
#pragma unroll
            for (int z = 0; z < 2; ++z) {
                const float x = fminf(g[z], clamp), y = fminf(fmaxf(u[z], -clamp), clamp);
                const float denominator = 1.f + (p.moe_fast_math ? __expf(-x) : expf(-x));
                const float silu = p.moe_fast_math ? x * __builtin_amdgcn_rcpf(denominator) : x / denominator;
                const float v = silu * y * weight;
                values[block * 8 + j * 2 + z] = v;
                amax = fmaxf(amax, fabsf(v));
            }
        }
    }
    const int64_t offset = int64_t(row) * (p.N / 2) + n0 / 2 + c * 16;
    if constexpr (BF16) {
        v4u packed[2];
#pragma unroll
        for (int j = 0; j < 8; ++j)
            packed[j / 4][j % 4] = pack_bf16x2(values[j * 2], values[j * 2 + 1]);
        if (row < mend) {
            auto *out = reinterpret_cast<v4u *>(p.moe_acts + offset * 2);
            if constexpr (COHERENT) {
                mega_moe_memory::coherent_store16(out, packed[0]);
                mega_moe_memory::coherent_store16(out + 1, packed[1]);
            } else {
                out[0] = packed[0];
                out[1] = packed[1];
            }
        }
    } else {
        amax = fmaxf(amax, __shfl_xor(amax, 1));
        const uint32_t rounded = (__float_as_uint(amax) + 0x7fffffu - (6291456)) >> 23;
        const uint32_t exponent = max(rounded, 113u) - 8u;
        const float inverse = __uint_as_float((254u - exponent) << 23);
        v4u packed;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int lo =
                __builtin_amdgcn_cvt_pk_fp8_f32(values[4 * j] * inverse, values[4 * j + 1] * inverse, 0, false);
            packed[j] =
                __builtin_amdgcn_cvt_pk_fp8_f32(values[4 * j + 2] * inverse, values[4 * j + 3] * inverse, lo, true);
        }
        if (row < mend) {
            if constexpr (COHERENT)
                mega_moe_memory::coherent_store16(p.moe_acts + offset, packed);
            else
                *reinterpret_cast<v4u *>(p.moe_acts + offset) = packed;
        }
        const int base = (tid % 64) & ~(Lanes - 1);
        uint32_t word = uint32_t(__shfl(int(exponent), base)) | (uint32_t(__shfl(int(exponent), base + 2)) << 8);
        if constexpr (TileN == 256)
            word |=
                (uint32_t(__shfl(int(exponent), base + 4)) << 16) | (uint32_t(__shfl(int(exponent), base + 6)) << 24);
        if (row < mend && c == 0) {
            auto *ptr = p.moe_sf + int64_t(n0 / 256) * p.moe_sf_pitch + row;
            if constexpr (TileN == 128) {
                // The row-group release/join completes both disjoint halfwords before
                // GEMM2 reads the word. Narrow coherent callers additionally keep both
                // halves with the same column owner; no read-modify-write is required.
                auto *half = reinterpret_cast<uint16_t *>(ptr) + (n0 % 256) / 128;
                if constexpr (COHERENT) {
                    const uint64_t address = reinterpret_cast<uint64_t>(half);
                    asm volatile("global_store_short %0, %1, off sc1\n\ts_nop 0" ::"v"(address), "v"(uint32_t(word))
                                 : "memory");
                } else
                    *half = uint16_t(word);
            } else if constexpr (COHERENT)
                __hip_atomic_store(ptr, int32_t(word), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            else
                *ptr = int32_t(word);
        }
    }
}

#include <deep_gemm/impls/cdna4_mega_moe_common.cuh>
namespace moe_pointwise {

__device__ __forceinline__ void output_pointers(const MegaMoeArgs &p) {
    using namespace mega_moe;
    const auto &l = p.layout;
    extern __shared__ int scratch[];
    uni_symm::device::fence_acquire<Scope::system>();
    const Rows rows = launch_rows(p, scratch);
    for (int row = rows.begin + blockIdx.x * blockDim.x + threadIdx.x; row < rows.count;
         row += gridDim.x * blockDim.x) {
        if (!rows.valid(row))
            continue;
        uint32_t source = region<const uint32_t>(p, l.l1_records)[int64_t(row) * l.record_ints + l.hidden / 128 + 1];
        const int slot = source % l.num_topk, token = source / l.num_topk % l.num_max_tokens,
                  rank = source / l.num_topk / l.num_max_tokens;
        region<int64_t>(p, l.output_rows)[row] = reinterpret_cast<int64_t>(
            p.view.map(region<uint16_t>(p, l.combine) + (int64_t(slot) * l.num_max_tokens + token) * l.hidden, rank));
    }
}

template <int H, int TopK> __device__ __forceinline__ void reduce(const MegaMoeArgs &p) {
    // For these row widths a wave never straddles tokens. Load its validity mask
    // once, and use scalar buffer bases plus 32-bit offsets for the payload.
    if constexpr (H % 512 == 0 && TopK <= 32) {
        using namespace mega_moe;
        const uint64_t plane = uint64_t(p.layout.num_max_tokens) * H * 2;
        const uint64_t output_bytes = uint64_t(p.num_tokens) * H * 2;
        if (plane * TopK <= 0xffffffffu && output_bytes <= 0xffffffffu) {
            // Peer stores are already released by the preceding cross-rank barrier.
            // SC1 payload reads use system coherence without per-workgroup invalidation.
            asm volatile("" ::: "memory");
            const auto *ids = region<const int64_t>(p, p.layout.topk_idx);
            const auto input = make_rsrc(region<const uint8_t>(p, p.layout.combine), uint32_t(plane * TopK));
            const auto output = make_rsrc(p.y, uint32_t(output_bytes));
            const auto shared = make_rsrc(p.shared_out, p.shared_out ? uint32_t(output_bytes) : 0u);
            for (uint32_t linear = blockIdx.x * blockDim.x + threadIdx.x; linear < uint32_t(p.num_tokens) * (H / 8);
                 linear += gridDim.x * blockDim.x) {
                const int token = __builtin_amdgcn_readfirstlane(linear / (H / 8));
#ifdef CDNA4_WAVE_SLOT_REDUCE
                // Each wave covers 512 channels of one token. The first
                // TopK lanes collect its signed physical IDs in one load.
                const int lane = threadIdx.x % 64;
                const unsigned mask = unsigned(__ballot(lane < TopK && ids[int64_t(token) * TopK + lane] >= 0));
#else
                unsigned mask = 0;
#pragma unroll
                for (int slot = 0; slot < TopK; ++slot)
                    mask |= unsigned(ids[int64_t(token) * TopK + slot] >= 0) << slot;
#endif
                v4i values[TopK];
#if defined(CDNA4_WAVE_SLOT_REDUCE) && CDNA4_WAVE_SLOT_REDUCE == 2
                if (mask == (0xffffffffu >> (32 - TopK))) {
#pragma unroll
                    for (int slot = 0; slot < TopK; ++slot)
                        values[slot] = __builtin_amdgcn_raw_buffer_load_b128(input, linear * 16, uint32_t(slot * plane), 16);
                } else
#endif
                {
#pragma unroll
                    for (int slot = 0; slot < TopK; ++slot) {
#if defined(CDNA4_WAVE_SLOT_REDUCE) && CDNA4_WAVE_SLOT_REDUCE == 3
                        // A disabled slot lies outside the bounded descriptor,
                        // so the hardware supplies the same zero operand.
                        const uint32_t offset = (mask & (1u << slot)) ? linear * 16 + uint32_t(slot * plane) : 0xffffffffu;
                        values[slot] = __builtin_amdgcn_raw_buffer_load_b128(input, offset, 0, 16);
#else
                        values[slot] = (mask & (1u << slot))
                                           ? __builtin_amdgcn_raw_buffer_load_b128(input, linear * 16, uint32_t(slot * plane), 16)
                                           : v4i{};
#endif
                    }
                }
                float sum[8] = {};
#pragma unroll
                for (int slot = 0; slot < TopK; ++slot) {
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        float a, b;
                        unpack_bf16x2(uint32_t(values[slot][i]), a, b);
                        sum[2 * i] += a;
                        sum[2 * i + 1] += b;
                    }
                }
                if (p.shared_out) {
                    const v4i value = __builtin_amdgcn_raw_buffer_load_b128(shared, linear * 16, 0, 0);
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        float a, b;
                        unpack_bf16x2(uint32_t(value[i]), a, b);
                        sum[2 * i] += a;
                        sum[2 * i + 1] += b;
                    }
                }
                v4u packed;
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    packed[i] = pack_bf16x2(sum[2 * i], sum[2 * i + 1]);
                buffer_store16<0>(packed, output, linear * 16, 0);
            }
            return;
        }
    }
    using namespace mega_moe;
    const auto &l = p.layout;
    const auto *ids = region<const int64_t>(p, l.topk_idx);
    const auto *input = region<const uint16_t>(p, l.combine);
    uni_symm::device::fence_acquire<Scope::system>();
    for (int64_t linear = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; linear < int64_t(p.num_tokens) * (H / 8);
         linear += int64_t(gridDim.x) * blockDim.x) {
        const int t = linear / (H / 8), h = (linear % (H / 8)) * 8;
        v4i values[TopK];
#pragma unroll
        for (int slot = 0; slot < TopK; ++slot) {
            const bool valid = ids[int64_t(t) * TopK + slot] >= 0;
            values[slot] =
                valid ? *reinterpret_cast<const v4i *>(input + (int64_t(slot) * l.num_max_tokens + t) * H + h) : v4i{};
        }
        __builtin_amdgcn_sched_barrier(0);
        float acc[8] = {};
#pragma unroll
        for (int slot = 0; slot < TopK; ++slot) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                float a, b;
                unpack_bf16x2(uint32_t(values[slot][i]), a, b);
                acc[2 * i] += a;
                acc[2 * i + 1] += b;
            }
        }
        if (p.shared_out) {
            const v4i v = *reinterpret_cast<const v4i *>(p.shared_out + int64_t(t) * H + h);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                float a, b;
                unpack_bf16x2(uint32_t(v[i]), a, b);
                acc[2 * i] += a;
                acc[2 * i + 1] += b;
            }
        }
        v4i out;
#pragma unroll
        for (int i = 0; i < 4; ++i)
            out[i] = pack_bf16x2(acc[2 * i], acc[2 * i + 1]);
        *reinterpret_cast<v4i *>(p.y + linear * 8) = out;
    }
}

} // namespace moe_pointwise
