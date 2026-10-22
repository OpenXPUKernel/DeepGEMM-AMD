// uni_symm device fabric: peer addressing into the symmetric buffers of a node and the ordered
// memory operations, waits and barriers kernels build their communication from. The same calls
// compile for ROCm (HIP / HIPRTC) and CUDA (NVCC / NVRTC).
//
// Ordering contract: a release operation at system scope orders the issuing thread's earlier
// writes (and, after a workgroup barrier, its workgroup's) before it for every rank; an acquire
// at system scope orders the thread's later reads after it. Data written by peers is read
// after an acquire by the reading workgroup (on ROCm each XCD's L2 is invalidated by its own
// acquires).
#pragma once

#if defined(__HIPCC__) || defined(__HIPCC_RTC__)
#include <hip/hip_runtime.h>
#define US_DEVICE __device__ __forceinline__
#define US_HOST_DEVICE __host__ __device__ __forceinline__
#define UNI_SYMM_HIP 1
#elif defined(__CUDACC__) || defined(__CUDACC_RTC__)
#include <cuda/atomic>
#define US_DEVICE __device__ __forceinline__
#define US_HOST_DEVICE __host__ __device__ __forceinline__
#define UNI_SYMM_CUDA 1
#else
#define US_HOST_DEVICE inline
#endif

#if defined(__CUDACC_RTC__)
#include <cuda/std/cstdint>
namespace uni_symm {
using cuda::std::int64_t;
using cuda::std::uint32_t;
using cuda::std::uint64_t;
}  // namespace uni_symm
#else
#include <cstdint>
#endif

namespace uni_symm {

// Ranks of a node-wide (NVL72-sized) domain
constexpr int kMaxRanks = 72;

// A rank's view of the symmetric buffers: a pointer into this rank's buffer maps to the same
// offset of a peer's buffer
template <int MAX_RANKS = kMaxRanks>
struct SymmView {
    int64_t base;
    int64_t offsets[MAX_RANKS];
    int rank, world_size;

#if !defined(__CUDACC_RTC__) && !defined(__HIPCC_RTC__)
    // from the buffer addresses of every rank (any indexable container of integers; host side)
    template <typename Pointers>
    static SymmView make(const Pointers& pointers, const int rank) {
        SymmView view{};
        view.rank = rank;
        view.world_size = static_cast<int>(pointers.size());
        view.base = static_cast<int64_t>(pointers[rank]);
        for (int i = 0; i < MAX_RANKS; ++i)
            view.offsets[i] = i < view.world_size ? static_cast<int64_t>(pointers[i]) - view.base : 0;
        return view;
    }
#endif

    template <typename T>
    US_HOST_DEVICE T* map(T* ptr, const int peer) const {
        return reinterpret_cast<T*>(reinterpret_cast<int64_t>(ptr) + offsets[peer]);
    }
};

#if defined(UNI_SYMM_HIP) || defined(UNI_SYMM_CUDA)
namespace device {

enum class Scope { device, system };

#if defined(UNI_SYMM_HIP)
namespace detail {
template <Scope S>
constexpr int hip_scope = S == Scope::device ? __HIP_MEMORY_SCOPE_AGENT : __HIP_MEMORY_SCOPE_SYSTEM;
}

template <Scope S = Scope::system, typename T>
US_DEVICE T load_acquire(const T* ptr) {
    return __hip_atomic_load(ptr, __ATOMIC_ACQUIRE, detail::hip_scope<S>);
}
// a coherent read at the scope without ordering (polling, reading peer-written words)
template <Scope S = Scope::system, typename T>
US_DEVICE T load_relaxed(const T* ptr) {
    return __hip_atomic_load(ptr, __ATOMIC_RELAXED, detail::hip_scope<S>);
}
template <Scope S = Scope::system, typename T>
US_DEVICE void store_release(T* ptr, const T value) {
    __hip_atomic_store(ptr, value, __ATOMIC_RELEASE, detail::hip_scope<S>);
}
template <Scope S = Scope::system, typename T>
US_DEVICE void store_relaxed(T* ptr, const T value) {
    __hip_atomic_store(ptr, value, __ATOMIC_RELAXED, detail::hip_scope<S>);
}
template <Scope S = Scope::system, typename T>
US_DEVICE T atomic_add(T* ptr, const T value) {
    return __hip_atomic_fetch_add(ptr, value, __ATOMIC_RELAXED, detail::hip_scope<S>);
}
template <Scope S = Scope::system, typename T>
US_DEVICE T atomic_add_release(T* ptr, const T value) {
    return __hip_atomic_fetch_add(ptr, value, __ATOMIC_RELEASE, detail::hip_scope<S>);
}
template <Scope S = Scope::system, typename T>
US_DEVICE T atomic_add_acq_rel(T* ptr, const T value) {
    return __hip_atomic_fetch_add(ptr, value, __ATOMIC_ACQ_REL, detail::hip_scope<S>);
}
template <Scope S = Scope::system>
US_DEVICE void fence_release() {
    if constexpr (S == Scope::device)
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    else
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
}
template <Scope S = Scope::system>
US_DEVICE void fence_acquire() {
    if constexpr (S == Scope::device)
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    else
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
}
// nanoseconds of a constant-rate clock
US_DEVICE uint64_t clock_ns() { return wall_clock64() * 10; }
US_DEVICE void workgroup_sync() { __syncthreads(); }
[[noreturn]] US_DEVICE void trap() { __builtin_trap(); }
#else
namespace detail {
template <Scope S>
constexpr cuda::thread_scope cuda_scope = S == Scope::device ? cuda::thread_scope_device : cuda::thread_scope_system;
template <Scope S, typename T>
US_DEVICE cuda::atomic_ref<T, cuda_scope<S>> ref(T* ptr) {
    return cuda::atomic_ref<T, cuda_scope<S>>(*ptr);
}
}  // namespace detail

template <Scope S = Scope::system, typename T>
US_DEVICE T load_acquire(const T* ptr) {
    return detail::ref<S>(const_cast<T*>(ptr)).load(cuda::memory_order_acquire);
}
template <Scope S = Scope::system, typename T>
US_DEVICE T load_relaxed(const T* ptr) {
    return detail::ref<S>(const_cast<T*>(ptr)).load(cuda::memory_order_relaxed);
}
template <Scope S = Scope::system, typename T>
US_DEVICE void store_release(T* ptr, const T value) {
    detail::ref<S>(ptr).store(value, cuda::memory_order_release);
}
template <Scope S = Scope::system, typename T>
US_DEVICE void store_relaxed(T* ptr, const T value) {
    detail::ref<S>(ptr).store(value, cuda::memory_order_relaxed);
}
template <Scope S = Scope::system, typename T>
US_DEVICE T atomic_add(T* ptr, const T value) {
    return detail::ref<S>(ptr).fetch_add(value, cuda::memory_order_relaxed);
}
template <Scope S = Scope::system, typename T>
US_DEVICE T atomic_add_release(T* ptr, const T value) {
    return detail::ref<S>(ptr).fetch_add(value, cuda::memory_order_release);
}
template <Scope S = Scope::system, typename T>
US_DEVICE T atomic_add_acq_rel(T* ptr, const T value) {
    return detail::ref<S>(ptr).fetch_add(value, cuda::memory_order_acq_rel);
}
template <Scope S = Scope::system>
US_DEVICE void fence_release() {
    cuda::atomic_thread_fence(cuda::memory_order_release, detail::cuda_scope<S>);
}
template <Scope S = Scope::system>
US_DEVICE void fence_acquire() {
    cuda::atomic_thread_fence(cuda::memory_order_acquire, detail::cuda_scope<S>);
}
US_DEVICE uint64_t clock_ns() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
US_DEVICE void workgroup_sync() { __syncthreads(); }
[[noreturn]] US_DEVICE void trap() { __trap(); }
#endif

// Spin until `pred()` holds; after `timeout_ns`, `on_timeout()` runs and the kernel traps
template <typename Pred, typename OnTimeout>
US_DEVICE void wait_until(const Pred& pred, const OnTimeout& on_timeout, const uint64_t timeout_ns = 60000000000ull) {
    const uint64_t start = clock_ns();
    while (!pred()) {
        if (clock_ns() - start >= timeout_ns) {
            on_timeout();
            trap();
        }
    }
}

// A barrier of one workgroup per rank across the ranks of `view`. `counter` is a word of this
// rank's memory that only this barrier uses (it selects the phase), `signals` two words of the
// symmetric buffer (this rank's addresses), all initially zero. Each phase adds +1 from every
// rank and then -1 every second time, so the words return to zero and no reset is needed; the
// two phases keep a rank from entering the next use of a word before all ranks left its last.
// Needs at least `world_size` threads; the workgroup's earlier writes are released to all
// ranks, and the peers' writes before their arrival are acquired.
template <int MAX_RANKS>
US_DEVICE void barrier_workgroup(const SymmView<MAX_RANKS>& view, uint32_t* counter, int* signals,
                                 const int thread_idx) {
    const uint32_t status = load_relaxed<Scope::device>(counter) & 3u;
    const int phase = static_cast<int>(status & 1u), sign = static_cast<int>(status >> 1);
    int* signal = signals + phase;
    fence_release<Scope::system>();
    workgroup_sync();
    if (thread_idx < view.world_size)
        atomic_add_release<Scope::system>(view.map(signal, thread_idx), sign ? -1 : 1);
    if (thread_idx == 0) {
        store_relaxed<Scope::device>(counter, load_relaxed<Scope::device>(counter) + 1u);
        const int target = sign ? 0 : view.world_size;
        wait_until([&] { return load_acquire<Scope::system>(signal) == target; },
                   [&] { printf("uni_symm barrier timeout: rank %d, signal %d, target %d\n", view.rank,
                                load_relaxed<Scope::system>(signal), target); });
    }
    workgroup_sync();
    fence_acquire<Scope::system>();
}

}  // namespace device
#endif

}  // namespace uni_symm
