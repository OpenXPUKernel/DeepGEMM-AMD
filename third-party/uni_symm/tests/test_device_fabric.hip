// Multi-process test of the uni_symm C++ host API and device fabric on ROCm: one process per GPU
// allocates, exports and maps every peer's buffer, then
//   - a workgroup barrier stress: many barriers in one kernel (and over launches), each checking
//     the ring value written before it,
//   - a multi-workgroup exchange: blocks write a chunk to the next rank and release-add its
//     counter; the next rank acquires the counter and reads the chunk with plain loads,
//   - all-to-all system-scope atomics.
//
//   hipcc -std=c++17 -O2 --offload-arch=gfx950 -I include tests/test_device_fabric.hip -o fabric_test
//   ./fabric_test [ranks]
#include <hip/hip_runtime.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <memory>
#include <new>
#include <vector>

#include <uni_symm/backend/hip/backend.hpp>
#include <uni_symm/device/fabric.hpp>
#include <uni_symm/memory.hpp>

using namespace uni_symm;
using device::Scope;

constexpr int kMax = 16;
constexpr int kChunk = 1 << 18;  // ints per rank and exchange
constexpr int kBlocks = 256, kThreads = 256;

// symmetric buffer layout (byte offsets)
constexpr size_t kSignals = 0, kCounter = 64, kRing = 128, kExchangeCounter = 4096, kAtomics = 8192,
                 kData = 1 << 16, kBytes = kData + size_t(kChunk) * 4;

__global__ void barrier_stress(SymmView<kMax> view, int iterations, int launch, int* errors) {
    auto* base = reinterpret_cast<uint8_t*>(view.base);
    auto* signals = reinterpret_cast<int*>(base + kSignals);
    auto* counter = reinterpret_cast<uint32_t*>(base + kCounter);
    auto* ring = reinterpret_cast<int*>(base + kRing);
    const int next = (view.rank + 1) % view.world_size;
    for (int it = 0; it < iterations; ++it) {
        const int value = launch * iterations + it + 1;
        if (threadIdx.x == 0)
            device::store_relaxed<Scope::system>(view.map(ring, next), value);
        device::barrier_workgroup(view, counter, signals, threadIdx.x);
        if (threadIdx.x == 0 && device::load_relaxed<Scope::system>(ring) != value)
            atomicAdd(errors, 1);
        device::barrier_workgroup(view, counter, signals, threadIdx.x);
    }
}

__global__ void exchange(SymmView<kMax> view, int epoch, int* errors) {
    auto* base = reinterpret_cast<uint8_t*>(view.base);
    const int next = (view.rank + 1) % view.world_size, prev = (view.rank + view.world_size - 1) % view.world_size;
    int* data = reinterpret_cast<int*>(base + kData);
    int* counter = reinterpret_cast<int*>(base + kExchangeCounter);
    // produce into the next rank
    int* remote = view.map(data, next);
    for (int i = blockIdx.x * kThreads + threadIdx.x; i < kChunk; i += kBlocks * kThreads)
        remote[i] = view.rank * 7919 + i * 3 + epoch;
    __syncthreads();
    if (threadIdx.x == 0)
        device::atomic_add_release<Scope::system>(view.map(counter, next), 1);
    // consume what the previous rank produced: every workgroup acquires, then plain loads
    if (threadIdx.x == 0)
        device::wait_until([&] { return device::load_acquire<Scope::system>(counter) >= epoch * kBlocks; },
                           [&] { printf("exchange timeout on rank %d\n", view.rank); });
    __syncthreads();
    device::fence_acquire<Scope::system>();
    int bad = 0;
    for (int i = blockIdx.x * kThreads + threadIdx.x; i < kChunk; i += kBlocks * kThreads)
        bad += data[i] != prev * 7919 + i * 3 + epoch;
    if (bad)
        atomicAdd(errors, bad);
}

__global__ void all_to_all_atomics(SymmView<kMax> view) {
    auto* atomics = reinterpret_cast<unsigned long long*>(reinterpret_cast<uint8_t*>(view.base) + kAtomics);
    for (int peer = 0; peer < view.world_size; ++peer)
        device::atomic_add<Scope::system>(view.map(atomics, peer), 1ull);
}

struct Shared {
    std::atomic<int> arrived[8];
    int pid[kMax], fd[kMax];
    size_t bytes[kMax], alignment[kMax];
};

static void host_barrier(Shared* s, int phase, int ranks) {
    s->arrived[phase].fetch_add(1);
    while (s->arrived[phase].load() < ranks)
        ;
}

static int run(int rank, int ranks, Shared* s) {
    US_HIP_CHECK(hipSetDevice(rank));
    LocalAllocation<HIP> local(kBytes, 0, rank);
    s->pid[rank] = getpid(), s->fd[rank] = local.fd(), s->bytes[rank] = local.bytes(),
    s->alignment[rank] = local.alignment();
    host_barrier(s, 0, ranks);
    std::vector<std::unique_ptr<PeerMapping<HIP>>> peers(ranks);
    std::vector<int64_t> pointers(ranks);
    for (int r = 0; r < ranks; ++r) {
        if (r != rank)
            peers[r] = std::make_unique<PeerMapping<HIP>>(s->pid[r], s->fd[r], s->bytes[r], s->alignment[r], rank);
        pointers[r] = reinterpret_cast<int64_t>(r == rank ? local.ptr() : peers[r]->ptr());
    }
    const auto view = SymmView<kMax>::make(pointers, rank);
    host_barrier(s, 1, ranks);

    int* errors;
    US_HIP_CHECK(hipMalloc(&errors, 4));
    US_HIP_CHECK(hipMemset(errors, 0, 4));
    const int iterations = 1000;
    for (int launch = 0; launch < 3; ++launch)
        barrier_stress<<<1, 64>>>(view, iterations, launch, errors);
    US_HIP_CHECK(hipDeviceSynchronize());
    int barrier_errors = 0, exchange_errors = 0;
    US_HIP_CHECK(hipMemcpy(&barrier_errors, errors, 4, hipMemcpyDeviceToHost));

    US_HIP_CHECK(hipMemset(errors, 0, 4));
    for (int epoch = 1; epoch <= 4; ++epoch) {
        exchange<<<kBlocks, kThreads>>>(view, epoch, errors);
        US_HIP_CHECK(hipDeviceSynchronize());
        host_barrier(s, 1 + epoch, ranks);  // the next epoch overwrites the chunks
    }
    US_HIP_CHECK(hipMemcpy(&exchange_errors, errors, 4, hipMemcpyDeviceToHost));

    all_to_all_atomics<<<64, 256>>>(view);
    US_HIP_CHECK(hipDeviceSynchronize());
    host_barrier(s, 6, ranks);
    unsigned long long atomics = 0;
    US_HIP_CHECK(hipMemcpy(&atomics, static_cast<uint8_t*>(local.ptr()) + kAtomics, 8, hipMemcpyDeviceToHost));
    const bool ok = barrier_errors == 0 && exchange_errors == 0 && atomics == 64ull * 256 * ranks;
    printf("rank %d: barrier errors %d, exchange errors %d, atomics %llu%s\n", rank, barrier_errors, exchange_errors,
           atomics, ok ? "" : "  FAILED");
    host_barrier(s, 7, ranks);
    return ok ? 0 : 1;
}

int main(int argc, char** argv) {
    const int ranks = argc > 1 ? atoi(argv[1]) : 8;
    if (ranks < 2 || ranks > kMax) {
        printf("2 to %d ranks\n", kMax);
        return 1;
    }
    auto* shared = static_cast<Shared*>(mmap(nullptr, sizeof(Shared), PROT_READ | PROT_WRITE,
                                             MAP_SHARED | MAP_ANONYMOUS, -1, 0));
    new (shared) Shared{};
    for (int r = 0; r < ranks; ++r)
        if (fork() == 0) {
            int code = 1;
            try {
                code = run(r, ranks, shared);
            } catch (const std::exception& e) {
                printf("rank %d: %s\n", r, e.what());
            }
            fflush(stdout);
            _exit(code);
        }
    int failed = 0, status = 0;
    for (int r = 0; r < ranks; ++r) {
        wait(&status);
        failed |= !WIFEXITED(status) || WEXITSTATUS(status) != 0;
    }
    printf(failed ? "FAILED\n" : "OK\n");
    return failed;
}
