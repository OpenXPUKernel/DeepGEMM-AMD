# uni_symm

uni_symm is a header-only symmetric memory library for the GPUs of a node, on ROCm and CUDA. Every
rank allocates a buffer, the ranks exchange them once, and kernels then read, write and
synchronize any rank's buffer directly over the GPU fabric (xGMI, NVLink). The Python interface is
the one of `torch.distributed._symmetric_memory`, so code written against PyTorch's symmetric memory
(for example DeepGEMM's Mega MoE) switches by module.

Like DeepJIT, uni_symm is embedded into a host extension: the extension registers the primitives
into its pybind11 module, and kernels include the device fabric header.

## Layers

| Path | Contents |
| --- | --- |
| `include/uni_symm/backend/hip/`, `backend/cuda/` | A backend: virtual memory allocations exportable as POSIX file descriptors, their import and mapping, stream-ordered signal writes and waits. HIP calls the runtime the host library links; CUDA resolves the driver API from `libcuda.so.1` at first use. |
| `include/uni_symm/memory.hpp` | `LocalAllocation` (exportable, with a zeroed signal pad at its tail), `PeerMapping` (a peer's allocation mapped for this device), and the registry that finds the allocation behind a tensor view. |
| `include/uni_symm/transport/fd.hpp` | Descriptor transport between the processes of a node (`pidfd_getfd`). |
| `include/uni_symm/python_api.hpp` | `register_python_api<Backend>(module)`: the primitives under `module.uni_symm`. |
| `include/uni_symm/device/fabric.hpp` | Device fabric for HIP, HIPRTC, NVCC and NVRTC: `SymmView` (peer addressing), loads, stores and atomics with an explicit scope and order, fences, `wait_until` with a timeout, and a cross-rank workgroup barrier. |
| `uni_symm/` | The Python package: `bind`, `empty`, `rendezvous`, `SymmetricMemory`. |

## Python

```python
import uni_symm
uni_symm.bind(host_extension)           # the module that called register_python_api

buffer = uni_symm.empty(num_bytes, dtype=torch.int8, device="cuda")
handle = uni_symm.rendezvous(buffer, group)   # collective over `group`
handle.buffer_ptrs        # every rank's buffer, mapped into this process
handle.buffer_ptrs_dev    # the same, as a device tensor
handle.signal_pad_ptrs    # every rank's signal pad (32-bit slots, zero-initialized)
handle.get_buffer(rank, sizes, dtype, storage_offset=0)   # a tensor on a rank's buffer
handle.barrier(channel=0) # stream-ordered barrier of the ranks
```

`rendezvous` exchanges the handles with `all_gather_object`, so any process group backend works
(gloo included). `barrier` writes and waits on the signal pads from the stream; its epochs are kept
on the host, so it is not for CUDA graph replay. Kernels synchronize with the device barrier
instead.

## Device fabric

```cpp
#include <uni_symm/device/fabric.hpp>
using namespace uni_symm;

__global__ void kernel(SymmView<kMaxRanks> view, ...) {
    int* remote = view.map(local_ptr, peer);                      // the same offset on `peer`
    device::atomic_add_release<device::Scope::system>(remote_counter, 1);
    device::wait_until([&] { return device::load_acquire(counter) == target; }, on_timeout);
    device::barrier_workgroup(view, local_counter, signals, threadIdx.x);
}
```

The host builds the view with `SymmView<N>::make(handle.buffer_ptrs, rank)`. A release at system
scope orders the thread's earlier writes (its workgroup's after a workgroup barrier) for every rank;
data written by peers is read after an acquire by the reading workgroup. The barrier alternates two
signal words and flips the sign of their updates every second use, so the words return to zero and
no reset is needed between launches.

## Requirements

Linux 5.6+ (`pidfd_getfd`); the ranks of a node run as the same user, with Yama `ptrace_scope` 0
(or a ptracer grant). ROCm 6.2+ with virtual memory management, or CUDA 12. Inter-node fabrics
(NVLink multi-node fabric handles, RDMA through mori or rocSHMEM) are a transport to add behind the
same interface.

## Tests

```bash
# Python API on every GPU of the node, bound to DeepGEMM's extension
python tests/test_python_api.py
# C++ host API and device fabric (ROCm)
hipcc -std=c++17 -O2 --offload-arch=gfx950 -I include tests/test_device_fabric.hip -o fabric_test && ./fabric_test 8
```

On ROCm both pass on 8 GPUs (xGMI). The CUDA backend and the CUDA device fabric are compile-checked
(the host backend with the CUDA 12.9 headers, the fabric with NVRTC for SM90), not yet run.
