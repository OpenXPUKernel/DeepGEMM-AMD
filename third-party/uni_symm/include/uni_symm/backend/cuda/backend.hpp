#pragma once

#include <cuda.h>

#include <cstddef>
#include <cstdint>

#include <uni_symm/utils/exception.hpp>
#include <uni_symm/utils/lazy_library.hpp>

namespace uni_symm {

namespace cuda_driver {

inline LazyLibrary& library() {
    static LazyLibrary library("libcuda.so.1");
    return library;
}

US_DECL_LAZY_FUNCTION(library, cuGetErrorString)
US_DECL_LAZY_FUNCTION(library, cuCtxGetDevice)
US_DECL_LAZY_FUNCTION(library, cuMemGetAllocationGranularity)
US_DECL_LAZY_FUNCTION(library, cuMemCreate)
US_DECL_LAZY_FUNCTION(library, cuMemExportToShareableHandle)
US_DECL_LAZY_FUNCTION(library, cuMemImportFromShareableHandle)
US_DECL_LAZY_FUNCTION(library, cuMemAddressReserve)
US_DECL_LAZY_FUNCTION(library, cuMemMap)
US_DECL_LAZY_FUNCTION(library, cuMemSetAccess)
US_DECL_LAZY_FUNCTION(library, cuMemUnmap)
US_DECL_LAZY_FUNCTION(library, cuMemAddressFree)
US_DECL_LAZY_FUNCTION(library, cuMemRelease)
US_DECL_LAZY_FUNCTION(library, cuMemsetD8_v2)
US_DECL_LAZY_FUNCTION(library, cuCtxSynchronize)
US_DECL_LAZY_FUNCTION(library, cuStreamWriteValue32_v2)
US_DECL_LAZY_FUNCTION(library, cuStreamWaitValue32_v2)

inline void check(const CUresult error, const char* expression) {
    if (error == CUDA_SUCCESS)
        return;
    const char* description = "unknown";
    lazy_cuGetErrorString(error, &description);
    panic(expression, " failed with CUDA error ", static_cast<int>(error), ": ", description);
}

}  // namespace cuda_driver

#define US_CUDA_CHECK(expr) ::uni_symm::cuda_driver::check((expr), #expr)

// CUDA backend: the same virtual memory management flow through the CUDA driver API (the
// allocations torch.distributed._symmetric_memory makes), resolved from libcuda at first use
struct CUDA {
    static constexpr const char* name = "cuda";
    using allocation_handle_t = CUmemGenericAllocationHandle;
    using stream_t = CUstream;

    static int current_device() {
        CUdevice device = 0;
        US_CUDA_CHECK(cuda_driver::lazy_cuCtxGetDevice(&device));
        return static_cast<int>(device);
    }

    static CUmemAllocationProp properties(const int device) {
        CUmemAllocationProp prop{};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device;
        return prop;
    }

    static size_t granularity(const int device) {
        size_t granularity = 0;
        const auto prop = properties(device);
        US_CUDA_CHECK(cuda_driver::lazy_cuMemGetAllocationGranularity(&granularity, &prop,
                                                                      CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
        return granularity;
    }

    static allocation_handle_t create(const size_t bytes, const int device) {
        allocation_handle_t handle{};
        const auto prop = properties(device);
        US_CUDA_CHECK(cuda_driver::lazy_cuMemCreate(&handle, bytes, &prop, 0));
        return handle;
    }

    static int export_fd(const allocation_handle_t handle) {
        int fd = -1;
        US_CUDA_CHECK(cuda_driver::lazy_cuMemExportToShareableHandle(&fd, handle,
                                                                     CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));
        return fd;
    }

    static allocation_handle_t import_fd(const int fd) {
        allocation_handle_t handle{};
        US_CUDA_CHECK(cuda_driver::lazy_cuMemImportFromShareableHandle(
            &handle, reinterpret_cast<void*>(static_cast<intptr_t>(fd)), CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
        return handle;
    }

    static void* map(const allocation_handle_t handle, const size_t bytes, const size_t alignment, const int device) {
        CUdeviceptr ptr = 0;
        US_CUDA_CHECK(cuda_driver::lazy_cuMemAddressReserve(&ptr, bytes, alignment, 0, 0));
        US_CUDA_CHECK(cuda_driver::lazy_cuMemMap(ptr, bytes, 0, handle, 0));
        CUmemAccessDesc access{};
        access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access.location.id = device;
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        US_CUDA_CHECK(cuda_driver::lazy_cuMemSetAccess(ptr, bytes, &access, 1));
        return reinterpret_cast<void*>(ptr);
    }

    static void unmap(void* ptr, const size_t bytes) {
        const auto address = reinterpret_cast<CUdeviceptr>(ptr);
        US_CUDA_CHECK(cuda_driver::lazy_cuMemUnmap(address, bytes));
        US_CUDA_CHECK(cuda_driver::lazy_cuMemAddressFree(address, bytes));
    }

    static void release(const allocation_handle_t handle) { US_CUDA_CHECK(cuda_driver::lazy_cuMemRelease(handle)); }

    static void zero(void* ptr, const size_t bytes) {
        US_CUDA_CHECK(cuda_driver::lazy_cuMemsetD8_v2(reinterpret_cast<CUdeviceptr>(ptr), 0, bytes));
        US_CUDA_CHECK(cuda_driver::lazy_cuCtxSynchronize());
    }

    static void write_value(const stream_t stream, void* ptr, const uint32_t value) {
        US_CUDA_CHECK(cuda_driver::lazy_cuStreamWriteValue32_v2(stream, reinterpret_cast<CUdeviceptr>(ptr), value, 0));
    }

    static void wait_value_geq(const stream_t stream, void* ptr, const uint32_t value) {
        US_CUDA_CHECK(cuda_driver::lazy_cuStreamWaitValue32_v2(stream, reinterpret_cast<CUdeviceptr>(ptr), value,
                                                              CU_STREAM_WAIT_VALUE_GEQ));
    }
};

}  // namespace uni_symm
