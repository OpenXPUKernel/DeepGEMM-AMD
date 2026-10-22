#pragma once

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include <uni_symm/utils/exception.hpp>

namespace uni_symm {

inline void check_hip(const hipError_t error, const char* expression) {
    if (error != hipSuccess)
        panic(expression, " failed with HIP error ", static_cast<int>(error), ": ", hipGetErrorString(error));
}

#define US_HIP_CHECK(expr) ::uni_symm::check_hip((expr), #expr)

// ROCm backend: virtual memory management allocations exported as POSIX file descriptors. The
// host library links the HIP runtime, so the calls bind directly (a second runtime instance
// loaded by name would not share allocations with the first).
struct HIP {
    static constexpr const char* name = "hip";
    using allocation_handle_t = hipMemGenericAllocationHandle_t;
    using stream_t = hipStream_t;

    static int current_device() {
        int device = 0;
        US_HIP_CHECK(hipGetDevice(&device));
        return device;
    }

    static hipMemAllocationProp properties(const int device) {
        hipMemAllocationProp prop{};
        prop.type = hipMemAllocationTypePinned;
        prop.requestedHandleType = hipMemHandleTypePosixFileDescriptor;
        prop.location.type = hipMemLocationTypeDevice;
        prop.location.id = device;
        return prop;
    }

    static size_t granularity(const int device) {
        size_t granularity = 0;
        const auto prop = properties(device);
        US_HIP_CHECK(hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityRecommended));
        return granularity;
    }

    static allocation_handle_t create(const size_t bytes, const int device) {
        allocation_handle_t handle{};
        const auto prop = properties(device);
        US_HIP_CHECK(hipMemCreate(&handle, bytes, &prop, 0));
        return handle;
    }

    static int export_fd(const allocation_handle_t handle) {
        int fd = -1;
        US_HIP_CHECK(hipMemExportToShareableHandle(&fd, handle, hipMemHandleTypePosixFileDescriptor, 0));
        return fd;
    }

    static allocation_handle_t import_fd(const int fd) {
        allocation_handle_t handle{};
        US_HIP_CHECK(hipMemImportFromShareableHandle(&handle, reinterpret_cast<void*>(static_cast<intptr_t>(fd)),
                                                     hipMemHandleTypePosixFileDescriptor));
        return handle;
    }

    // a virtual address range mapping the allocation, readable and writable by `device`
    static void* map(const allocation_handle_t handle, const size_t bytes, const size_t alignment, const int device) {
        void* ptr = nullptr;
        US_HIP_CHECK(hipMemAddressReserve(&ptr, bytes, alignment, nullptr, 0));
        US_HIP_CHECK(hipMemMap(ptr, bytes, 0, handle, 0));
        hipMemAccessDesc access{};
        access.location.type = hipMemLocationTypeDevice;
        access.location.id = device;
        access.flags = hipMemAccessFlagsProtReadWrite;
        US_HIP_CHECK(hipMemSetAccess(ptr, bytes, &access, 1));
        return ptr;
    }

    static void unmap(void* ptr, const size_t bytes) {
        US_HIP_CHECK(hipMemUnmap(ptr, bytes));
        US_HIP_CHECK(hipMemAddressFree(ptr, bytes));
    }

    static void release(const allocation_handle_t handle) { US_HIP_CHECK(hipMemRelease(handle)); }

    static void zero(void* ptr, const size_t bytes) {
        US_HIP_CHECK(hipMemset(ptr, 0, bytes));
        US_HIP_CHECK(hipDeviceSynchronize());
    }

    // stream-ordered 32-bit signal write (any mapped address, peers included) and wait (>= value)
    static void write_value(const stream_t stream, void* ptr, const uint32_t value) {
        US_HIP_CHECK(hipStreamWriteValue32(stream, ptr, value, 0));
    }

    static void wait_value_geq(const stream_t stream, void* ptr, const uint32_t value) {
        US_HIP_CHECK(hipStreamWaitValue32(stream, ptr, value, hipStreamWaitValueGte, 0xffffffffu));
    }
};

}  // namespace uni_symm
