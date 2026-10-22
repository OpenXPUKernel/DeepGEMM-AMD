#pragma once

#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>

#include <uni_symm/transport/fd.hpp>
#include <uni_symm/utils/exception.hpp>

namespace uni_symm {

// Cleanup in destructors (possibly at process exit, after the runtime shut down) is best effort
template <typename Function>
inline void release_quietly(const Function& function) noexcept {
    try {
        function();
    } catch (...) {
    }
}

// A physical allocation of this process on `device`, mapped at `ptr` and exportable to the peer
// processes of the node as a file descriptor. Its tail holds the signal pad (zeroed).
template <typename Backend>
class LocalAllocation {
public:
    LocalAllocation(const size_t requested_bytes, const size_t signal_pad_bytes, const int device)
        : device_(device), requested_bytes_(requested_bytes), signal_pad_bytes_(signal_pad_bytes) {
        alignment_ = Backend::granularity(device);
        bytes_ = (requested_bytes + signal_pad_bytes + alignment_ - 1) / alignment_ * alignment_;
        handle_ = Backend::create(bytes_, device);
        ptr_ = Backend::map(handle_, bytes_, alignment_, device);
        Backend::zero(ptr_, bytes_);
        fd_ = Backend::export_fd(handle_);
    }

    ~LocalAllocation() {
        if (fd_ >= 0)
            close(fd_);
        release_quietly([&] { Backend::unmap(ptr_, bytes_); });
        release_quietly([&] { Backend::release(handle_); });
    }

    LocalAllocation(const LocalAllocation&) = delete;
    LocalAllocation& operator=(const LocalAllocation&) = delete;

    int device() const { return device_; }
    void* ptr() const { return ptr_; }
    size_t bytes() const { return bytes_; }
    size_t alignment() const { return alignment_; }
    size_t requested_bytes() const { return requested_bytes_; }
    size_t signal_pad_offset() const { return bytes_ - signal_pad_bytes_; }
    size_t signal_pad_bytes() const { return signal_pad_bytes_; }
    int fd() const { return fd_; }

private:
    int device_;
    size_t requested_bytes_, signal_pad_bytes_, bytes_ = 0, alignment_ = 0;
    typename Backend::allocation_handle_t handle_{};
    void* ptr_ = nullptr;
    int fd_ = -1;
};

// A peer process's allocation imported from its descriptor and mapped for `device`
template <typename Backend>
class PeerMapping {
public:
    PeerMapping(const int pid, const int fd, const size_t bytes, const size_t alignment, const int device)
        : bytes_(bytes) {
        const int local_fd = duplicate_peer_fd(pid, fd);
        handle_ = Backend::import_fd(local_fd);
        close(local_fd);
        ptr_ = Backend::map(handle_, bytes, alignment, device);
    }

    ~PeerMapping() {
        release_quietly([&] { Backend::unmap(ptr_, bytes_); });
        release_quietly([&] { Backend::release(handle_); });
    }

    PeerMapping(const PeerMapping&) = delete;
    PeerMapping& operator=(const PeerMapping&) = delete;

    void* ptr() const { return ptr_; }
    size_t bytes() const { return bytes_; }

private:
    size_t bytes_;
    typename Backend::allocation_handle_t handle_{};
    void* ptr_ = nullptr;
};

// The live local allocations by base address, to find the allocation behind a tensor view
template <typename Backend>
class AllocationRegistry {
public:
    static AllocationRegistry& instance() {
        static AllocationRegistry registry;
        return registry;
    }

    void add(const std::shared_ptr<LocalAllocation<Backend>>& allocation) {
        std::lock_guard<std::mutex> lock(mutex_);
        allocations_[reinterpret_cast<uintptr_t>(allocation->ptr())] = allocation;
    }

    void remove(void* base) {
        std::lock_guard<std::mutex> lock(mutex_);
        allocations_.erase(reinterpret_cast<uintptr_t>(base));
    }

    // the allocation containing `ptr`, or null
    std::shared_ptr<LocalAllocation<Backend>> find(const void* ptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto address = reinterpret_cast<uintptr_t>(ptr);
        auto it = allocations_.upper_bound(address);
        if (it == allocations_.begin())
            return nullptr;
        --it;
        const auto& allocation = it->second;
        return address < it->first + allocation->bytes() ? allocation : nullptr;
    }

private:
    std::mutex mutex_;
    std::map<uintptr_t, std::shared_ptr<LocalAllocation<Backend>>> allocations_;
};

}  // namespace uni_symm
