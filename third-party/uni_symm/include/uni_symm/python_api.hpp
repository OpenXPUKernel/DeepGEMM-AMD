#pragma once

#include <unistd.h>

#include <cstdint>
#include <memory>

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <uni_symm/memory.hpp>

namespace uni_symm {

// Signal pad at the tail of every allocation: 32-bit slots [channel][rank]
constexpr size_t kSignalPadBytes = 64 * 1024;

// The primitives under `module.uni_symm`; the `uni_symm` Python package composes empty,
// rendezvous and the barriers from them. Streams are passed as their integer handles.
template <typename Backend>
inline void register_python_api(pybind11::module_& module) {
    namespace py = pybind11;
    using Local = LocalAllocation<Backend>;
    using Peer = PeerMapping<Backend>;
    auto m = module.def_submodule("uni_symm", "uni_symm symmetric memory primitives");

    py::class_<Peer, std::shared_ptr<Peer>>(m, "PeerMapping", py::module_local())
        .def_property_readonly("ptr", [](const Peer& peer) { return reinterpret_cast<int64_t>(peer.ptr()); })
        .def_property_readonly("bytes", &Peer::bytes);

    m.def("backend", [] { return Backend::name; });
    m.def("signal_pad_bytes", [] { return kSignalPadBytes; });

    // a uint8 tensor of `bytes` bytes on `device` backed by an exportable allocation
    m.def("empty", [](const int64_t bytes, const int device) {
        US_ASSERT(bytes >= 0, "negative size");
        auto allocation = std::make_shared<Local>(static_cast<size_t>(bytes), kSignalPadBytes, device);
        AllocationRegistry<Backend>::instance().add(allocation);
        void* base = allocation->ptr();
        return torch::from_blob(base, {bytes}, [base](void*) { AllocationRegistry<Backend>::instance().remove(base); },
                                torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, device));
    });

    // (pid, fd, allocation bytes, alignment, byte offset of the tensor's data, signal pad offset)
    m.def("export_handle", [](const torch::Tensor& tensor) {
        const auto allocation = AllocationRegistry<Backend>::instance().find(tensor.data_ptr());
        US_ASSERT(allocation != nullptr, "the tensor was not allocated by uni_symm.empty");
        const auto offset = static_cast<int64_t>(static_cast<const uint8_t*>(tensor.data_ptr()) -
                                                 static_cast<const uint8_t*>(allocation->ptr()));
        return py::make_tuple(static_cast<int>(getpid()), allocation->fd(), allocation->bytes(),
                              allocation->alignment(), offset, allocation->signal_pad_offset());
    });

    m.def("import_handle", [](const int pid, const int fd, const size_t bytes, const size_t alignment, const int device) {
        return std::make_shared<Peer>(pid, fd, bytes, alignment, device);
    });

    // a uint8 view of `bytes` bytes at `ptr` of a peer mapping, which it keeps alive
    m.def("peer_tensor", [](const std::shared_ptr<Peer>& peer, const int64_t ptr, const int64_t bytes, const int device) {
        US_ASSERT(ptr >= reinterpret_cast<int64_t>(peer->ptr()) &&
                  ptr + bytes <= reinterpret_cast<int64_t>(peer->ptr()) + static_cast<int64_t>(peer->bytes()),
                  "the view is outside the peer mapping");
        return torch::from_blob(reinterpret_cast<void*>(ptr), {bytes}, [peer](void*) {},
                                torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, device));
    });

    m.def("write_value", [](const int64_t stream, const int64_t ptr, const uint32_t value) {
        Backend::write_value(reinterpret_cast<typename Backend::stream_t>(stream), reinterpret_cast<void*>(ptr), value);
    });
    m.def("wait_value_geq", [](const int64_t stream, const int64_t ptr, const uint32_t value) {
        Backend::wait_value_geq(reinterpret_cast<typename Backend::stream_t>(stream), reinterpret_cast<void*>(ptr), value);
    });
}

}  // namespace uni_symm
