"""uni_symm: symmetric memory across the GPUs of a node, with the interface of
torch.distributed._symmetric_memory (``empty``, ``rendezvous`` and a handle holding every rank's
buffer pointer), on ROCm and CUDA.

The primitives are header-only C++ (``uni_symm/python_api.hpp``) that a host extension registers
under its ``uni_symm`` submodule; ``bind`` points this package at them. Kernels address the peers
through ``uni_symm/device/fabric.hpp`` with the pointers of the handle.
"""
import math
from typing import List, Optional, Sequence, Union

import torch
import torch.distributed as dist

__all__ = ["bind", "is_bound", "backend", "empty", "rendezvous", "SymmetricMemory"]

_primitives = None


def bind(module) -> None:
    """Use the primitives a host extension registered (``module.uni_symm``)."""
    global _primitives
    _primitives = getattr(module, "uni_symm", module)


def is_bound() -> bool:
    return _primitives is not None


def _ops():
    if _primitives is None:
        raise RuntimeError("uni_symm is not bound to a host library: call uni_symm.bind(extension_module)")
    return _primitives


def backend() -> str:
    """``"hip"`` or ``"cuda"``."""
    return _ops().backend()


def _device_index(device) -> int:
    if device is None:
        return torch.cuda.current_device()
    device = torch.device(device)
    return torch.cuda.current_device() if device.index is None else device.index


def empty(*size, dtype: torch.dtype = torch.uint8, device=None) -> torch.Tensor:
    """An uninitialized (zero-filled) tensor whose memory can be shared with ``rendezvous``."""
    if len(size) == 1 and isinstance(size[0], (tuple, list, torch.Size)):
        size = tuple(size[0])
    itemsize = torch.empty((), dtype=dtype).element_size()
    raw = _ops().empty(math.prod(size) * itemsize, _device_index(device))
    return raw.view(dtype).view(size)


class SymmetricMemory:
    """Every rank's view of the tensors passed to ``rendezvous``: ``buffer_ptrs[r]`` is rank r's
    tensor mapped into this process, ``signal_pad_ptrs[r]`` its allocation's signal pad (32-bit
    slots, zero-initialized)."""

    def __init__(self, tensor: torch.Tensor, group, rank: int, world_size: int, buffer_ptrs: List[int],
                 signal_pad_ptrs: List[int], mappings: list):
        self.tensor = tensor
        self.group = group
        self.rank = rank
        self.world_size = world_size
        self.buffer_ptrs = buffer_ptrs
        self.signal_pad_ptrs = signal_pad_ptrs
        self.buffer_size = tensor.numel() * tensor.element_size()
        self.signal_pad_size = _ops().signal_pad_bytes()
        self._mappings = mappings
        self._device = tensor.device
        self._epochs = {}
        self._buffer_ptrs_dev = None
        self._signal_pad_ptrs_dev = None

    @property
    def buffer_ptrs_dev(self) -> torch.Tensor:
        if self._buffer_ptrs_dev is None:
            self._buffer_ptrs_dev = torch.tensor(self.buffer_ptrs, dtype=torch.int64, device=self._device)
        return self._buffer_ptrs_dev

    @property
    def signal_pad_ptrs_dev(self) -> torch.Tensor:
        if self._signal_pad_ptrs_dev is None:
            self._signal_pad_ptrs_dev = torch.tensor(self.signal_pad_ptrs, dtype=torch.int64, device=self._device)
        return self._signal_pad_ptrs_dev

    def get_buffer(self, rank: int, sizes: Sequence[int], dtype: torch.dtype, storage_offset: int = 0) -> torch.Tensor:
        """A tensor of ``sizes`` and ``dtype`` at element ``storage_offset`` of rank ``rank``'s buffer."""
        itemsize = torch.empty((), dtype=dtype).element_size()
        offset, nbytes = storage_offset * itemsize, math.prod(sizes) * itemsize
        if offset + nbytes > self.buffer_size:
            raise ValueError("the view is outside the buffer")
        if rank == self.rank:
            raw = self.tensor.view(-1).view(torch.uint8)[offset:offset + nbytes]
        else:
            raw = _ops().peer_tensor(self._mappings[rank], self.buffer_ptrs[rank] + offset, nbytes,
                                     self._device.index)
        return raw.view(dtype).view(tuple(sizes))

    def barrier(self, channel: int = 0) -> None:
        """A barrier of the ranks ordered on the current stream: earlier work on every rank's
        stream completes before later work on any. The epochs are host-side, so a captured
        CUDA graph must not replay it; kernels use ``uni_symm::device::barrier_workgroup``."""
        slots = self.signal_pad_size // 4
        if (channel + 1) * self.world_size > slots:
            raise ValueError(f"channel {channel} is outside the signal pad")
        epoch = self._epochs.get(channel, 0) + 1
        self._epochs[channel] = epoch
        stream = torch.cuda.current_stream(self._device).cuda_stream
        for r in range(self.world_size):
            _ops().write_value(stream, self.signal_pad_ptrs[r] + 4 * (channel * self.world_size + self.rank), epoch)
        for r in range(self.world_size):
            _ops().wait_value_geq(stream, self.signal_pad_ptrs[self.rank] + 4 * (channel * self.world_size + r), epoch)


def rendezvous(tensor: torch.Tensor, group: Optional[Union[dist.ProcessGroup, str]] = None) -> SymmetricMemory:
    """Exchange the tensors (from ``empty``) of every rank of ``group`` (collective)."""
    if isinstance(group, str):
        group = dist.distributed_c10d._resolve_process_group(group)
    group = group if group is not None else dist.group.WORLD
    world_size = dist.get_world_size(group) if dist.is_initialized() else 1
    rank = dist.get_rank(group) if dist.is_initialized() else 0
    handle = tuple(_ops().export_handle(tensor))
    handles = [handle]
    if world_size > 1:
        handles = [None] * world_size
        dist.all_gather_object(handles, handle, group=group)
    device = tensor.device.index
    mappings, buffer_ptrs, signal_pad_ptrs = [], [], []
    for r, (pid, fd, nbytes, alignment, offset, signal_pad_offset) in enumerate(handles):
        if r == rank:
            mapping, base = None, tensor.data_ptr() - offset
        else:
            mapping = _ops().import_handle(pid, fd, nbytes, alignment, device)
            base = mapping.ptr
        mappings.append(mapping)
        buffer_ptrs.append(base + offset)
        signal_pad_ptrs.append(base + signal_pad_offset)
    if world_size > 1:
        # the peers' descriptors stay open while their allocations live; this only keeps the
        # ranks together until every mapping exists
        dist.barrier(group=group)
    return SymmetricMemory(tensor, group, rank, world_size, buffer_ptrs, signal_pad_ptrs, mappings)
