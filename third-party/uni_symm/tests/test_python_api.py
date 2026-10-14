"""Multi-process test of the uni_symm Python API on every GPU of the node (bound to the host
library given by UNI_SYMM_HOST_MODULE, deep_gemm._C by default): allocation, rendezvous, peer
views, the stream-ordered barrier, and data visibility in both directions (remote writes read
locally, local writes read remotely), repeated.

    python tests/test_python_api.py [--ranks N]
"""
import argparse
import importlib
import os
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank: int, world_size: int, store_path: str) -> None:
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    torch.cuda.set_device(rank)
    # the rendezvous exchanges host objects only: gloo (RCCL needs a large /dev/shm), with a file
    # store (no host name lookups)
    dist.init_process_group("gloo", init_method=f"file://{store_path}", rank=rank, world_size=world_size)
    host = importlib.import_module(os.environ.get("UNI_SYMM_HOST_MODULE", "deep_gemm._C"))
    import uni_symm
    uni_symm.bind(host)

    numel = 1 << 22
    buffer = uni_symm.empty(2, numel, dtype=torch.int32, device="cuda")
    assert buffer.shape == (2, numel) and buffer.dtype == torch.int32 and torch.count_nonzero(buffer) == 0
    handle = uni_symm.rendezvous(buffer, dist.group.WORLD)
    assert handle.rank == rank and handle.world_size == world_size and len(handle.buffer_ptrs) == world_size
    assert handle.buffer_ptrs[rank] == buffer.data_ptr()
    assert handle.buffer_ptrs_dev.tolist() == handle.buffer_ptrs

    nxt, prv = (rank + 1) % world_size, (rank - 1) % world_size
    base = torch.arange(numel, dtype=torch.int32, device="cuda")
    for it in range(5):
        # row 0: written remotely by the previous rank; row 1: written locally, read by the next rank
        handle.get_buffer(nxt, (numel, ), torch.int32).copy_(base + 1000 * rank + it)
        buffer[1].copy_(base * 3 + rank + it)
        handle.barrier()
        assert torch.equal(buffer[0], base + 1000 * prv + it), f"rank {rank}: remote writes, iteration {it}"
        remote = handle.get_buffer(nxt, (2, numel), torch.int32)[1].clone()
        assert torch.equal(remote, base * 3 + nxt + it), f"rank {rank}: remote reads, iteration {it}"
        handle.barrier(channel=1)

    # a second allocation and a view at an offset of it
    other = uni_symm.empty(1000, dtype=torch.float32)
    other_handle = uni_symm.rendezvous(other[10:])
    assert other_handle.buffer_ptrs[rank] == other.data_ptr() + 40
    other.fill_(rank)
    other_handle.barrier()
    assert torch.all(other_handle.get_buffer(prv, (990, ), torch.float32) == prv)
    other_handle.barrier()
    torch.cuda.synchronize()
    if rank == 0:
        print(f"uni_symm ({uni_symm.backend()}) Python API: {world_size} ranks OK", flush=True)
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranks", type=int, default=torch.cuda.device_count())
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(worker, args=(args.ranks, os.path.join(directory, "store")), nprocs=args.ranks, join=True)


if __name__ == "__main__":
    main()
