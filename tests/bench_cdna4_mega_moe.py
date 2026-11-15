"""Measure a complete AMD Mega MoE call including shared experts and communication."""

import argparse
import json
import os
import statistics
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_cdna4_mega_moe import quantize_rows


def worker(rank, world, store, args):
    import deep_gemm
    from deep_gemm.testing.moe import reference_mega_moe
    from deep_gemm.utils.math import per_token_cast_to_fp8

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "gloo", init_method=f"file://{store}", rank=rank, world_size=world
    )
    torch.manual_seed(17 + rank)
    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        dist.group.WORLD,
        args.experts,
        args.tokens,
        args.topk,
        args.hidden,
        args.inter,
        num_shared_experts=1,
        mma_type="fp8xfp4",
    )
    local = args.experts // world

    def weights(shape, kind):
        q, sf, _ = quantize_rows(torch.randn(shape, device="cuda") * 0.05, kind)
        return q, sf

    l1, l2 = deep_gemm.transform_weights_for_mega_moe(
        weights((local, 2 * args.inter, args.hidden), "fp4"),
        weights((local, args.hidden, args.inter), "fp4"),
    )
    l1, l2 = deep_gemm.prepare_weights_for_resident_mega_moe(l1, l2)
    s1, s2 = deep_gemm.transform_weights_for_mega_moe(
        weights((2 * args.inter, args.hidden), "fp8"),
        weights((args.hidden, args.inter), "fp8"),
    )
    x = torch.randn((args.tokens, args.hidden), device="cuda")
    q, sf = per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    buffer.x[: args.tokens].copy_(q)
    buffer.x_sf[: args.tokens].copy_(sf)
    scores, ids = torch.randn((args.tokens, args.experts), device="cuda").topk(
        args.topk, dim=-1
    )
    buffer.topk_idx[: args.tokens].copy_(ids)
    buffer.topk_weights[: args.tokens].copy_(scores.softmax(-1))
    y = torch.empty_like(x, dtype=torch.bfloat16)
    expected = torch.empty_like(y)
    kwargs = dict(shared_l1_weights=s1, shared_l2_weights=s2, activation_clamp=10)
    reference_mega_moe(expected, l1, l2, buffer, **kwargs)

    def run():
        deep_gemm.fp8_fp4_mega_moe(y, l1, l2, buffer, **kwargs)

    run()
    torch.cuda.synchronize()
    assert torch.equal(y, expected)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(args.warmup):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(args.samples):
        dist.barrier()
        begin, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        begin.record()
        for _ in range(args.iterations):
            graph.replay()
        end.record()
        end.synchronize()
        row = [None] * world
        dist.all_gather_object(row, begin.elapsed_time(end) * 1000 / args.iterations)
        samples.append(row)
    assert torch.equal(y, expected)
    if rank == 0:
        print(
            json.dumps(
                dict(
                    rank_max_median_us=statistics.median(max(s) for s in samples),
                    samples_us=samples,
                    ranks=world,
                    tokens_per_rank=args.tokens,
                    hidden=args.hidden,
                    intermediate=args.inter,
                    experts=args.experts,
                    topk=args.topk,
                    shared_experts=1,
                )
            )
        )
    buffer.destroy()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranks", type=int, default=torch.cuda.device_count())
    for name, default in (
        ("tokens", 512),
        ("hidden", 1024),
        ("inter", 512),
        ("experts", 48),
        ("topk", 6),
        ("warmup", 2000),
        ("iterations", 100),
        ("samples", 9),
    ):
        parser.add_argument("--" + name, type=int, default=default)
    args = parser.parse_args()
    if args.ranks < 1 or args.experts % args.ranks:
        parser.error("experts must divide evenly across a positive number of ranks")
    with tempfile.TemporaryDirectory() as folder:
        mp.spawn(
            worker,
            args=(args.ranks, os.path.join(folder, "store"), args),
            nprocs=args.ranks,
            join=True,
        )


if __name__ == "__main__":
    main()
