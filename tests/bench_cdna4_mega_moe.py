"""Benchmark the CDNA4 Mega MoE on the default shapes of tests/test_mega_moe.py (one process per GPU):
the time per call, the TFLOPS of the routed and shared expert GEMMs, and the fabric bytes per
second (each received token's hidden values: one byte (FP8) dispatched, two (BF16) combined).

    python tests/bench_cdna4_mega_moe.py [--ranks N] [--mma-type fp8xfp4] [--relative-only]
"""
import argparse
import os
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank: int, world_size: int, store_path: str, args) -> None:
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", init_method=f"file://{store_path}", rank=rank, world_size=world_size)
    import deep_gemm
    from deep_gemm.utils.math import per_token_cast_to_fp4, per_token_cast_to_fp8

    num_tokens, hidden, inter, num_experts, num_topk = args.num_tokens, args.hidden, args.inter, args.experts, args.topk
    experts_per_rank, with_sf = num_experts // world_size, args.mma_type != "bf16xbf16"
    buffer = deep_gemm.get_symm_buffer_for_mega_moe(dist.group.WORLD, num_experts, num_tokens, num_topk, hidden, inter,
                                                    num_shared_experts=1, mma_type=args.mma_type)
    torch.manual_seed(rank)

    def cast(w):
        if not with_sf:
            return w.bfloat16()
        cast_fn = per_token_cast_to_fp4 if args.mma_type == "fp8xfp4" else per_token_cast_to_fp8
        q, sf = cast_fn(w, use_ue8m0=True, gran_k=32)
        return q, sf

    def expert_weights(n, k, groups):
        parts = [cast(torch.randn((n, k), device="cuda") * 0.05) for _ in range(groups)]
        if not with_sf:
            return torch.stack(parts)
        q, sf = torch.stack([p[0] for p in parts]), torch.stack([p[1] for p in parts])
        return q, deep_gemm.transform_sf_into_required_layout(sf, n, k, (1, 32), groups)

    def shared_weights(n, k):
        w = torch.randn((n, k), device="cuda") * 0.05
        if not with_sf:
            return w.bfloat16()
        q, sf = per_token_cast_to_fp8(w, use_ue8m0=True, gran_k=32)
        return q, deep_gemm.transform_sf_into_required_layout(sf, n, k, (1, 32))

    l1, l2 = deep_gemm.transform_weights_for_mega_moe(expert_weights(2 * inter, hidden, experts_per_rank),
                                                      expert_weights(hidden, inter, experts_per_rank))
    s1, s2 = deep_gemm.transform_weights_for_mega_moe(shared_weights(2 * inter, hidden), shared_weights(hidden, inter))
    x = torch.randn((num_tokens, hidden), device="cuda")
    topk_weights, topk_idx = torch.topk(torch.randn((num_tokens, num_experts), device="cuda"), num_topk, dim=-1)
    if with_sf:
        x_fp8, x_sf = per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
        buffer.x[:num_tokens].copy_(x_fp8)
        buffer.x_sf[:num_tokens].copy_(x_sf)
    else:
        buffer.x[:num_tokens].copy_(x.bfloat16())
    buffer.topk_idx[:num_tokens].copy_(topk_idx)
    buffer.topk_weights[:num_tokens].copy_(topk_weights.softmax(-1))
    y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    fn = deep_gemm.fp8_fp4_mega_moe if with_sf else deep_gemm.bf16_mega_moe
    run = lambda: fn(y, l1, l2, buffer, shared_l1_weights=s1, shared_l2_weights=s2, activation_clamp=10.0)

    for _ in range(3):
        run()
    torch.cuda.synchronize()
    dist.barrier()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.iterations):
        run()
    end.record()
    end.synchronize()
    seconds = start.elapsed_time(end) / 1e3 / args.iterations

    gathered = [None] * world_size
    dist.all_gather_object(gathered, topk_idx.cpu(), group=dist.group.WORLD)
    all_idx = torch.cat(gathered)
    received = int(((all_idx >= rank * experts_per_rank) & (all_idx < (rank + 1) * experts_per_rank)).sum())
    flops = 2 * 3 * hidden * inter * (received + num_tokens)
    fabric_bytes = received * hidden * 3
    line = (f"rank {rank}: {flops / seconds / 1e12:5.0f} TFLOPS, fabric {fabric_bytes / seconds / 1e9:4.0f} GB/s"
            if args.relative_only else
            f"rank {rank}: {seconds * 1e6:8.0f} us, {flops / seconds / 1e12:5.0f} TFLOPS, "
            f"fabric {fabric_bytes / seconds / 1e9:4.0f} GB/s, {received} received rows")
    lines = [None] * world_size
    dist.all_gather_object(lines, line)

    if rank == 0:
        print(f"{args.mma_type}: {world_size} ranks, {num_tokens} tokens per rank, hidden {hidden}, "
              f"intermediate {inter}, top-{num_topk} of {num_experts} experts + 1 shared", flush=True)
        for entry in lines:
            print("  " + entry, flush=True)
    dist.barrier()
    buffer.destroy()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranks", type=int, default=torch.cuda.device_count())
    parser.add_argument("--mma-type", default="fp8xfp4", choices=("fp8xfp4", "fp8xfp8", "bf16xbf16"))
    parser.add_argument("--num-tokens", type=int, default=8192)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--inter", type=int, default=3072)
    parser.add_argument("--experts", type=int, default=384)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--relative-only", action="store_true")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(worker, args=(args.ranks, os.path.join(directory, "store"), args), nprocs=args.ranks, join=True)


if __name__ == "__main__":
    main()
