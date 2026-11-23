"""AMD Mega MoE: dense oracle, independent serial reference and dynamic graphs."""

import argparse
import os
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def bf16(x: torch.Tensor) -> torch.Tensor:
    return x.bfloat16().float()


def quantize_rows(w: torch.Tensor, kind: str):
    """(stored weights, packed MN-major scales, dequantized FP32 weights) of [..., n, k] weights"""
    import deep_gemm
    from deep_gemm.utils.math import (
        cast_back_from_fp4,
        cast_back_from_fp8,
        per_token_cast_to_fp4,
        per_token_cast_to_fp8,
    )

    squeeze = w.dim() == 2
    w = w.unsqueeze(0) if squeeze else w
    g, n, k = w.shape
    stored, scales, dequantized = ([], [], [])
    for i in range(g):
        if kind == "fp4":
            q, sf = per_token_cast_to_fp4(w[i], use_ue8m0=True, gran_k=32)
            dequantized.append(cast_back_from_fp4(q, sf, gran_k=32))
        else:
            q, sf = per_token_cast_to_fp8(w[i], use_ue8m0=True, gran_k=32)
            dequantized.append(cast_back_from_fp8(q, sf, gran_k=32))
        (stored.append(q), scales.append(sf))
    stored, scales, dequantized = (
        torch.stack(stored),
        torch.stack(scales),
        torch.stack(dequantized),
    )
    packed = deep_gemm.transform_sf_into_required_layout(scales, n, k, (1, 32), g)
    if squeeze:
        return (stored[0], packed[0], dequantized[0])
    return (stored, packed, dequantized)


def swiglu_reference(
    l1: torch.Tensor, weight: torch.Tensor, clamp: float, with_sf: bool
) -> torch.Tensor:
    from deep_gemm.utils.math import cast_back_from_fp8, per_token_cast_to_fp8

    inter = l1.size(-1) // 2
    gate, up = (bf16(l1[:, :inter]), bf16(l1[:, inter:]))
    clamp_bf16 = float(torch.tensor(clamp).bfloat16())
    gate, up = (gate.clamp(max=clamp_bf16), up.clamp(-clamp_bf16, clamp_bf16))
    act = gate / (1 + torch.exp(-gate)) * up * weight[:, None]
    if not with_sf:
        return bf16(act)
    q, sf = per_token_cast_to_fp8(act, use_ue8m0=True, gran_k=32)
    return cast_back_from_fp8(q, sf, gran_k=32)


def run(rank, world_size, mma_type):
    import deep_gemm
    from deep_gemm.testing import calc_diff
    from deep_gemm.testing.moe import reference_mega_moe
    from deep_gemm.utils.math import cast_back_from_fp8, per_token_cast_to_fp8

    hidden = int(os.getenv("DG_MEGA_MOE_TEST_HIDDEN", "1280"))
    inter = int(os.getenv("DG_MEGA_MOE_TEST_INTER", "768"))
    experts, topk, capacity, clamp = 6 * world_size, 6, 512, 10.0
    with_sf = mma_type != "bf16xbf16"
    kind = "fp4" if mma_type == "fp8xfp4" else "fp8"
    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        dist.group.WORLD,
        experts,
        capacity,
        topk,
        hidden,
        inter,
        num_shared_experts=1,
        mma_type=mma_type,
    )
    gen = torch.Generator(device="cuda").manual_seed(1234)

    def model(shape, dtype):
        value = torch.randn(shape, device="cuda", generator=gen) * 0.05
        if with_sf:
            return quantize_rows(value, dtype)
        return value.bfloat16(), None, bf16(value)

    w1, sf1, ref1 = model((experts, 2 * inter, hidden), kind)
    w2, sf2, ref2 = model((experts, hidden, inter), kind)
    s1, ssf1, refs1 = model((2 * inter, hidden), "fp8")
    s2, ssf2, refs2 = model((hidden, inter), "fp8")
    mine = slice(rank * 6, (rank + 1) * 6)
    raw1 = (w1[mine], sf1[mine]) if with_sf else w1[mine]
    raw2 = (w2[mine], sf2[mine]) if with_sf else w2[mine]
    raw1, raw2 = deep_gemm.transform_weights_for_mega_moe(raw1, raw2)
    shared1, shared2 = deep_gemm.transform_weights_for_mega_moe(
        (s1, ssf1) if with_sf else s1, (s2, ssf2) if with_sf else s2
    )
    prepared = (
        deep_gemm.prepare_weights_for_resident_mega_moe(raw1, raw2)
        if mma_type == "fp8xfp4"
        else None
    )
    weights = prepared if prepared is not None else (raw1, raw2)
    fn = deep_gemm.fp8_fp4_mega_moe if with_sf else deep_gemm.bf16_mega_moe
    for case in range(6):
        torch.manual_seed(100 * rank + case)
        tokens = capacity - (rank * 37 + case * 11) % 200
        if case == 2 and rank == world_size - 1:
            tokens = 0
        x = torch.randn((tokens, hidden), device="cuda")
        scores = torch.randn((tokens, experts), device="cuda")
        values, ids = scores.topk(topk, dim=-1)
        route_weights = values.softmax(-1)
        mask = torch.rand_like(route_weights) < 0.1
        ids = ids.masked_fill(mask, -1)
        route_weights.masked_fill_(mask, 0)
        if case == 3:
            ids = torch.arange(topk, device="cuda").expand(tokens, -1).contiguous()
        elif case == 4:
            ids.fill_(-1)
            route_weights.zero_()
        elif case == 5:
            codes = torch.arange(tokens, device="cuda") % (1 << topk)
            active = (codes[:, None] & (1 << torch.arange(topk, device="cuda"))) != 0
            ids = scores.topk(topk, dim=-1).indices.masked_fill(~active, -1)
            route_weights = values.softmax(-1).masked_fill(~active, 0)
            begin = buffer.layout["combine"]
            size = topk * buffer.num_max_tokens_per_rank * hidden * 2
            buffer.buffer[begin : begin + size].view(torch.uint8).fill_(255)
            torch.cuda.synchronize()
            dist.barrier()
        if with_sf:
            q, qs = per_token_cast_to_fp8(
                x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
            )
            buffer.x[:tokens].copy_(q)
            buffer.x_sf[:tokens].copy_(qs)
            xref = cast_back_from_fp8(q, qs, gran_k=32, use_packed_ue8m0=True)
        else:
            buffer.x[:tokens].copy_(x.bfloat16())
            xref = bf16(x)
        buffer.topk_idx[:tokens].copy_(ids)
        buffer.topk_weights[:tokens].copy_(route_weights)
        stats = torch.zeros(6, device="cuda", dtype=torch.int32)
        kwargs = dict(
            shared_l1_weights=shared1,
            shared_l2_weights=shared2,
            cumulative_local_expert_recv_stats=stats,
            activation_clamp=clamp,
        )
        y = torch.empty((tokens, hidden), device="cuda", dtype=torch.bfloat16)
        y2 = torch.empty_like(y)
        routed_ids = [None] * world_size
        dist.all_gather_object(routed_ids, ids.cpu())
        all_ids = torch.cat(routed_ids)
        counts = torch.stack([(all_ids == rank * 6 + e).sum() for e in range(6)]).to(
            torch.int32
        )
        fn(y, *weights, buffer, **kwargs)
        torch.cuda.synchronize()
        assert torch.equal(stats.cpu(), counts), (
            f"rank {rank}: received expert statistics differ"
        )
        reference_mega_moe(y2, raw1, raw2, buffer, **kwargs)
        torch.cuda.synchronize()
        assert torch.equal(y, y2), f"rank {rank}: independent serial reference differs"
        dense = torch.zeros((tokens, hidden), device="cuda")
        for e in range(experts):
            rows, slots = (ids == e).nonzero(as_tuple=True)
            if rows.numel():
                activation = swiglu_reference(
                    bf16(xref[rows] @ ref1[e].T),
                    route_weights[rows, slots],
                    clamp,
                    with_sf,
                )
                dense.index_add_(0, rows, bf16(activation @ ref2[e].T))
        activation = swiglu_reference(
            bf16(xref @ refs1.T), torch.ones(tokens, device="cuda"), clamp, with_sf
        )
        dense += bf16(activation @ refs2.T)
        diff = float(calc_diff(y.float(), dense)) if tokens else 0.0
        assert diff < 2e-5, (rank, case, diff)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn(y2, *weights, buffer, **kwargs)
        before_stats = stats.clone()
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(y, y2), f"rank {rank}: graph replay differs"
        assert torch.equal((stats - before_stats).cpu(), 3 * counts)
        tensors = [
            buffer.x[:tokens],
            buffer.topk_idx[:tokens],
            buffer.topk_weights[:tokens],
        ]
        if with_sf:
            tensors += [buffer.x_sf[:tokens], raw1[0], shared1[0]]
            if prepared is not None:
                tensors.append(prepared[0].weight)
        originals = [t.clone() for t in tensors]

        def change(frame):
            for t, original in zip(tensors, originals):
                t.copy_(original)
            bit = frame % 2
            buffer.x[:tokens].view(torch.uint8).bitwise_xor_(128 if bit else 0)
            buffer.topk_idx[:tokens].copy_(
                torch.where(ids >= 0, (ids + frame * 7) % experts, ids)
            )
            buffer.topk_weights[:tokens].copy_(route_weights * (0.5 if bit else 1))
            if with_sf:
                buffer.x_sf[:tokens].add_(0x01010101 if bit else 0)
                raw1[0].view(torch.uint8).bitwise_xor_(
                    0x88 if kind == "fp4" and bit else 128 if bit else 0
                )
                shared1[0].view(torch.uint8).bitwise_xor_(128 if bit else 0)
                if prepared is not None:
                    prepared[0].weight.view(torch.uint8).bitwise_xor_(
                        0x88 if bit else 0
                    )

        try:
            snapshots = []
            for frame in range(8):
                change(frame)
                graph.replay()
                snapshots.append(y2.clone())
            torch.cuda.synchronize()
            for frame, actual in enumerate(snapshots):
                change(frame)
                reference_mega_moe(y2, raw1, raw2, buffer, **kwargs)
                torch.cuda.synchronize()
                assert torch.equal(y2, actual), (
                    f"rank {rank}: asynchronous changed graph case {frame} differs"
                )
        finally:
            for t, original in zip(tensors, originals):
                t.copy_(original)
            torch.cuda.synchronize()
        if rank == 0:
            print(
                f"  {mma_type} iteration {case}: tokens {tokens}, diff {diff:.2e}; serial + graph + 8 async updates OK",
                flush=True,
            )
    buffer.destroy()


def worker(rank, world, store, mma_types):
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "gloo", init_method=f"file://{store}", rank=rank, world_size=world
    )
    for mma_type in mma_types:
        run(rank, world, mma_type)
    dist.barrier()
    if rank == 0:
        print(f"Mega MoE: {world} ranks OK", flush=True)
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranks", type=int, default=torch.cuda.device_count())
    parser.add_argument("--mma-types", default="fp8xfp4")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as folder:
        mp.spawn(
            worker,
            args=(args.ranks, os.path.join(folder, "store"), args.mma_types.split(",")),
            nprocs=args.ranks,
            join=True,
        )


if __name__ == "__main__":
    main()
