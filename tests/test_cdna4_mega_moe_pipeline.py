"""Opt-in persistent Mega MoE pipeline regression on multiple MI355X GPUs.

Checks FP32 reference, serial-path equality, skewed routing, and graph replays
with changed activations, scales, routing and expert/shared weights.
Run: python tests/test_cdna4_mega_moe_pipeline.py --ranks 8
"""
import argparse
import os
import sys
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F


def bf16(x: torch.Tensor) -> torch.Tensor:
    return x.bfloat16().float()


def quantize_rows(w: torch.Tensor, kind: str):
    """(stored weights, packed MN-major scales, dequantized FP32 weights) of [..., n, k] weights"""
    import deep_gemm
    from deep_gemm.utils.math import cast_back_from_fp4, cast_back_from_fp8, per_token_cast_to_fp4, \
        per_token_cast_to_fp8
    squeeze = w.dim() == 2
    w = w.unsqueeze(0) if squeeze else w
    g, n, k = w.shape
    stored, scales, dequantized = [], [], []
    for i in range(g):
        if kind == "fp4":
            q, sf = per_token_cast_to_fp4(w[i], use_ue8m0=True, gran_k=32)
            dequantized.append(cast_back_from_fp4(q, sf, gran_k=32))
        else:
            q, sf = per_token_cast_to_fp8(w[i], use_ue8m0=True, gran_k=32)
            dequantized.append(cast_back_from_fp8(q, sf, gran_k=32))
        stored.append(q), scales.append(sf)
    stored, scales, dequantized = torch.stack(stored), torch.stack(scales), torch.stack(dequantized)
    packed = deep_gemm.transform_sf_into_required_layout(scales, n, k, (1, 32), g)
    if squeeze:
        return stored[0], packed[0], dequantized[0]
    return stored, packed, dequantized


def swiglu_reference(l1: torch.Tensor, weight: torch.Tensor, clamp: float, with_sf: bool) -> torch.Tensor:
    from deep_gemm.utils.math import cast_back_from_fp8, per_token_cast_to_fp8
    inter = l1.size(-1) // 2
    gate, up = bf16(l1[:, :inter]), bf16(l1[:, inter:])
    clamp_bf16 = float(torch.tensor(clamp).bfloat16())
    gate, up = gate.clamp(max=clamp_bf16), up.clamp(-clamp_bf16, clamp_bf16)
    act = gate / (1 + torch.exp(-gate)) * up * weight[:, None]
    if not with_sf:
        return bf16(act)
    q, sf = per_token_cast_to_fp8(act, use_ue8m0=True, gran_k=32)
    return cast_back_from_fp8(q, sf, gran_k=32)


def run(rank: int, world_size: int, mma_type: str) -> None:
    import deep_gemm
    from deep_gemm.utils.math import per_token_cast_to_fp8
    from deep_gemm.testing import calc_diff

    num_experts, num_topk, hidden, inter, num_shared = 6 * world_size, 6, 1024, 512, 1
    hidden = int(os.getenv("DG_MEGA_MOE_TEST_HIDDEN", hidden))
    inter = int(os.getenv("DG_MEGA_MOE_TEST_INTER", inter))
    num_max_tokens, clamp, with_sf = 512, 10.0, mma_type != "bf16xbf16"
    experts_per_rank = num_experts // world_size
    group = dist.group.WORLD
    buffer = deep_gemm.get_symm_buffer_for_mega_moe(group, num_experts, num_max_tokens, num_topk, hidden, inter,
                                                    num_shared_experts=num_shared, mma_type=mma_type)

    # the model, the same on every rank
    gen = torch.Generator(device="cuda").manual_seed(1234)
    l1_all = torch.randn((num_experts, 2 * inter, hidden), device="cuda", generator=gen) * 0.05
    l2_all = torch.randn((num_experts, hidden, inter), device="cuda", generator=gen) * 0.05
    shared_l1 = torch.randn((2 * inter * num_shared, hidden), device="cuda", generator=gen) * 0.05
    shared_l2 = torch.randn((hidden, inter * num_shared), device="cuda", generator=gen) * 0.05
    weight_kind = {"fp8xfp4": "fp4", "fp8xfp8": "fp8", "bf16xbf16": None}[mma_type]
    if weight_kind is None:
        l1_ref, l2_ref = bf16(l1_all), bf16(l2_all)
        s1_ref, s2_ref = bf16(shared_l1), bf16(shared_l2)
        mine = slice(rank * experts_per_rank, (rank + 1) * experts_per_rank)
        l1_w, l2_w = l1_all[mine].bfloat16(), l2_all[mine].bfloat16()
        s1_w, s2_w = shared_l1.bfloat16(), shared_l2.bfloat16()
    else:
        q1, sf1, l1_ref = quantize_rows(l1_all, weight_kind)
        q2, sf2, l2_ref = quantize_rows(l2_all, weight_kind)
        qs1, sfs1, s1_ref = quantize_rows(shared_l1, "fp8")
        qs2, sfs2, s2_ref = quantize_rows(shared_l2, "fp8")
        mine = slice(rank * experts_per_rank, (rank + 1) * experts_per_rank)
        l1_w, l2_w = (q1[mine], sf1[mine]), (q2[mine], sf2[mine])
        s1_w, s2_w = (qs1, sfs1), (qs2, sfs2)
    l1_w, l2_w = deep_gemm.transform_weights_for_mega_moe(l1_w, l2_w)
    s1_w, s2_w = deep_gemm.transform_weights_for_mega_moe(s1_w, s2_w)

    for iteration in range(5 if os.getenv("DG_MEGA_MOE_TEST_SKEW") else 3):
        # this rank's tokens
        torch.manual_seed(100 * rank + iteration)
        # the last iteration leaves the last rank without tokens (it still routes and serves experts)
        num_tokens = num_max_tokens - (rank * 37 + iteration * 11) % 200
        if iteration == 2 and rank == world_size - 1:
            num_tokens = 0
        x = torch.randn((num_tokens, hidden), device="cuda")
        scores = torch.randn((num_tokens, num_experts), device="cuda")
        topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1, sorted=False)
        topk_weights = topk_weights.softmax(-1)
        masked = torch.rand_like(topk_weights) < 0.1
        topk_idx = topk_idx.masked_fill(masked, -1)
        topk_weights = topk_weights.masked_fill(masked, 0)
        if iteration == 3:
            # Every routed token goes to rank zero; all other ranks have empty local experts.
            topk_idx = torch.arange(num_topk, device="cuda").expand(num_tokens, -1).contiguous()
        elif iteration == 4:
            topk_idx.fill_(-1)
            topk_weights.zero_()
        if with_sf:
            x_fp8, x_sf = per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
            from deep_gemm.utils.math import cast_back_from_fp8
            x_ref = cast_back_from_fp8(x_fp8, x_sf, gran_k=32, use_packed_ue8m0=True)
            buffer.x[:num_tokens].copy_(x_fp8)
            buffer.x_sf[:num_tokens].copy_(x_sf)
        else:
            x_ref = bf16(x)
            buffer.x[:num_tokens].copy_(x.bfloat16())
        buffer.topk_idx[:num_tokens].copy_(topk_idx)
        buffer.topk_weights[:num_tokens].copy_(topk_weights)

        stats = torch.randint(0, 100, (experts_per_rank, ), dtype=torch.int32, device="cuda")
        stats_before = stats.clone()
        y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
        kwargs = dict(shared_l1_weights=s1_w, shared_l2_weights=s2_w, cumulative_local_expert_recv_stats=stats,
                      activation_clamp=clamp)
        (deep_gemm.fp8_fp4_mega_moe if with_sf else deep_gemm.bf16_mega_moe)(y, l1_w, l2_w, buffer, **kwargs)
        torch.cuda.synchronize()

        # reference: every slot's expert, then the shared expert
        y_ref = torch.zeros((num_tokens, hidden), device="cuda")
        for e in range(num_experts):
            tokens, slots = (topk_idx == e).nonzero(as_tuple=True)
            if tokens.numel() == 0:
                continue
            l1 = bf16(x_ref[tokens] @ l1_ref[e].T)
            act = swiglu_reference(l1, topk_weights[tokens, slots], clamp, with_sf)
            y_ref.index_add_(0, tokens, bf16(act @ l2_ref[e].T))
        shared_act = swiglu_reference(bf16(x_ref @ s1_ref.T), torch.ones(num_tokens, device="cuda"), clamp, with_sf)
        y_ref += bf16(shared_act @ s2_ref.T)
        diff = float(calc_diff(y.float(), y_ref)) if num_tokens > 0 else 0.0
        assert diff < 2e-5, f"rank {rank} {mma_type} iteration {iteration}: diff {diff}"

        # received rows per local expert from every rank's top-k
        gathered = [None] * world_size
        dist.all_gather_object(gathered, topk_idx.cpu(), group=group)
        all_idx = torch.cat(gathered)
        expected = torch.stack([(all_idx == rank * experts_per_rank + e).sum() for e in range(experts_per_rank)])
        assert torch.equal((stats - stats_before).cpu(), expected.to(torch.int32)), f"rank {rank}: stats"

        # repeatable
        y2 = torch.empty_like(y)
        (deep_gemm.fp8_fp4_mega_moe if with_sf else deep_gemm.bf16_mega_moe)(y2, l1_w, l2_w, buffer, **kwargs)
        torch.cuda.synchronize()
        assert torch.equal(y, y2), f"rank {rank} {mma_type}: not repeatable"
        if os.getenv("DG_MEGA_MOE_COMPARE_SERIAL"):
            keys = ("DG_MEGA_MOE_CHUNKS", "DG_MEGA_MOE_CUMASK", "DG_MEGA_MOE_FUSE_SWIGLU",
                    "DG_MEGA_MOE_ORDERED_PUSH", "DG_MEGA_MOE_DIRECT_COMBINE", "DG_MEGA_MOE_TILE_GROUP_M",
                    "DG_MEGA_MOE_PIPELINE_COMBINE", "DG_MEGA_MOE_PIPELINE_CUS",
                    "DG_MEGA_MOE_SPECIAL_EPILOGUE", "DG_MEGA_MOE_NV_PIPELINE")
            saved = {key: os.environ.pop(key, None) for key in keys}
            try:
                os.environ["DG_MEGA_MOE_CHUNKS"] = "1"
                os.environ["DG_MEGA_MOE_DIRECT_COMBINE"] = "0"
                os.environ["DG_MEGA_MOE_TILE_GROUP_M"] = "8"
                (deep_gemm.fp8_fp4_mega_moe if with_sf else deep_gemm.bf16_mega_moe)(
                    y2, l1_w, l2_w, buffer, **kwargs)
                torch.cuda.synchronize()
                assert torch.equal(y, y2), f"rank {rank} {mma_type}: serial path differs"
            finally:
                for key, value in saved.items():
                    if value is not None:
                        os.environ[key] = value
                    else:
                        os.environ.pop(key, None)
        # A captured multi-stream call must preserve output and accumulated receive statistics.
        if os.getenv("DG_MEGA_MOE_TEST_GRAPH") or os.getenv("DG_MEGA_MOE_TEST_GRAPH_UPDATES"):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                (deep_gemm.fp8_fp4_mega_moe if with_sf else deep_gemm.bf16_mega_moe)(
                    y2, l1_w, l2_w, buffer, **kwargs)
            before_replay = stats.clone()
            for _ in range(3):
                graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(y, y2), f"rank {rank} {mma_type}: graph differs"
            assert torch.equal((stats - before_replay).cpu(), expected.to(torch.int32) * 3)
            if os.getenv("DG_MEGA_MOE_TEST_GRAPH_UPDATES"):
                # Reuse graph-private output allocations with different data.
                # An unchanged-input replay cannot expose stale payload caches.
                expected_y = torch.empty_like(y)
                original_x = buffer.x[:num_tokens].clone()
                original_idx = buffer.topk_idx[:num_tokens].clone()
                original_sf = buffer.x_sf[:num_tokens].clone() if with_sf else None
                saved_pipe = os.environ.get("DG_MEGA_MOE_PIPELINE_COMBINE")
                saved_queue = os.environ.get("DG_MEGA_MOE_NV_PIPELINE")
                saved_direct = os.environ.get("DG_MEGA_MOE_DIRECT_COMBINE")
                try:
                    for scale in (0.0, 0.5, 1.0):
                        if with_sf:
                            buffer.x_sf[:num_tokens].copy_(original_sf + (0x01010101 if scale == 0.5 else 0))
                            buffer.x[:num_tokens].view(torch.uint8).copy_(original_x.view(torch.uint8) ^ (0x80 if scale == 0.5 else 0))
                        else:
                            buffer.x[:num_tokens].copy_(original_x * (-1 if scale == 0.5 else 1))
                        shifted_idx = torch.where(original_idx >= 0, (original_idx + (1 if scale == 0.5 else 0)) % num_experts, original_idx)
                        buffer.topk_idx[:num_tokens].copy_(shifted_idx)
                        buffer.topk_weights[:num_tokens].copy_(topk_weights * scale)
                        graph.replay()
                        torch.cuda.synchronize()
                        os.environ["DG_MEGA_MOE_PIPELINE_COMBINE"] = "0"
                        os.environ["DG_MEGA_MOE_NV_PIPELINE"] = "0"
                        os.environ["DG_MEGA_MOE_DIRECT_COMBINE"] = "0"
                        (deep_gemm.fp8_fp4_mega_moe if with_sf else deep_gemm.bf16_mega_moe)(
                            expected_y, l1_w, l2_w, buffer, **kwargs)
                        torch.cuda.synchronize()
                        assert torch.equal(y2, expected_y), f"rank {rank}: stale graph output at weight scale {scale}"
                    if rank == 0: print("    graph replay + changed routes/weights/activations/scales: bitwise OK", flush=True)
                finally:
                    if with_sf: buffer.x_sf[:num_tokens].copy_(original_sf)
                    buffer.x[:num_tokens].copy_(original_x)
                    buffer.topk_idx[:num_tokens].copy_(original_idx)
                    buffer.topk_weights[:num_tokens].copy_(topk_weights)
                    for key, value in (("DG_MEGA_MOE_NV_PIPELINE", saved_queue), ("DG_MEGA_MOE_PIPELINE_COMBINE", saved_pipe),
                                       ("DG_MEGA_MOE_DIRECT_COMBINE", saved_direct)):
                        if value is None: os.environ.pop(key, None)
                        else: os.environ[key] = value
            if os.getenv("DG_MEGA_MOE_TEST_GRAPH_ASYNC"):
                original_x=buffer.x[:num_tokens].clone()
                original_sf=buffer.x_sf[:num_tokens].clone()
                original_idx=buffer.topk_idx[:num_tokens].clone()
                original_weight=buffer.topk_weights[:num_tokens].clone()
                original_l1=l1_w[0].view(torch.uint8).clone()
                original_s1=s1_w[0].view(torch.uint8).clone()
                def change(case):
                    bit=case%2
                    buffer.x[:num_tokens].view(torch.uint8).copy_(original_x.view(torch.uint8)^(0x80 if bit else 0))
                    buffer.x_sf[:num_tokens].copy_(original_sf+(0x01010101 if bit else 0))
                    buffer.topk_idx[:num_tokens].copy_(torch.where(original_idx<0,original_idx,(original_idx+case*7)%num_experts))
                    buffer.topk_weights[:num_tokens].copy_(original_weight*(.5 if bit else 1))
                    l1_w[0].view(torch.uint8).copy_(original_l1^(0x88 if bit else 0))
                    s1_w[0].view(torch.uint8).copy_(original_s1^(0x80 if bit else 0))
                saved_queue=os.environ.get("DG_MEGA_MOE_NV_PIPELINE")
                snapshots=[]
                try:
                    for case in range(8):
                        change(case);graph.replay();snapshots.append(y2.clone())
                    torch.cuda.synchronize()
                    os.environ["DG_MEGA_MOE_NV_PIPELINE"]="0"
                    for case,actual in enumerate(snapshots):
                        change(case)
                        deep_gemm.fp8_fp4_mega_moe(y2,l1_w,l2_w,buffer,**kwargs)
                        torch.cuda.synchronize()
                        assert torch.equal(y2,actual),f"rank {rank}: asynchronous changed graph case {case} differs"
                finally:
                    if saved_queue is None:os.environ.pop("DG_MEGA_MOE_NV_PIPELINE",None)
                    else:os.environ["DG_MEGA_MOE_NV_PIPELINE"]=saved_queue
                    buffer.x[:num_tokens].copy_(original_x);buffer.x_sf[:num_tokens].copy_(original_sf)
                    buffer.topk_idx[:num_tokens].copy_(original_idx);buffer.topk_weights[:num_tokens].copy_(original_weight)
                    l1_w[0].view(torch.uint8).copy_(original_l1);s1_w[0].view(torch.uint8).copy_(original_s1)
                    torch.cuda.synchronize()
                if rank==0:print("    8 asynchronous graph calls, changed routing/input/scales/expert and shared weights: bitwise OK",flush=True)

        if rank == 0:
            print(f"  {mma_type} iteration {iteration}: tokens {num_tokens}, diff {diff:.2e}", flush=True)
    buffer.destroy()


def worker(rank: int, world_size: int, store_path: str, mma_types) -> None:
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", init_method=f"file://{store_path}", rank=rank, world_size=world_size)
    for mma_type in mma_types:
        run(rank, world_size, mma_type)
    dist.barrier()
    if rank == 0:
        print(f"Mega MoE: {world_size} ranks OK", flush=True)
    dist.destroy_process_group()


def main():
    for key in ("DG_MEGA_MOE_NV_PIPELINE", "DG_MEGA_MOE_TEST_SKEW", "DG_MEGA_MOE_COMPARE_SERIAL",
                "DG_MEGA_MOE_TEST_GRAPH_UPDATES", "DG_MEGA_MOE_TEST_GRAPH_ASYNC"):
        os.environ.setdefault(key, "1")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranks", type=int, default=torch.cuda.device_count())
    parser.add_argument("--mma-types", default="fp8xfp4")
    args = parser.parse_args()
    if args.mma_types != "fp8xfp4":
        parser.error("This pipeline regression requires --mma-types fp8xfp4")
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(worker, args=(args.ranks, os.path.join(directory, "store"), args.mma_types.split(",")),
                 nprocs=args.ranks, join=True)


if __name__ == "__main__":
    main()
