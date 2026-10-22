"""Benchmark the CDNA4 Mega MoE on the default shapes of tests/test_mega_moe.py (one process per GPU):
the time per call, the TFLOPS of the routed and shared expert GEMMs, and the fabric bytes per
second (each received token's hidden values: one byte (FP8) dispatched, two (BF16) combined).

    python tests/bench_cdna4_mega_moe.py [--ranks N] [--mma-type fp8xfp4] [--relative-only]
"""
import argparse
import json
import os
import statistics
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from _amd_benchmark_metrics import peak_metrics


def measure_stage_mfu(run, y, total_flops, peak, world_size, rank, iterations):
    """External HIP event nodes timestamp both stages of the complete graph."""
    import deep_gemm
    expected = y.clone()
    os.environ["DG_MEGA_MOE_STAGE1_EVENTS"] = "1"
    try:
        run()  # Allocate diagnostic events before capture.
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
    finally:
        os.environ.pop("DG_MEGA_MOE_STAGE1_EVENTS", None)
    for _ in range(100):
        graph.replay()
    torch.cuda.synchronize()
    dist.barrier()
    samples = []
    for trial in range(10):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
        end.synchronize()
        phases = deep_gemm._C.cdna4_stage1_phase_times()
        assert len(phases) == 6 and all(v >= 0 for v in phases), phases
        assert torch.equal(y, expected), "phase-instrumented graph changed output"
        row = dict(rank=rank, phases_ms=phases,
                   stage1_ms=sum(phases[:2]), stage2_ms=sum(phases[2:]),
                   total_ms=sum(phases), graph_batch_ms=begin.elapsed_time(end) / iterations)
        rows = [None] * world_size
        dist.all_gather_object(rows, row)
        samples.append(dict(trial=trial, ranks=rows))
        dist.barrier()
    # Detect accidentally loading an older extension whose ordinary HIP events
    # preserve stale eager timestamps instead of measuring graph replay.
    assert len({tuple(s["ranks"][rank]["phases_ms"]) for s in samples}) > 1, "stale phase timestamps; rebuild extension"
    if rank != 0:
        return None
    result = dict(
        method="External HIP timing event nodes in full graph; final replay of each batch; median of ten rank-max samples",
        phase_names=["routing_metadata_init", "dispatch_gemm1_swiglu_quant",
                     "stage2_prepare", "gemm2_bf16_return", "return_barrier", "combine_reduce"],
        samples=samples, stages={},
    )
    for name, factor in [("stage1", 2 / 3), ("stage2", 1 / 3), ("total", 1)]:
        ms = statistics.median(max(row[name + "_ms"] for row in sample["ranks"]) for sample in samples)
        result["stages"][name] = dict(latency_ms=ms, useful_flops=total_flops * factor,
                                     mfu=total_flops * factor / (ms * 1e-3 * world_size * peak * 1e12))
    result["instrumented_batch_ms"] = statistics.median(
        max(row["graph_batch_ms"] for row in sample["ranks"]) for sample in samples)
    print("Two-stage MFU diagnostics: " + json.dumps(result["stages"]), flush=True)
    return result


def worker(rank: int, world_size: int, store_path: str, args) -> None:
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", init_method=f"file://{store_path}", rank=rank, world_size=world_size)
    import deep_gemm
    from deep_gemm.utils.math import per_token_cast_to_fp4, per_token_cast_to_fp8

    num_tokens, hidden, inter, num_experts, num_topk = args.num_tokens, args.hidden, args.inter, args.experts, args.topk
    experts_per_rank, with_sf = num_experts // world_size, args.mma_type != "bf16xbf16"
    buffer = deep_gemm.get_symm_buffer_for_mega_moe(dist.group.WORLD, num_experts, num_tokens, num_topk, hidden, inter,
                                                    num_shared_experts=args.shared_experts, mma_type=args.mma_type)
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
    s1, s2 = (deep_gemm.transform_weights_for_mega_moe(shared_weights(2 * inter, hidden), shared_weights(hidden, inter))
              if args.shared_experts else (None, None))
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
    eager_run = run

    for _ in range(3):
        run()
    torch.cuda.synchronize()
    if args.compare_serial:
        actual = y.clone()
        overrides = {"DG_MEGA_MOE_CHUNKS": "1", "DG_MEGA_MOE_NV_PIPELINE": "0", "DG_MEGA_MOE_STAGE1_QUEUE": "0",
                     "DG_MEGA_MOE_PIPELINE_COMBINE": "0",
                     "DG_MEGA_MOE_SPECIAL_EPILOGUE": "0",
                     "DG_MEGA_MOE_DIRECT_COMBINE": "0", "DG_MEGA_MOE_FUSE_SWIGLU": "0",
                     "DG_MEGA_MOE_TILE_GROUP_M": "8"}
        saved = {key: os.environ.get(key) for key in overrides}
        try:
            os.environ.update(overrides)
            run()
            torch.cuda.synchronize()
            assert torch.equal(actual, y), f"rank {rank}: full workload differs from serial path"
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        if rank == 0:
            print("Full workload: bitwise serial-path comparison passed", flush=True)
    dist.barrier()
    if args.graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        torch.cuda.synchronize()
        dist.barrier()
        run = graph.replay
        for _ in range(100 if args.phase_mfu else 3):
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
    flops = 2 * 3 * hidden * inter * (received + args.shared_experts * num_tokens)
    fabric_bytes = received * hidden * 3
    line = (f"rank {rank}: {flops / seconds / 1e12:5.0f} TFLOPS, fabric {fabric_bytes / seconds / 1e9:4.0f} GB/s"
            if args.relative_only else
            f"rank {rank}: {seconds * 1e6:8.0f} us, {flops / seconds / 1e12:5.0f} TFLOPS, "
            f"fabric {fabric_bytes / seconds / 1e9:4.0f} GB/s, {received} received rows")
    lines = [None] * world_size
    dist.all_gather_object(lines, line)

    if rank == 0:
        print(f"{args.mma_type}: {world_size} ranks, {num_tokens} tokens per rank, hidden {hidden}, "
              f"intermediate {inter}, top-{num_topk} of {num_experts} experts + {args.shared_experts} shared", flush=True)
        for entry in lines:
            print("  " + entry, flush=True)
    measurements = [None] * world_size
    dist.all_gather_object(measurements, {"rank": rank, "seconds": seconds, "flops": flops})
    if rank == 0:
        # One layer spans every rank: use the slowest rank, and useful (unpadded) expert FLOPs.
        latency = max(item["seconds"] for item in measurements)
        total_flops = sum(item["flops"] for item in measurements)
        result = {"mma_type": args.mma_type, "ranks": world_size, "tokens_per_rank": num_tokens,
                  "hidden": hidden, "intermediate": inter, "experts": num_experts, "topk": num_topk,
                  "shared_experts": args.shared_experts, "graph": args.graph, "iterations": args.iterations,
                  "serial_comparison": args.compare_serial,
                  "chunks": os.getenv("DG_MEGA_MOE_CHUNKS", "default"), "latency_ms": latency * 1e3,
                  "overrides": {key: value for key, value in os.environ.items() if key.startswith(("DG_MEGA_MOE_", "DG_MOE_PIPELINE_"))},
                  "useful_flops": total_flops,
                  "measurements": measurements}
        result.update(peak_metrics(total_flops, latency, world_size, args.peak_tflops, args.target_mfu))
        summary = "End-to-end AMD GPU benchmark"
        if not args.relative_only:
            summary += f", slowest rank {latency * 1e3:.3f} ms"
        if args.peak_tflops is not None:
            summary += f", caller-normalized MFU {result['mfu']:.2%}"
        print(summary, flush=True)
        if args.json_output:
            with open(args.json_output, "w") as stream:
                json.dump(result, stream, indent=2)
    if args.phase_mfu:
        diagnostic = measure_stage_mfu(
            eager_run, y, sum(item["flops"] for item in measurements),
            args.peak_tflops, world_size, rank, args.iterations)
        if rank == 0 and args.json_output:
            result["phase_diagnostics"] = diagnostic
            with open(args.json_output, "w") as stream:
                json.dump(result, stream, indent=2)
    if args.profile:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as prof:
            dist.barrier()
            run()
            torch.cuda.synchronize()
        if rank == 0:
            print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=20), flush=True)
            if args.trace_output:
                prof.export_chrome_trace(args.trace_output)
    if args.gemm_ceiling:
        # Diagnostic only: exclude dispatch, activation, combine and shared experts.
        # This is NOT the end-to-end MFU above.
        counts = torch.bincount(all_idx.flatten(), minlength=num_experts)[
            rank * experts_per_rank:(rank + 1) * experts_per_rank].to(device="cuda", dtype=torch.int32)
        aligned = ((counts + 255) // 256) * 256
        ends = (aligned.cumsum(0) - aligned + counts).to(torch.int32)
        stage_times = []
        from deep_gemm import _C
        for acts, sf, weights, n in [
            (buffer.l1_acts, buffer.l1_acts_sf, l1, 2 * inter),
            (buffer.l2_acts, buffer.l2_acts_sf, l2, hidden),
        ]:
            out = torch.empty((acts.size(0), n), dtype=torch.bfloat16, device="cuda")
            w, wsf = weights if with_sf else (weights, None)
            gemm = lambda: _C.cdna4_m_grouped_fp8_gemm_contiguous(
                acts, sf, w, wsf, out, ends, 32, 32, True, False, 256, 0)
            for _ in range(2):
                gemm()
            start.record()
            for _ in range(10):
                gemm()
            end.record()
            end.synchronize()
            stage_times.append(start.elapsed_time(end) / 10)
        ceiling = [None] * world_size
        dist.all_gather_object(ceiling, stage_times)
        if rank == 0:
            print("Diagnostic isolated routed GEMMs only (L1/L2 ms per rank; NOT full MoE): " + str(ceiling), flush=True)
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
    parser.add_argument("--shared-experts", type=int, choices=(0, 1), default=1)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--relative-only", action="store_true")
    parser.add_argument("--graph", action="store_true", help="time graph replay including all MoE stages")
    parser.add_argument("--compare-serial", action="store_true", help="check full-workload output before timing")
    parser.add_argument("--profile", action="store_true", help="profile a separate untimed call")
    parser.add_argument("--trace-output", help="rank-zero Chrome trace from --profile")
    parser.add_argument("--gemm-ceiling", action="store_true", help="separately measure isolated routed GEMMs")
    parser.add_argument("--phase-mfu", action="store_true", help="diagnose Stage1 dispatch/GEMM1 and Stage2 GEMM2/combine MFU with external HIP event nodes")
    parser.add_argument("--peak-tflops", type=float, help="optional caller-supplied denominator for private MFU measurements; no default")
    parser.add_argument("--target-mfu", type=float, default=0.50)
    parser.add_argument("--json-output", help="write per-rank timings; normalized metrics require --peak-tflops")
    args = parser.parse_args()
    if args.phase_mfu and (os.getenv("DG_MEGA_MOE_STAGE1_QUEUE") != "1" or args.shared_experts != 0
                           or args.mma_type != "fp8xfp4" or not args.graph):
        parser.error("--phase-mfu requires DG_MEGA_MOE_STAGE1_QUEUE=1, --shared-experts 0, --mma-type fp8xfp4 and --graph")
    if args.phase_mfu and os.getenv("DG_MEGA_MOE_STAGE1_EVENTS", "0") != "0":
        parser.error("unset DG_MEGA_MOE_STAGE1_EVENTS for graph phase profiling")
    if not 0 < args.target_mfu <= 1:
        parser.error("--target-mfu must be in (0, 1]")
    if args.ranks < 1 or args.iterations < 1 or args.num_tokens < 1 or args.experts % args.ranks:
        parser.error("positive ranks/tokens/iterations and experts divisible by ranks are required")
    if args.peak_tflops is not None:
        try:
            peak_metrics(0, 1, 1, args.peak_tflops, args.target_mfu)
        except ValueError as error:
            parser.error(str(error))
    if args.phase_mfu and args.peak_tflops is None:
        parser.error("--phase-mfu requires an explicit caller-supplied --peak-tflops")
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(worker, args=(args.ranks, os.path.join(directory, "store"), args), nprocs=args.ranks, join=True)


if __name__ == "__main__":
    main()
