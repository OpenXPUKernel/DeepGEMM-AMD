"""Benchmark the CDNA4 FP8 x FP4 and FP4 x FP4 GEMMs against the FP8 GEMM and the PyTorch MXFP8 / MXFP4
GEMMs (torch._scaled_mm, hipBLASLt) on the forward shapes of tests/test_fp8_fp4.py.

Calls are captured in a CUDA graph, back to back, each on its own copy of B; the copies exceed the
last-level cache, so weights are read from memory as in inference. Reported ratios are speedups
(other / ours); `--relative-only` prints only those.
"""
import argparse

import torch

import deep_gemm

NK = [(2112, 7168), (576, 7168), (24576, 1536), (32768, 512), (7168, 16384), (4096, 7168), (7168, 2048)]


def graph_us(calls, reps=10):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for f in calls:
            f()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for f in calls:
            f()
    for _ in range(3):
        graph.replay()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1e3 / (reps * len(calls))


def quantize(x, fp4):
    if fp4:
        data, sf = deep_gemm.per_token_cast_to_fp4(x, use_ue8m0=True, gran_k=32)
    else:
        data, sf = deep_gemm.per_token_cast_to_fp8(x, True, 32)
    return data, deep_gemm.transform_sf_into_required_layout(sf, x.size(0), x.size(1), (1, 32))


def scaled_mm_operand(rows, k, fp4):
    if fp4:
        x = torch.randint(0, 255, (rows, k // 2), device="cuda", dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
    else:
        x = torch.randn(rows, k, device="cuda").to(torch.float8_e4m3fn)
    return x, torch.full((rows, k // 32), 127, device="cuda", dtype=torch.uint8).view(torch.float8_e8m0fnu)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", default="1,128,4096", help="comma-separated M values")
    parser.add_argument("--relative-only", action="store_true")
    args = parser.parse_args()
    ms = [int(x) for x in args.m.split(",")]
    cases = [(m, n, k) for m in ms for n, k in NK] + [(s, s, s) for s in (4096, 8192)]
    columns = ["fp8", "fp8xfp4", "fp4xfp4", "torch_mxfp8", "torch_mxfp4"]
    print(f"{'m':>5} {'n':>6} {'k':>6} " + ("" if args.relative_only else " ".join(f"{c:>11}" for c in columns)) +
          f" {'fp8/fp8xfp4':>11} {'fp8/fp4xfp4':>11} {'mxfp4/ours':>10} {'mxfp8/ours':>10}", flush=True)
    for m, n, k in cases:
        torch.manual_seed(0)
        copies = max(2, min(8, -(-768 * 2**20 // (n * k))))
        x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
        weights = [torch.randn((n, k), device="cuda", dtype=torch.bfloat16) for _ in range(copies)]
        d = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        t = {}
        for name, fp4_a, fp4_b in (("fp8", False, False), ("fp8xfp4", False, True), ("fp4xfp4", True, True)):
            a = quantize(x, fp4_a)
            bs = [quantize(w, fp4_b) for w in weights]
            t[name] = graph_us([lambda b=b: deep_gemm.fp8_fp4_gemm_nt(a, b, d, recipe_a=(1, 32), recipe_b=(1, 32))
                                for b in bs])
            del bs
        del weights
        for name, fp4 in (("torch_mxfp8", False), ("torch_mxfp4", True)):
            try:
                a, sa = scaled_mm_operand(m, k, fp4)
                bs = [scaled_mm_operand(n, k, fp4) for _ in range(copies)]
                t[name] = graph_us([lambda b=b: torch._scaled_mm(a, b[0].T, sa, b[1], out_dtype=torch.bfloat16)
                                    for b in bs])
                del bs
            except RuntimeError:  # shapes the library does not support
                t[name] = float("nan")
        absolute = "" if args.relative_only else " ".join(f"{t[c]:11.2f}" for c in columns)
        print(f"{m:5} {n:6} {k:6} {absolute} {t['fp8'] / t['fp8xfp4']:11.2f} {t['fp8'] / t['fp4xfp4']:11.2f} "
              f"{t['torch_mxfp4'] / t['fp4xfp4']:10.2f} {t['torch_mxfp8'] / t['fp8']:10.2f}", flush=True)


if __name__ == "__main__":
    main()
