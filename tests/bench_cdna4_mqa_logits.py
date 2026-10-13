"""Benchmark the CDNA4 MQA logits on the prefill (contiguous, causal) and decoding (paged) shapes of
tests/test_attention.py, in the FP8, MXFP8 and MXFP4 formats. Prefill rows also time aiter's Triton
FP8 MQA logits kernel when aiter is installed: `vs_aiter = aiter_us / ours_us`.

Calls are captured in a CUDA graph and replayed; decoding rows report the KV bytes read per second.
`--relative-only` prints no absolute numbers.
"""
import argparse
import os
import sys

os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".cache", "triton_dg_bench"))
import torch

sys.path.insert(0, os.path.dirname(__file__))
import deep_gemm
from deep_gemm.utils import per_custom_dims_cast_to_fp8
import test_attention as ref

try:
    from aiter.ops.triton.attention.fp8_mqa_logits import fp8_mqa_logits as aiter_mqa_logits
except Exception:  # aiter is optional
    aiter_mqa_logits = None

FORMATS = ("fp8", "mxfp8", "mxfp4")


def graph_us(fn, reps=10):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    graph.replay()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1e3 / reps


def quantize_kv(kv, fmt):
    return per_custom_dims_cast_to_fp8(kv, (0, ), False) if fmt == "fp8" else ref.quantize_mqa_q(kv, fmt)[0]


def prefill(relative_only):
    print("prefill: " + ("speedup over FP8" if relative_only else "us (TFLOPS)") + ", vs aiter FP8")
    for seq_len, seq_len_kv, num_heads in ((2048, 8192, 64), (8192, 8192, 64), (2048, 65536, 64),
                                           (8192, 65536, 64), (8192, 65536, 32)):
        head_dim = 128
        q = torch.randn(seq_len, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
        kv = torch.randn(seq_len_kv, head_dim, device="cuda", dtype=torch.bfloat16)
        ks = torch.zeros(seq_len, dtype=torch.int, device="cuda")
        ke = torch.arange(seq_len, dtype=torch.int, device="cuda") + (seq_len_kv - seq_len)
        flops = 2 * num_heads * head_dim * (ke - ks).sum().item()
        times, aiter_us = {}, float("nan")
        for fmt in FORMATS:
            dtype = torch.float if fmt == "fp8" else torch.bfloat16
            weights = ref.to_mqa_weights(torch.randn(seq_len, num_heads, device="cuda", dtype=dtype))
            q_in, kv_in = ref.quantize_mqa_q(q, fmt)[0], quantize_kv(kv, fmt)
            max_seqlen_k = (ke - ks).max().item()
            times[fmt] = graph_us(lambda: deep_gemm.fp8_fp4_mqa_logits(q_in, kv_in, weights, ks, ke, max_seqlen_k))
            if fmt == "fp8" and aiter_mqa_logits is not None:
                dense = weights.contiguous()
                aiter_us = graph_us(lambda: aiter_mqa_logits(q_in[0], kv_in[0], kv_in[1], dense, ks, ke))
        cells = [f"{f} {times['fp8'] / times[f]:.2f}x" if relative_only else
                 f"{f} {times[f]:8.1f} ({flops / times[f] / 1e6:4.0f})" for f in FORMATS]
        print(f"SQ={seq_len:5} SK={seq_len_kv:6} H={num_heads}: {', '.join(cells)}, vs aiter "
              f"{aiter_us / times['fp8']:.2f}x", flush=True)


def paged(relative_only):
    print("paged decoding (next_n = 1, 64 rows per page): " + ("speedup over FP8" if relative_only else "us (TB/s of KV)"))
    for batch, avg_kv, num_heads in ((64, 8192, 64), (64, 65536, 64), (256, 8192, 64), (256, 65536, 64),
                                     (128, 32768, 32)):
        head_dim, block_kv = 128, 64
        lens = torch.full((batch, 1), avg_kv, device="cuda", dtype=torch.int)
        blocks = (avg_kv + block_kv - 1) // block_kv
        table = torch.randperm(batch * blocks, device="cuda", dtype=torch.int).view(batch, blocks)
        kv = torch.randn((batch * blocks, block_kv, 1, head_dim), device="cuda", dtype=torch.bfloat16)
        q = torch.randn((batch, 1, num_heads, head_dim), device="cuda", dtype=torch.bfloat16)
        times, rates = {}, {}
        for fmt in FORMATS:
            dtype = torch.float if fmt == "fp8" else torch.bfloat16
            weights = ref.to_mqa_weights(torch.randn(batch, num_heads, device="cuda", dtype=dtype))
            q_in = ref.quantize_mqa_q(q, fmt)[0]
            cast = {"fp8": ref.kv_cache_cast_to_fp8, "mxfp8": ref.kv_cache_cast_to_mxfp8,
                    "mxfp4": ref.kv_cache_cast_to_mxfp4}[fmt]
            cache = cast(kv)[0]
            meta = deep_gemm.get_paged_mqa_logits_metadata(lens, block_kv, deep_gemm.get_num_sms())
            times[fmt] = graph_us(lambda: deep_gemm.fp8_fp4_paged_mqa_logits(q_in, cache, weights, lens, table, meta,
                                                                              blocks * block_kv))
            rates[fmt] = batch * avg_kv * (head_dim // (2 if fmt == "mxfp4" else 1) + 4) / times[fmt] / 1e6
            del cache
        cells = [f"{f} {times['fp8'] / times[f]:.2f}x" if relative_only else
                 f"{f} {times[f]:7.1f} ({rates[f]:.2f})" for f in FORMATS]
        print(f"B={batch:4} L={avg_kv:6} H={num_heads}: {', '.join(cells)}", flush=True)
        del kv
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["all", "prefill", "paged"], default="all")
    parser.add_argument("--relative-only", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(0)
    if args.kind in ("all", "prefill"):
        prefill(args.relative_only)
    if args.kind in ("all", "paged"):
        paged(args.relative_only)


if __name__ == "__main__":
    main()
