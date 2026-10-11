"""Benchmark the CDNA4 grouped FP8 GEMMs on the shapes of tests/test_fp8_fp4.py.

Kernel time comes from the profiler (L2 flushed before every call), as in the CUDA tests.
Each row also times the dense FP8 GEMM on the same total problem (one weight matrix) as an
efficiency reference: `vs_dense = dense_us / grouped_us`.
"""
import argparse
import csv
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
import deep_gemm
from deep_gemm.testing import bench_kineto, calc_diff, count_bytes
from generators import (MajorTypeAB, generate_k_grouped_contiguous, generate_m_grouped_contiguous,
                        generate_m_grouped_masked)

KERNEL = "dg_cdna4_gemm"
NK = [(6144, 7168), (7168, 3072), (4096, 4096), (4096, 2048)]


def kernel_us(fn):
    return bench_kineto(fn, KERNEL, suppress_kineto_output=True) * 1e6


def dense_us(m, n, k, major_a=MajorTypeAB.KMajor, major_b=MajorTypeAB.KMajor, out_dtype=torch.bfloat16):
    a = deep_gemm.per_token_cast_to_fp8(torch.randn((m, k), device="cuda", dtype=torch.bfloat16), True, 128)
    b = deep_gemm.per_block_cast_to_fp8(torch.randn((n, k), device="cuda", dtype=torch.bfloat16), True, 128)
    if major_a == MajorTypeAB.MNMajor:
        a = (a[0].T.contiguous().T, a[1])
    if major_b == MajorTypeAB.MNMajor:
        b = (b[0].T.contiguous().T, b[1])
    d = torch.empty((m, n), device="cuda", dtype=out_dtype)
    return kernel_us(lambda: deep_gemm.fp8_gemm_nt(a, b, d))


def bench_contiguous(rows):
    for groups, expected_m in ((4, 8192), (8, 4096)):
        for n, k in NK:
            for major_b in (MajorTypeAB.KMajor, MajorTypeAB.MNMajor):
                for psum in (False, True):
                    m, a, b, layout, d, ref, valid = generate_m_grouped_contiguous(
                        groups, expected_m, n, k, MajorTypeAB.KMajor, major_b, use_ue8m0=True,
                        use_psum_layout=psum)
                    fn = lambda: deep_gemm.m_grouped_fp8_gemm_nt_contiguous(a, b, d, layout, use_psum_layout=psum)
                    fn()
                    diff = calc_diff(d[valid], ref[valid])
                    t = kernel_us(fn)
                    t_dense = dense_us(m, n, k, major_b=major_b)
                    rows.append(dict(kind="contiguous", groups=groups, m=m, n=n, k=k,
                                     layout="NT" if major_b == MajorTypeAB.KMajor else "NN", psum=int(psum),
                                     us=t, tflops=2 * m * n * k / t / 1e6, gbps=count_bytes(a, b, d) / t / 1e3,
                                     dense_us=t_dense, vs_dense=t_dense / t, diff=diff))
                    report(rows[-1])


def bench_masked(rows):
    max_m = 4096
    for groups, expected_m in ((32, 192), (6, 1024), (32, 20), (6, 20)):
        for n, k in NK:
            for psum in (False, True):
                ts, ops, nbytes, diffs = [], 0, 0, []
                for _ in range(4):
                    a, b, layout, d, ref, valid = generate_m_grouped_masked(
                        groups, max_m, expected_m, n, k, use_ue8m0=True, use_psum_layout=psum)
                    em = int(expected_m * 1.2)
                    if psum:
                        fn = lambda: deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                            a, b, d, layout, use_psum_layout=True, expected_m_for_psum_layout=em)
                    else:
                        fn = lambda: deep_gemm.m_grouped_fp8_gemm_nt_masked(a, b, d, layout, em)
                    fn()
                    diffs.append(calc_diff(d[valid], ref[valid]))
                    valid_m = int(valid.sum().item())
                    t = kernel_us(fn)
                    ts.append(t)
                    ops += 2 * valid_m * n * k
                    rows_a = a[0].size(0) if psum else groups * max_m
                    nbytes += count_bytes(a, d) * valid_m / rows_a + count_bytes(b)
                t = sum(ts) / len(ts)
                rows.append(dict(kind="masked", groups=groups, m=expected_m, n=n, k=k, layout="NT", psum=int(psum),
                                 us=t, tflops=ops / sum(ts) / 1e6, gbps=nbytes / sum(ts) / 1e3,
                                 dense_us=float("nan"), vs_dense=float("nan"), diff=max(diffs)))
                report(rows[-1])


def bench_k_grouped(rows):
    for groups, m, n, expected_k in ((4, 4096, 7168, 8192), (4, 7168, 2048, 8192), (8, 4096, 7168, 4096),
                                     (8, 7168, 2048, 4096), (16, 4096, 7168, 2048), (16, 7168, 2048, 2048)):
        ks = [max(1, int(expected_k * random.uniform(0.7, 1.3))) for _ in range(groups)]
        for gran_k in (32, 128):
            for psum in (False, True):
                deep_gemm.set_mk_alignment_for_contiguous_layout(128)
                total_k, a, b, c, d, ref, layout, host_ks = generate_k_grouped_contiguous(
                    groups, m, n, MajorTypeAB.MNMajor, MajorTypeAB.MNMajor, ks, use_ue8m0=True, gran_k=gran_k,
                    use_psum_layout=psum, k_alignment=128, accumulate=True, out_dtype=torch.float)
                fn = lambda: deep_gemm.k_grouped_fp8_gemm_tn_contiguous(
                    a, b, d, host_ks, layout, d, recipe=(1, 1, gran_k), use_psum_layout=psum)
                d0 = d.clone()
                fn()
                diff = calc_diff(d, ref)
                d.copy_(d0)
                t = kernel_us(fn)
                logical_k = sum(ks)
                rows.append(dict(kind="k_grouped", groups=groups, m=m, n=n, k=logical_k, layout="TN",
                                 psum=int(psum), us=t, tflops=2 * m * n * logical_k / t / 1e6,
                                 gbps=count_bytes(a, b, c, d) / t / 1e3, dense_us=float("nan"),
                                 vs_dense=float("nan"), diff=diff, gran_k=gran_k))
                report(rows[-1])
    deep_gemm.set_mk_alignment_for_contiguous_layout(256)


def report(r):
    extra = f" | dense {r['dense_us']:7.1f} us vs_dense {r['vs_dense']:.3f}" if r["dense_us"] == r["dense_us"] else ""
    print(f"{r['kind']:10} G={r['groups']:2} m={r['m']:5} n={r['n']:5} k={r['k']:6} {r['layout']} psum={r['psum']}"
          f"{' gran_k=' + str(r['gran_k']) if 'gran_k' in r else ''}: {r['us']:8.1f} us {r['tflops']:6.0f} TFLOPS "
          f"{r['gbps']:6.0f} GB/s diff={r['diff']:.1e}{extra}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["all", "contiguous", "masked", "k_grouped"], default="all")
    parser.add_argument("--csv")
    args = parser.parse_args()
    random.seed(0)
    torch.manual_seed(0)
    deep_gemm.set_mk_alignment_for_contiguous_layout(256)
    rows = []
    if args.kind in ("all", "contiguous"):
        bench_contiguous(rows)
    if args.kind in ("all", "masked"):
        bench_masked(rows)
    if args.kind in ("all", "k_grouped"):
        bench_k_grouped(rows)
    if args.csv:
        keys = sorted({k for r in rows for k in r})
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
