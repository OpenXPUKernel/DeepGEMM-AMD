"""Print CDNA4 FP8 GEMM speedups relative to a PyTorch reference."""
import argparse
import torch
import deep_gemm

SHAPES = [(128, 7168, 4096), (4096, 7168, 4096), (4096, 2048, 7168)]


def _inputs(m, n, k):
    xa = torch.arange(max(1, m * k), device="cuda", dtype=torch.float32).reshape(m, k)
    xb = torch.arange(max(1, n * k), device="cuda", dtype=torch.float32).reshape(n, k)
    xa = ((xa % 257) - 128).bfloat16() / 64
    xb = ((xb % 193) - 96).bfloat16() / 48
    return (deep_gemm.per_token_cast_to_fp8(xa, use_ue8m0=True, gran_k=32),
            deep_gemm.per_token_cast_to_fp8(xb, use_ue8m0=True, gran_k=32))


def _reference(a, b):
    aq = a[0].float() * a[1].repeat_interleave(32, dim=1)[:, :a[0].size(1)]
    bq = b[0].float() * b[1].repeat_interleave(32, dim=1)[:, :b[0].size(1)]
    return aq @ bq.transpose(0, 1)


def _time(fn, warmup, rounds):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rounds):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / rounds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    if torch.version.hip is None or not torch.cuda.is_available():
        raise SystemExit("A ROCm CDNA4 device is required")
    for m, n, k in SHAPES:
        a, b = _inputs(m, n, k)
        out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        ref = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        a_ref = (a[0].float() * a[1].repeat_interleave(32, dim=1)[:, :k]).bfloat16()
        b_ref = (b[0].float() * b[1].repeat_interleave(32, dim=1)[:, :k]).bfloat16()
        op = lambda: deep_gemm.fp8_gemm_nt(a, b, out, recipe_a=(1, 32), recipe_b=(1, 32))
        pytorch = lambda: torch.matmul(a_ref, b_ref.transpose(0, 1), out=ref)
        op_ms = _time(op, args.warmup, args.rounds)
        pytorch_ms = _time(pytorch, args.warmup, args.rounds)
        print(f"shape=({m},{n},{k}) speedup_vs_pytorch={pytorch_ms / op_ms:.3f}x")


if __name__ == "__main__":
    main()
