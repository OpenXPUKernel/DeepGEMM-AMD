"""Benchmark CDNA4 FP8 GEMM against PyTorch with a full shape/layout catalog.

Run --list-shapes without PyTorch or a GPU to inspect the catalog. Timings are
GPU events around the public APIs, including launch gaps and layout conversion.
Quantization, scale packing, baseline preparation and JIT warmup are excluded.
"""
import argparse
import csv
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import statistics
import time


FORWARD_NK = (
    (2112, 7168), (576, 7168), (24576, 1536), (32768, 512),
    (7168, 16384), (4096, 7168), (7168, 2048),
)
EDGE_SHAPES = (
    (1, 1, 32), (17, 33, 96), (300, 500, 1000), (257, 255, 2000),
    (4097, 4099, 4160), (1000, 3000, 7000), (5, 8192, 136),
    (8192, 5, 136), (64, 64, 16),
)
SQUARE_SHAPES = tuple((size, size, size) for size in (128, 256, 512, 1024, 2048, 4096, 8192)) + (
    (16384, 16384, 8192),
)


@dataclass(frozen=True)
class Case:
    suite: str
    m: int
    n: int
    k: int
    dtype: str = 'bf16'
    accumulate: bool = False
    layout: str = 'KK'


def enumerate_cases(suite='all'):
    """49 tutorial performance configurations plus 68 additional configurations."""
    cases = []
    if suite in ('all', 'tutorial', 'forward'):
        cases += [Case('forward', m, n, k) for m in (1, 128, 4096) for n, k in FORWARD_NK]
    if suite in ('all', 'tutorial', 'dgrad'):
        cases += [Case('dgrad', 4096, k, n, layout=layout)
                  for n, k in FORWARD_NK for layout in ('KK', 'KM')]
    if suite in ('all', 'tutorial', 'wgrad'):
        cases += [Case('wgrad', n, 4096, k, 'fp32', True, layout)
                  for n, k in FORWARD_NK for layout in ('KK', 'MM')]
    if suite in ('all', 'extended'):
        cases += [Case('extended', m, n, k) for m in (8, 16, 32, 64, 256, 1024)
                  for n, k in FORWARD_NK]
    if suite in ('all', 'square'):
        cases += [Case('square', *shape) for shape in SQUARE_SHAPES]
    if suite in ('all', 'edge'):
        cases += [Case('edge', *shape, dtype=dtype, accumulate=dtype == 'fp32')
                  for shape in EDGE_SHAPES for dtype in ('bf16', 'fp32')]
    return cases


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return result


def shape_arg(value):
    try:
        result = tuple(int(x) for x in value.lower().replace('x', ',').split(','))
    except ValueError as exc:
        raise argparse.ArgumentTypeError('expected M,N,K or MxNxK') from exc
    if len(result) != 3 or min(result) <= 0:
        raise argparse.ArgumentTypeError('expected three positive dimensions')
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=('all', 'tutorial', 'forward', 'dgrad', 'wgrad',
                                           'extended', 'square', 'edge'), default='all')
    parser.add_argument('--shape', type=shape_arg, action='append', help='custom M,N,K; repeat to add shapes')
    parser.add_argument('--dtype', choices=('bf16', 'fp32', 'both'),
                        help='filter the catalog, or select types for custom shapes (default: BF16)')
    parser.add_argument('--layout', choices=('KK', 'KM', 'MK', 'MM'), default='KK',
                        help='custom shapes: K means K-major, M means MN-major')
    parser.add_argument('--accumulate', action='store_true', help='custom shapes: accumulate into D')
    parser.add_argument('--rounds', type=positive_int, default=30, help='timed event samples per API')
    parser.add_argument('--warmup', type=positive_int, default=5, help='minimum warmup calls per API')
    parser.add_argument('--warmup-ms', type=float, default=200.0, help='minimum warmup duration per API')
    parser.add_argument('--no-flush-cache', action='store_true', help='reuse warm caches between samples')
    parser.add_argument('--relative-only', action='store_true', help='omit absolute timing/throughput fields')
    parser.add_argument('--csv', type=Path, help='save all result rows, including failures and OOM skips')
    parser.add_argument('--list-shapes', action='store_true', help='list selected cases without loading PyTorch')
    args = parser.parse_args(argv)
    if not math.isfinite(args.warmup_ms) or args.warmup_ms < 0:
        parser.error('--warmup-ms must be finite and nonnegative')
    if not args.shape and (args.layout != 'KK' or args.accumulate):
        parser.error('--layout and --accumulate require --shape')
    return args


def select_cases(args):
    if args.shape:
        dtypes = ('bf16', 'fp32') if args.dtype == 'both' else (args.dtype or 'bf16',)
        return [Case('custom', *shape, dtype, args.accumulate, args.layout)
                for shape in args.shape for dtype in dtypes]
    cases = enumerate_cases(args.suite)
    if args.dtype and args.dtype != 'both':
        cases = [case for case in cases if case.dtype == args.dtype]
    return cases


def metric_fields(relative_only):
    fields = ['suite', 'm', 'n', 'k', 'dtype', 'accumulate', 'layout', 'baseline', 'status', 'relative_l2']
    if not relative_only:
        fields += ['deepgemm_us', 'pytorch_us', 'deepgemm_tflops', 'pytorch_tflops']
    return fields + ['speedup']


def performance_metrics(m, n, k, op_us, pytorch_us):
    # Two arithmetic operations per multiply-add; timings use microseconds.
    ops = 2 * m * n * k
    return dict(deepgemm_us=op_us, pytorch_us=pytorch_us,
                deepgemm_tflops=ops / op_us / 1e6,
                pytorch_tflops=ops / pytorch_us / 1e6, speedup=pytorch_us / op_us)


def time_call(torch, fn, rounds, warmup, warmup_ms, flush_buffer):
    start = time.perf_counter()
    calls = 0
    while calls < warmup or (time.perf_counter() - start) * 1000 < warmup_ms:
        fn()
        torch.cuda.synchronize()
        calls += 1
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
              for _ in range(rounds)]
    for begin, end in events:
        if flush_buffer is not None:
            flush_buffer.zero_()
        begin.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    elapsed = statistics.median(begin.elapsed_time(end) * 1000 for begin, end in events)
    if elapsed <= 0 or not math.isfinite(elapsed):
        raise RuntimeError('invalid GPU event timing')
    return elapsed


def run_case(torch, deep_gemm, case, args, flush_buffer):
    dtype = torch.bfloat16 if case.dtype == 'bf16' else torch.float32

    def operand(rows, major):
        x = torch.randn((rows, case.k), device='cuda', dtype=torch.bfloat16)
        q, sf = deep_gemm.per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=32)
        reference = (q.float() * sf.repeat_interleave(32, dim=1)[:, :case.k]).to(dtype)
        packed = deep_gemm.transform_sf_into_required_layout(sf, rows, case.k, (1, 32))
        if major == 'M':
            q = q.T.contiguous().T
            reference = reference.T.contiguous().T
        return (q, packed), reference

    a, a_ref = operand(case.m, case.layout[0])
    b, b_ref = operand(case.n, case.layout[1])
    out = torch.zeros((case.m, case.n), device='cuda', dtype=dtype)
    ref_out = torch.zeros_like(out)

    def op():
        deep_gemm.fp8_gemm_nt(a, b, out, c=out if case.accumulate else None, recipe=(1, 1, 32))

    def pytorch():
        if case.accumulate:
            torch.addmm(ref_out, a_ref, b_ref.T, out=ref_out)
        else:
            torch.matmul(a_ref, b_ref.T, out=ref_out)

    # Check distributed samples before repeated in-place accumulation. Use an
    # FP32 reference on the dequantized operands, independently of timed dtype.
    rows = torch.linspace(0, case.m - 1, min(case.m, 32), device='cuda').long().unique()
    cols = torch.linspace(0, case.n - 1, min(case.n, 32), device='cuda').long().unique()
    expected = a_ref.index_select(0, rows).float() @ b_ref.index_select(0, cols).float().T
    op()
    got = out.index_select(0, rows).index_select(1, cols).float()
    error = float(((got - expected).square().sum() / expected.square().sum().clamp_min(1e-20)).sqrt())
    if not math.isfinite(error) or error > 0.01:
        return dict(status='accuracy_failed', relative_l2=error)

    # The baseline gets identical operands/layouts and zero initial accumulation.
    out.zero_()
    ref_out.zero_()
    op_us = time_call(torch, op, args.rounds, args.warmup, args.warmup_ms, flush_buffer)
    pytorch_us = time_call(torch, pytorch, args.rounds, args.warmup, args.warmup_ms, flush_buffer)
    return dict(status='ok', relative_l2=error,
                **performance_metrics(case.m, case.n, case.k, op_us, pytorch_us))


def main(argv=None):
    args = parse_args(argv)
    cases = select_cases(args)
    if not cases:
        raise SystemExit('No cases match the selection')
    unique = len({(case.m, case.n, case.k) for case in cases})
    print(f'Cases: {len(cases)}; unique shapes: {unique}', flush=True)
    if args.list_shapes:
        for case in cases:
            print(f'{case.suite:8} {case.m:6} {case.n:6} {case.k:6} '
                  f'{case.dtype:4} acc={int(case.accumulate)} layout={case.layout}')
        return 0

    import torch
    import deep_gemm
    if torch.version.hip is None or not torch.cuda.is_available():
        raise SystemExit('A ROCm PyTorch environment with a CDNA4 device is required')
    torch.backends.cuda.matmul.allow_tf32 = False
    print('Architecture: CDNA4; timing: median GPU events around public APIs; '
          f'cache: {"warm" if args.no_flush_cache else "flushed"}', flush=True)
    print('PyTorch uses the listed dtype and layout with matmul or in-place addmm. '
          'K=K-major, M=MN-major; accuracy is sampled relative L2 error.', flush=True)
    if not args.relative_only:
        print('Absolute fields: API latency in us; effective throughput in TFLOPS (2*M*N*K/time).', flush=True)
    flush_buffer = None if args.no_flush_cache else torch.empty(256_000_000, device='cuda', dtype=torch.uint8)
    results = []
    output_file = None
    try:
        writer = None
        if args.csv:
            args.csv.parent.mkdir(parents=True, exist_ok=True)
            output_file = args.csv.open('w', newline='')
            writer = csv.DictWriter(output_file, fieldnames=metric_fields(args.relative_only),
                                    extrasaction='ignore', lineterminator='\n')
            writer.writeheader()
        for index, case in enumerate(cases, 1):
            row = dict(asdict(case), baseline=f'pytorch_{case.dtype}_{"addmm" if case.accumulate else "matmul"}')
            try:
                row.update(run_case(torch, deep_gemm, case, args, flush_buffer))
            except torch.OutOfMemoryError:
                # Keep the missing row visible without printing device configuration.
                torch.cuda.empty_cache()
                row.update(status='oom')
            except Exception as exc:
                row.update(status='error')
                print(f'Case {index}: {type(exc).__name__}', flush=True)
            results.append(row)
            prefix = (f'[{index:3}/{len(cases)}] {case.suite:8} '
                      f'({case.m},{case.n},{case.k}) {case.dtype} acc={int(case.accumulate)} {case.layout}')
            if row['status'] == 'ok':
                absolute = '' if args.relative_only else (
                    f' DG={row["deepgemm_us"]:.2f} us / {row["deepgemm_tflops"]:.2f} TFLOPS'
                    f' PyTorch={row["pytorch_us"]:.2f} us / {row["pytorch_tflops"]:.2f} TFLOPS')
                print(f'{prefix}{absolute} speedup={row["speedup"]:.3f}x '
                      f'rel_l2={row["relative_l2"]:.3e}', flush=True)
            else:
                print(f'{prefix} status={row["status"]}', flush=True)
            if writer:
                writer.writerow(row)
                output_file.flush()
    finally:
        if output_file:
            output_file.close()

    good = [row for row in results if row['status'] == 'ok']
    for suite in dict.fromkeys(row['suite'] for row in results):
        suite_rows = [row for row in good if row['suite'] == suite]
        if suite_rows:
            speedup = math.exp(statistics.mean(math.log(row['speedup']) for row in suite_rows))
            print(f'{suite}: {len(suite_rows)} successful cases, geometric mean speedup={speedup:.3f}x')
    print(f'Completed: {len(good)}/{len(results)} successful; failures/skips remain in the result table.')
    return 0 if len(good) == len(results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
