"""CPU-only benchmark catalog, CLI, and result-format regressions.

    python tests/test_cdna4_benchmark.py
"""

import contextlib
import csv
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


BENCHMARK = Path(__file__).with_name('bench_cdna4_fp8.py')
spec = importlib.util.spec_from_file_location('cdna4_benchmark_under_test', BENCHMARK)
bench = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bench
spec.loader.exec_module(bench)

# These are the tutorial's public workload contracts, independent of the
# benchmark's constants so accidentally dropping a workload fails this check.
TUTORIAL_NK = (
    (2112, 7168), (576, 7168), (24576, 1536), (32768, 512),
    (7168, 16384), (4096, 7168), (7168, 2048),
)
TUTORIAL_EDGES = {
    (1, 1, 32), (17, 33, 96), (300, 500, 1000), (257, 255, 2000),
    (4097, 4099, 4160), (1000, 3000, 7000), (5, 8192, 136),
    (8192, 5, 136), (64, 64, 16),
}


def workload(case):
    """Compare actual work independently of the name assigned to its suite."""
    return case.m, case.n, case.k, case.dtype, case.accumulate, case.layout


class BenchmarkCatalogTests(unittest.TestCase):
    def test_tutorial_covers_all_forward_and_backward_configurations(self):
        expected = {
            'forward': {(m, n, k, 'bf16', False, 'KK')
                        for m in (1, 128, 4096) for n, k in TUTORIAL_NK},
            'dgrad': {(4096, k, n, 'bf16', False, layout)
                      for n, k in TUTORIAL_NK for layout in ('KK', 'KM')},
            'wgrad': {(n, 4096, k, 'fp32', True, layout)
                      for n, k in TUTORIAL_NK for layout in ('KK', 'MM')},
        }
        cases = bench.select_cases(bench.parse_args(['--suite', 'tutorial']))
        self.assertEqual(len(cases), 52)
        for suite, contract in expected.items():
            with self.subTest(suite=suite):
                self.assertEqual({workload(c) for c in cases if c.suite == suite}, contract)
                selected = bench.select_cases(bench.parse_args(['--suite', suite]))
                self.assertEqual({workload(c) for c in selected}, contract)

    def test_default_expands_tutorial_without_duplicate_workloads(self):
        cases = bench.select_cases(bench.parse_args([]))
        tutorial = bench.select_cases(bench.parse_args(['--suite', 'tutorial']))
        self.assertGreaterEqual(len(cases), 129)
        self.assertGreaterEqual(len({(c.m, c.n, c.k) for c in cases}), 93)
        self.assertTrue({workload(c) for c in tutorial} <= {workload(c) for c in cases})
        self.assertEqual(len(cases), len(set(cases)))
        self.assertEqual(len(cases), len({(*workload(c), c.gran_k_a, c.gran_k_b) for c in cases}))

    def test_all_edge_shapes_cover_bf16_and_fp32_accumulation(self):
        cases = bench.select_cases(bench.parse_args(['--suite', 'edge']))
        for dtype, accumulate in (('bf16', False), ('fp32', True)):
            with self.subTest(dtype=dtype):
                expected = {(*shape, dtype, accumulate, 'KK') for shape in TUTORIAL_EDGES}
                self.assertTrue(expected <= {workload(c) for c in cases})

    def test_native_scale_recipe_coverage(self):
        cases = bench.select_cases(bench.parse_args(['--suite', 'recipe']))
        self.assertEqual({(c.layout, c.gran_k_a, c.gran_k_b) for c in cases},
                         {(layout, ga, gb) for layout in ('KK', 'KM', 'MK', 'MM')
                          for ga, gb in ((128, 128), (32, 128), (128, 32))})
        custom = bench.select_cases(bench.parse_args(
            ['--shape', '17,33,96', '--gran-k-a', '128', '--gran-k-b', '32']))
        self.assertEqual((custom[0].gran_k_a, custom[0].gran_k_b), (128, 32))

    def test_custom_shapes_select_dtype_layout_and_accumulation(self):
        shapes = ['--shape', '17,33,96', '--shape', '64x64x16']
        default = bench.select_cases(bench.parse_args(shapes))
        self.assertEqual([workload(c) for c in default], [
            (17, 33, 96, 'bf16', False, 'KK'), (64, 64, 16, 'bf16', False, 'KK'),
        ])
        for layout in ('KK', 'KM', 'MK', 'MM'):
            with self.subTest(layout=layout):
                args = bench.parse_args(shapes + ['--dtype', 'both', '--layout', layout, '--accumulate'])
                self.assertEqual({workload(c) for c in bench.select_cases(args)}, {
                    (*shape, dtype, True, layout)
                    for shape in ((17, 33, 96), (64, 64, 16)) for dtype in ('bf16', 'fp32')
                })

    def test_dtype_filter_preserves_selected_workloads(self):
        tutorial = bench.select_cases(bench.parse_args(['--suite', 'tutorial']))
        for dtype in ('bf16', 'fp32'):
            with self.subTest(dtype=dtype):
                selected = bench.select_cases(bench.parse_args(['--suite', 'tutorial', '--dtype', dtype]))
                self.assertEqual(selected, [c for c in tutorial if c.dtype == dtype])
        with self.assertRaisesRegex(SystemExit, 'No cases match'):
            bench.main(['--suite', 'wgrad', '--dtype', 'bf16', '--list-shapes'])

    def test_invalid_arguments_fail_before_running_a_benchmark(self):
        invalid = [
            ['--shape', '1,2'], ['--shape', '1,2,3,4'], ['--shape', '0,2,3'],
            ['--shape=-1,2,3'], ['--shape', 'one,2,3'], ['--suite', 'unknown'],
            ['--dtype', 'fp16'], ['--shape', '1,2,3', '--layout', 'unknown'],
            ['--layout', 'MM'], ['--accumulate'], ['--rounds', '0'],
            ['--rounds', '-1'], ['--rounds', 'many'], ['--warmup', '0'],
            ['--warmup-ms', '-1'], ['--warmup-ms', 'nan'], ['--warmup-ms', 'inf'],
        ]
        for argv in invalid:
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as result:
                    bench.parse_args(argv)
                self.assertEqual(result.exception.code, 2)

    def test_list_shapes_requires_only_the_standard_library(self):
        # Isolated mode ignores PYTHONPATH; -S disables site packages, including
        # PyTorch. This exercises the actual CLI without a GPU or mocked imports.
        result = subprocess.run(
            [sys.executable, '-I', '-S', str(BENCHMARK), '--list-shapes'],
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        self.assertIn('Cases:', lines[0])
        self.assertGreaterEqual(len(lines) - 1, 129)
        self.assertTrue(any('wgrad' in line and 'fp32' in line and 'acc=1 layout=MM' in line
                            for line in lines))


class BenchmarkResultTests(unittest.TestCase):
    def test_throughput_uses_microseconds_and_two_operations_per_fma(self):
        metrics = bench.performance_metrics(100, 200, 300, op_us=4, pytorch_us=12)
        self.assertEqual(metrics['deepgemm_us'], 4)
        self.assertEqual(metrics['pytorch_us'], 12)
        self.assertAlmostEqual(metrics['deepgemm_tflops'], 3.0)
        self.assertAlmostEqual(metrics['pytorch_tflops'], 1.0)
        self.assertAlmostEqual(metrics['speedup'], 3.0)

    def test_csv_and_console_honor_relative_only(self):
        absolute_fields = {'deepgemm_us', 'pytorch_us', 'deepgemm_tflops', 'pytorch_tflops'}
        relative_fields = {'suite', 'm', 'n', 'k', 'dtype', 'accumulate', 'layout',
                           'gran_k_a', 'gran_k_b', 'baseline', 'status', 'relative_l2', 'speedup'}
        result = dict(status='ok', relative_l2=0.002,
                      **bench.performance_metrics(100, 200, 300, op_us=4, pytorch_us=12))
        # Replace only the GPU execution boundary; exercise the real CLI result
        # reporting and CSV serialization, including removal of absolute values.
        torch = types.SimpleNamespace(
            version=types.SimpleNamespace(hip='test'),
            cuda=types.SimpleNamespace(is_available=lambda: True),
            backends=types.SimpleNamespace(cuda=types.SimpleNamespace(
                matmul=types.SimpleNamespace(allow_tf32=True))),
        )
        for relative_only in (False, True):
            with self.subTest(relative_only=relative_only), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'results.csv'
                argv = ['--shape', '100,200,300', '--dtype', 'fp32', '--accumulate',
                        '--csv', str(path), '--no-flush-cache']
                if relative_only:
                    argv.append('--relative-only')
                stdout = io.StringIO()
                with patch.dict(sys.modules, {'torch': torch, 'deep_gemm': types.ModuleType('deep_gemm')}), \
                        patch.object(bench, 'run_case', return_value=result), \
                        contextlib.redirect_stdout(stdout):
                    self.assertEqual(bench.main(argv), 0)
                with path.open(newline='') as stream:
                    reader = csv.DictReader(stream)
                    fields = set(reader.fieldnames)
                    rows = list(reader)
                self.assertEqual(fields, relative_fields if relative_only else relative_fields | absolute_fields)
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]['baseline'], 'pytorch_fp32_addmm')
                self.assertEqual(rows[0]['status'], 'ok')
                self.assertEqual(float(rows[0]['speedup']), 3.0)
                self.assertIn('speedup=3.000x', stdout.getvalue())
                if relative_only:
                    self.assertNotIn('TFLOPS', stdout.getvalue())
                    self.assertNotRegex(stdout.getvalue(), r'\b\d+(?:\.\d+)? us\b')
                else:
                    self.assertEqual(float(rows[0]['deepgemm_tflops']), 3.0)
                    self.assertIn('TFLOPS', stdout.getvalue())


if __name__ == '__main__':
    unittest.main()
