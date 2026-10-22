"""CPU-only checks for benchmark hardware-information publication defaults."""

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("amd_benchmark_metrics", ROOT / "_amd_benchmark_metrics.py")
METRICS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(METRICS)


class BenchmarkMetricTests(unittest.TestCase):
    def test_no_hardware_denominator_or_derived_utilization_by_default(self):
        self.assertEqual(METRICS.peak_metrics(1e12, 1, 2, None, 0.5), {})

    def test_explicit_private_normalization_retains_the_original_formula(self):
        result = METRICS.peak_metrics(1e12, 1, 2, 1, 0.5)
        self.assertEqual(result["peak_tflops_per_gpu"], 1)
        self.assertEqual(result["mfu"], 0.5)
        self.assertTrue(result["target_met"])
        self.assertEqual(result["target_latency_ms"], 1000)

    def test_invalid_explicit_normalization_fails_closed(self):
        for value in (0, -1, float("inf"), float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                METRICS.peak_metrics(1e12, 1, 2, value, 0.5)

    def test_published_validation_has_no_peak_normalized_fields(self):
        import json
        payload = json.loads((ROOT.parent / "docs/validation/amd-hip-jit.json").read_text())

        def walk(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    self.assertNotIn(key, {"mfu", "peak_tflops_per_gpu", "target_latency_ms"})
                    walk(child)
            elif isinstance(value, list):
                for child in value:
                    walk(child)

        walk(payload)
        self.assertEqual(payload["environment"]["gpu"], "AMD GPU")


if __name__ == "__main__":
    unittest.main()
