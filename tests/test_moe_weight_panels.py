"""CPU byte-address and format-contract checks, not GPU numerical validation."""

import ast
import os
import random
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import patch as patch


def load_contract():
    source = Path(__file__).resolve().parents[1] / "deep_gemm/_hip_mega_moe.py"
    tree = ast.parse(source.read_text())
    names = {
        "ResidentMoEPanelWeights",
        "prepare_weights_for_resident_mega_moe",
        "transform_weights_for_mega_moe",
        "_mega_moe",
    }
    module = ast.Module(
        body=[
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
        ],
        type_ignores=[],
    )

    class Weight:
        dtype = "int8"
        ndim = 3
        device = "gpu"

        def is_contiguous(self):
            return True

    torch = SimpleNamespace(Tensor=Weight, uint8="uint8", int8="int8", int32="int32")
    calls = []

    def prepare(w):
        calls.append(w)
        return Weight()

    env = dict(
        torch=torch,
        NamedTuple=NamedTuple,
        os=os,
        SymmBuffer=object,
        _C=SimpleNamespace(cdna4_prepare_moe_weight_panels=prepare),
    )
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), env)
    return env, Weight, calls


class WeightPanelTests(unittest.TestCase):
    def test_vector_permutation_is_bijective_and_keeps_nibbles_together(self):
        for e, n, kb in ((1, 256, 64), (2, 512, 128), (3, 256, 256)):
            visited = set()
            for chunk in range(e * n * kb // 16):
                panel = chunk // 1024
                row = (chunk // 4) % 256
                column = (chunk % 4) * 16
                kp = panel % (kb // 64)
                np = (panel // (kb // 64)) % (n // 256)
                expert = panel // ((kb // 64) * (n // 256))
                source = ((expert * n + np * 256 + row) * kb) + kp * 64 + column
                self.assertEqual(source % 16, 0)
                self.assertNotIn(source, visited)
                visited.add(source)
                expected = (
                    ((expert * (n // 256) + np) * (kb // 64) + kp) * 256 + row
                ) * 64 + column
                self.assertEqual(expected, chunk * 16)
            self.assertEqual(visited, set(range(0, e * n * kb, 16)))

    def test_native_dma_reads_same_original_bytes_after_permutation(self):
        rng = random.Random(17)
        for n, k in ((4608, 5120), (5120, 2304), (256, 128)):
            for _ in range(10000):
                expert = rng.randrange(48)
                n0 = rng.randrange(n // 256) * 256
                kp = rng.randrange(k // 128)
                j = rng.randrange(4)
                wave = rng.randrange(4)
                lane = rng.randrange(64)
                br = 16 * wave + lane // 4
                swizzle = (br >> 2) & 3
                row = j * 64 + br
                offset = ((lane % 4) ^ swizzle) * 16
                physical = (
                    expert * n * (k // 2)
                    + n0 * (k // 2)
                    + kp * 256 * 64
                    + row * 64
                    + offset
                )
                panel, within = divmod(physical, 256 * 64)
                row2, offset2 = divmod(within, 64)
                kp2 = panel % (k // 128)
                np = (panel // (k // 128)) % (n // 256)
                e2 = panel // ((k // 128) * (n // 256))
                original = (e2 * n + np * 256 + row2) * (k // 2) + kp2 * 64 + offset2
                expected = (expert * n + n0 + row) * (k // 2) + kp * 64 + offset
                self.assertEqual(original, expected)

    def test_preparation_is_explicit_idempotent_and_preserves_scale_identity(self):
        env, Weight, calls = load_contract()
        prepare = env["prepare_weights_for_resident_mega_moe"]
        sf = SimpleNamespace(dtype="int32", device="gpu")
        pairs = ((Weight(), sf), (Weight(), sf))
        result = prepare(*pairs)
        self.assertEqual(len(calls), 2)
        for original, prepared in zip(pairs, result):
            self.assertIsInstance(prepared, env["ResidentMoEPanelWeights"])
            self.assertIs(prepared.scales, original[1])
            self.assertIsNot(prepared.weight, original[0])
        self.assertEqual(prepare(*result), result)
        self.assertEqual(len(calls), 2)
        with self.assertRaises(ValueError):
            env["transform_weights_for_mega_moe"](*result)

    def test_incompatible_formats_fail_before_any_device_work(self):
        env, Weight, calls = load_contract()
        prepare = env["prepare_weights_for_resident_mega_moe"]
        sf = SimpleNamespace(dtype="int32", device="gpu")
        weight = Weight()
        for bad in (weight, (weight,), [weight, sf]):
            with self.assertRaises(ValueError):
                prepare(bad, (weight, sf))
        weight.dtype = "fp8"
        with self.assertRaises(ValueError):
            prepare((weight, sf), (weight, sf))
        self.assertEqual(calls, [])
        weight.dtype = "int8"
        panel = env["ResidentMoEPanelWeights"](weight, sf)
        buffer = SimpleNamespace(mma_type="fp8xfp4")
        invoke = lambda a, b: env["_mega_moe"](
            None, a, b, buffer, None, None, None, "swiglu", 10, True, True
        )
        with self.assertRaises(ValueError):
            invoke(panel, (weight, sf))


if __name__ == "__main__":
    unittest.main()
