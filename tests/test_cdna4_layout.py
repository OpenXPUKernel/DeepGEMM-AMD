"""Runs the correctness checks of tests/test_layout.py on the CDNA4 backend, the k-grouped and PSUM
layouts (SM100-only in the CUDA test) included; the timings of the CUDA layout kernels are skipped.
"""
import os
import random
import sys

import pytest

try:
    import torch
except Exception as exc:  # pragma: no cover - collection-only environments
    pytest.skip(f"PyTorch is required: {exc}", allow_module_level=True)

if torch.version.hip is None or not torch.cuda.is_available():
    pytest.skip("CDNA4 tests require a ROCm device", allow_module_level=True)

sys.path.insert(0, os.path.dirname(__file__))
import test_layout as cuda_test


@pytest.mark.parametrize("name", ["test_sf_layout_kernels", "test_k_grouped_sf_layout_kernels",
                                  "test_k_grouped_psum_sf_layout_kernels"])
def test_layout_checks(name, monkeypatch):
    monkeypatch.setattr(cuda_test, "get_arch_major", lambda: 10)
    monkeypatch.setattr(cuda_test, "bench_kineto", lambda *args, **kwargs: 1.0)
    torch.manual_seed(1)
    random.seed(1)
    getattr(cuda_test, name)()
