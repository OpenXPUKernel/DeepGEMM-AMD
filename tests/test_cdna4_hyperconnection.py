"""Correctness tests for the CDNA4 tf32_hc_prenorm_gemm on the shapes of tests/test_hyperconnection.py,
against an FP32 reference (the kernel's two BF16 terms of B are more precise than TF32).
"""
import pytest

try:
    import torch
except Exception as exc:  # pragma: no cover - collection-only environments
    pytest.skip(f"PyTorch is required: {exc}", allow_module_level=True)

if torch.version.hip is None or not torch.cuda.is_available():
    pytest.skip("CDNA4 tests require a ROCm device", allow_module_level=True)

import deep_gemm
from deep_gemm.testing import calc_diff


@pytest.mark.parametrize("m", [13, 137, 4096, 8192])
@pytest.mark.parametrize("n,k", [(24, 28672), (24, 7680), (24, 7168), (40, 7000), (64, 1024)])
@pytest.mark.parametrize("num_splits", [None, 16, 3])
def test_hc_prenorm_gemm(m, n, k, num_splits):
    torch.manual_seed(0)
    a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((n, k), dtype=torch.float, device="cuda")
    shape_d, shape_s = ((m, n), (m, )) if num_splits is None else ((num_splits, m, n), (num_splits, m))
    d = torch.full(shape_d, float("nan"), dtype=torch.float, device="cuda")
    s = torch.full(shape_s, float("nan"), dtype=torch.float, device="cuda")
    deep_gemm.tf32_hc_prenorm_gemm(a, b, d, s, num_splits=num_splits)
    final_d = d if num_splits is None else d.sum(0)
    final_s = s if num_splits is None else s.sum(0)
    ref_d = (a.double() @ b.double().T).float()
    ref_s = a.double().square().sum(-1).float()
    assert calc_diff(final_d, ref_d) < 1e-10
    assert calc_diff(final_s, ref_s) < 1e-10
    # repeatable
    d2, s2 = torch.empty_like(d), torch.empty_like(s)
    deep_gemm.tf32_hc_prenorm_gemm(a, b, d2, s2, num_splits=num_splits)
    assert torch.equal(d, d2) and torch.equal(s, s2)
