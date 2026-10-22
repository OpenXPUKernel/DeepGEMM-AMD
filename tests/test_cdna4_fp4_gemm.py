"""Correctness tests for the CDNA4 HIP FP8 x FP4 and FP4 x FP4 GEMMs (dense, m-grouped contiguous,
PSUM and masked layouts, K-grouped FP4), on the inputs and tolerances of the CUDA test generators.
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

import deep_gemm
from deep_gemm.testing import calc_diff
from generators import (KernelType, MajorTypeAB, QuantConfig, generate_k_grouped_contiguous,
                        generate_m_grouped_contiguous, generate_m_grouped_masked, generate_normal)

FULL = os.environ.get("DG_TEST_FULL") == "1"
K, M = MajorTypeAB.KMajor, MajorTypeAB.MNMajor
FP8_FP4, FP4_FP8, FP4_FP4 = (128, 32, False, True), (32, 128, True, False), (32, 32, True, True)
QUANT_IDS = {FP8_FP4: "fp8xfp4", FP4_FP8: "fp4xfp8", FP4_FP4: "fp4xfp4"}


@pytest.fixture(autouse=True)
def _seed_and_alignment():
    random.seed(0)
    torch.manual_seed(0)
    deep_gemm.set_mk_alignment_for_contiguous_layout(256)
    yield
    deep_gemm.set_mk_alignment_for_contiguous_layout(256)


def _repeat_equal(fn, out, mask=None, times=3):
    first = out.clone()
    for _ in range(times):
        fn()
        assert torch.equal(out[mask], first[mask]) if mask is not None else torch.equal(out, first)


def _recipes(q, a, b, m, n):
    """recipe_a / recipe_b of the generated scales: per-row (1, gran_k), or FP8 B in 128x128 blocks"""
    return (1, q.gran_k_a), ((1, q.gran_k_b) if b[1].size(-2) == n else (q.gran_k_b, q.gran_k_b))


NK = ([(2112, 7168), (576, 7168), (24576, 1536), (32768, 512), (7168, 16384), (4096, 7168), (7168, 2048)]
      if FULL else [(2112, 7168), (576, 1536), (1000, 1056), (4096, 2048)])
FORWARD = [(m, n, k) for m in (1, 128, 4096) for n, k in NK] + ([] if FULL else [(300, 1000, 1056), (64, 768, 96)])


@pytest.mark.parametrize("m,n,k", FORWARD)
@pytest.mark.parametrize("quant", [FP8_FP4, FP4_FP8, FP4_FP4], ids=QUANT_IDS.get)
@pytest.mark.parametrize("out_dtype,accumulate", [(torch.bfloat16, False), (torch.float, True)], ids=["bf16", "fp32_acc"])
def test_forward(m, n, k, quant, out_dtype, accumulate):
    q = QuantConfig(quant)
    a, b, c, d, ref = generate_normal(m, n, k, K, K, accumulate, out_dtype, KernelType.Kernel1D1D,
                                      use_ue8m0=True, quant_config=q)
    recipe_a, recipe_b = _recipes(q, a, b, m, n)
    run = lambda: deep_gemm.fp8_fp4_gemm_nt(a, b, d, c=c, recipe_a=recipe_a, recipe_b=recipe_b)
    run()
    assert calc_diff(d, ref) < q.max_diff()
    if not accumulate:
        _repeat_equal(run, d)


BACKWARD = [(4096, n, k) for n, k in NK]


@pytest.mark.parametrize("m,n,k", BACKWARD)
@pytest.mark.parametrize("kind", ["dgrad", "wgrad_fp32_acc", "wgrad_bf16"])
def test_backward_fp8_fp4(m, n, k, kind):
    """The CUDA backward shapes: FP4 B MN-major (dgrad), both operands MN-major (wgrad)."""
    q = QuantConfig(FP8_FP4)
    if kind == "dgrad":
        shape, majors, accumulate, out_dtype = (m, k, n), (K, M), False, torch.bfloat16
    else:
        shape, majors = (n, m, k), (M, M)
        accumulate, out_dtype = (True, torch.float) if kind == "wgrad_fp32_acc" else (False, torch.bfloat16)
    a, b, c, d, ref = generate_normal(*shape, *majors, accumulate, out_dtype, KernelType.Kernel1D1D,
                                      use_ue8m0=True, quant_config=q)
    recipe_a, recipe_b = _recipes(q, a, b, shape[0], shape[1])
    deep_gemm.fp8_fp4_gemm_nt(a, b, d, c=c, recipe_a=recipe_a, recipe_b=recipe_b)
    assert calc_diff(d, ref) < q.max_diff()


CONTIGUOUS = [(4, 8192, 6144, 7168), (8, 4096, 4096, 2048)] if FULL else [(4, 1024, 4096, 2048), (3, 700, 576, 1536)]


@pytest.mark.parametrize("groups,expected_m,n,k", CONTIGUOUS)
@pytest.mark.parametrize("use_psum_layout", [False, True])
@pytest.mark.parametrize("quant,major_b", [(FP8_FP4, K), (FP8_FP4, M), (FP4_FP4, K)],
                         ids=["fp8xfp4-KB", "fp8xfp4-MB", "fp4xfp4-KB"])
def test_m_grouped_contiguous(groups, expected_m, n, k, use_psum_layout, quant, major_b):
    q = QuantConfig(quant)
    m, a, b, layout, d, ref, valid = generate_m_grouped_contiguous(
        groups, expected_m, n, k, K, major_b, use_ue8m0=True, use_psum_layout=use_psum_layout, quant_config=q)
    recipe_a, recipe_b = _recipes(q, a, b, m, n)
    run = lambda: deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
        a, b, d, layout, recipe_a=recipe_a, recipe_b=recipe_b, use_psum_layout=use_psum_layout)
    run()
    assert calc_diff(d[valid], ref[valid]) < q.max_diff()
    _repeat_equal(run, d, valid)


@pytest.mark.parametrize("alignment", [128, 64])
def test_m_grouped_contiguous_small_alignment(alignment):
    deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
    q = QuantConfig(FP8_FP4)
    m, a, b, layout, d, ref, valid = generate_m_grouped_contiguous(
        5, 300, 1024, 1536, K, K, use_ue8m0=True, use_psum_layout=True, quant_config=q)
    deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(a, b, d, layout, recipe_a=(1, 128), recipe_b=(1, 32),
                                                   use_psum_layout=True)
    assert calc_diff(d[valid], ref[valid]) < q.max_diff()


MASKED = [(32, 4096, 192), (6, 4096, 1024), (32, 4096, 20), (6, 4096, 20)] if FULL else [(8, 1024, 192), (6, 512, 20)]


@pytest.mark.parametrize("groups,max_m,expected_m", MASKED)
@pytest.mark.parametrize("n,k", [(4096, 2048), (576, 1536)])
@pytest.mark.parametrize("use_psum_layout", [False, True])
@pytest.mark.parametrize("quant", [FP8_FP4, FP4_FP4], ids=QUANT_IDS.get)
def test_m_grouped_masked(groups, max_m, expected_m, n, k, use_psum_layout, quant):
    q = QuantConfig(quant)
    a, b, layout, d, ref, valid = generate_m_grouped_masked(groups, max_m, expected_m, n, k, use_ue8m0=True,
                                                            use_psum_layout=use_psum_layout, quant_config=q)
    recipe_a, recipe_b = (1, q.gran_k_a), (1, q.gran_k_b)
    if use_psum_layout:
        run = lambda: deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
            a, b, d, layout, recipe_a=recipe_a, recipe_b=recipe_b, use_psum_layout=True,
            expected_m_for_psum_layout=int(expected_m * 1.2))
    else:
        run = lambda: deep_gemm.m_grouped_fp8_fp4_gemm_nt_masked(a, b, d, layout, int(expected_m * 1.2),
                                                                 recipe_a=recipe_a, recipe_b=recipe_b)
    run()
    assert calc_diff(d[valid], ref[valid]) < q.max_diff()
    _repeat_equal(run, d, valid)


K_GROUPED = ([(8, 768, 2048, 128), (4, 4096, 7168, 8192), (16, 4096, 7168, 2048)] if FULL else
             [(8, 768, 2048, 128), (4, 1024, 1536, 2048)])


@pytest.mark.parametrize("groups,m,n,expected_k", K_GROUPED)
@pytest.mark.parametrize("k_alignment", [256, 512])
@pytest.mark.parametrize("use_psum_layout", [False, True])
@pytest.mark.parametrize("accumulate,out_dtype", [(True, torch.float), (False, torch.float), (False, torch.bfloat16)])
def test_k_grouped_fp4(groups, m, n, expected_k, k_alignment, use_psum_layout, accumulate, out_dtype):
    q = QuantConfig(FP4_FP4)
    ks = [max(1, int(expected_k * random.uniform(0.7, 1.3))) for _ in range(groups)]
    ks[0] = 0 if groups > 2 else ks[0]
    deep_gemm.set_mk_alignment_for_contiguous_layout(k_alignment)
    total_k, a, b, c, d, ref, layout, host_ks = generate_k_grouped_contiguous(
        groups, m, n, K, K, ks, use_ue8m0=True, gran_k=32, quant_config=q, use_psum_layout=use_psum_layout,
        k_alignment=k_alignment, accumulate=accumulate, out_dtype=out_dtype)
    out = d.clone() if accumulate else torch.full_like(d, float("nan"))
    deep_gemm.k_grouped_fp4_gemm_nt_contiguous(a, b, out, host_ks, layout, out if accumulate else None,
                                               use_psum_layout=use_psum_layout)
    assert calc_diff(out, ref) < q.max_diff()
