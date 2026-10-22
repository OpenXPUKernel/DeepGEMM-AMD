"""Correctness tests for the CDNA4 HIP BF16 GEMMs (dense, m-grouped contiguous, PSUM and masked
layouts, and K-grouped GEMMs), on the inputs of the CUDA test generators.
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
from generators import (KernelType, MajorTypeAB, generate_k_grouped_contiguous, generate_m_grouped_contiguous,
                        generate_m_grouped_masked, generate_normal)

FULL = os.environ.get("DG_TEST_FULL") == "1"
MAX_DIFF = 1e-5
K, M = MajorTypeAB.KMajor, MajorTypeAB.MNMajor


@pytest.fixture(autouse=True)
def _seed_and_alignment():
    random.seed(0)
    torch.manual_seed(0)
    deep_gemm.set_mk_alignment_for_contiguous_layout(256)
    yield
    deep_gemm.set_mk_alignment_for_contiguous_layout(256)


def _repeat_equal(fn, out, mask=None, times=5):
    first = out.clone()
    for _ in range(times):
        fn()
        assert torch.equal(out[mask], first[mask]) if mask is not None else torch.equal(out, first)


DENSE_SHAPES = ([(m, n, k) for m in (1, 128, 4096) for n, k in
                 [(2112, 7168), (576, 7168), (24576, 1536), (32768, 512), (7168, 16384), (4096, 7168), (7168, 2048)]]
                if FULL else [(1, 2112, 7168), (128, 576, 1536), (300, 1000, 520), (4096, 4096, 2048), (512, 768, 64)])


@pytest.mark.parametrize("m,n,k", DENSE_SHAPES)
@pytest.mark.parametrize("major_a,major_b", [(K, K), (K, M), (M, K), (M, M)], ids=["KK", "KM", "MK", "MM"])
@pytest.mark.parametrize("out_dtype,accumulate", [(torch.bfloat16, False), (torch.float, False), (torch.float, True)],
                         ids=["bf16", "fp32", "fp32_acc"])
def test_bf16_gemm(m, n, k, major_a, major_b, out_dtype, accumulate):
    a, b, c, d, ref = generate_normal(m, n, k, major_a, major_b, accumulate, out_dtype, KernelType.KernelNoSF,
                                      use_bf16=True)
    d0 = d.clone()
    run = lambda: deep_gemm.bf16_gemm_nt(a, b, d, c=c)
    run()
    assert calc_diff(d, ref) < MAX_DIFF
    if not accumulate:
        _repeat_equal(run, d)
    # the layout aliases take the operands as stored
    name = "bf16_gemm_" + ("n" if major_a == K else "t") + ("t" if major_b == K else "n")
    a_s = a if major_a == K else a.T
    b_s = b if major_b == K else b.T
    d2 = d0.clone()
    getattr(deep_gemm, name)(a_s, b_s, d2, c=d2 if accumulate else None)
    assert calc_diff(d2, ref) < MAX_DIFF


def test_bf16_gemm_alpha():
    a, b, c, d, ref = generate_normal(260, 300, 1024, K, K, False, torch.float, KernelType.KernelNoSF,
                                      use_bf16=True, alpha=0.375)
    deep_gemm.bf16_gemm_nt(a, b, d, alpha=0.375)
    assert calc_diff(d, ref) < MAX_DIFF


CONTIGUOUS = [(4, 8192, 6144, 7168), (8, 4096, 4096, 2048)] if FULL else [(4, 1024, 4096, 2048), (3, 700, 576, 1536)]


@pytest.mark.parametrize("groups,expected_m,n,k", CONTIGUOUS)
@pytest.mark.parametrize("use_psum_layout", [False, True])
@pytest.mark.parametrize("major_b", [K, M], ids=["KB", "MB"])
def test_m_grouped_bf16_contiguous(groups, expected_m, n, k, use_psum_layout, major_b):
    m, a, b, layout, d, ref, valid = generate_m_grouped_contiguous(
        groups, expected_m, n, k, K, major_b, use_bf16=True, use_psum_layout=use_psum_layout)
    run = lambda: deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a, b, d, layout, use_psum_layout=use_psum_layout)
    run()
    assert calc_diff(d[valid], ref[valid]) < MAX_DIFF
    _repeat_equal(run, d, valid)
    if major_b == M:
        d_nn = torch.empty_like(d)
        deep_gemm.m_grouped_bf16_gemm_nn_contiguous(a, b.mT, d_nn, layout, use_psum_layout=use_psum_layout)
        assert torch.equal(d_nn[valid], d[valid])


@pytest.mark.parametrize("alignment", [128, 64])
@pytest.mark.parametrize("use_psum_layout", [False, True])
def test_m_grouped_bf16_contiguous_small_alignment(alignment, use_psum_layout):
    deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
    m, a, b, layout, d, ref, valid = generate_m_grouped_contiguous(
        5, 300, 1024, 1536, K, K, use_bf16=True, use_psum_layout=use_psum_layout)
    run = lambda: deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a, b, d, layout, use_psum_layout=use_psum_layout)
    run()
    assert calc_diff(d[valid], ref[valid]) < MAX_DIFF
    _repeat_equal(run, d, valid)


MASKED = [(32, 4096, 192), (6, 4096, 1024), (32, 4096, 20)] if FULL else [(8, 1024, 192), (6, 512, 300)]


@pytest.mark.parametrize("groups,max_m,expected_m", MASKED)
@pytest.mark.parametrize("n,k", [(4096, 2048), (576, 1536)])
@pytest.mark.parametrize("use_psum_layout", [False, True])
def test_m_grouped_bf16_masked(groups, max_m, expected_m, n, k, use_psum_layout):
    a, b, layout, d, ref, valid = generate_m_grouped_masked(groups, max_m, expected_m, n, k, use_bf16=True,
                                                            use_psum_layout=use_psum_layout)
    if use_psum_layout:
        run = lambda: deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a, b, d, layout, use_psum_layout=True)
    else:
        run = lambda: deep_gemm.m_grouped_bf16_gemm_nt_masked(a, b, d, layout, int(expected_m * 1.2))
    run()
    assert calc_diff(d[valid], ref[valid]) < MAX_DIFF
    _repeat_equal(run, d, valid)


K_GROUPED = ([(4, 4096, 7168, 8192), (16, 4096, 7168, 2048), (8, 768, 2048, 128)] if FULL else
             [(8, 768, 2048, 128), (4, 1024, 1536, 2048)])


@pytest.mark.parametrize("groups,m,n,expected_k", K_GROUPED)
@pytest.mark.parametrize("use_psum_layout", [False, True])
@pytest.mark.parametrize("accumulate,out_dtype", [(True, torch.float), (False, torch.float), (False, torch.bfloat16)])
def test_k_grouped_bf16_tn(groups, m, n, expected_k, use_psum_layout, accumulate, out_dtype):
    deep_gemm.set_mk_alignment_for_contiguous_layout(128)
    ks = [max(1, int(expected_k * random.uniform(0.7, 1.3))) for _ in range(groups)]
    ks[0] = 0 if groups > 2 else ks[0]
    total_k, a, b, c, d, ref, layout, host_ks = generate_k_grouped_contiguous(
        groups, m, n, M, M, ks, use_bf16=True, use_psum_layout=use_psum_layout, k_alignment=128,
        accumulate=accumulate, out_dtype=out_dtype)
    out = d.clone() if accumulate else torch.full_like(d, float("nan"))
    run = lambda o, acc: deep_gemm.k_grouped_bf16_gemm_tn_contiguous(a, b, o, host_ks, layout, acc,
                                                                     use_psum_layout=use_psum_layout)
    run(out, out if accumulate else None)
    if accumulate:
        assert calc_diff(out, ref) < MAX_DIFF
    else:
        fp32 = torch.zeros_like(out, dtype=torch.float)
        run(fp32, fp32)
        assert torch.equal(out, fp32.to(out.dtype))
        assert calc_diff(fp32, ref.float()) < MAX_DIFF
