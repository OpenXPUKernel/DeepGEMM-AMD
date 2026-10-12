"""Correctness tests for the CDNA4 HIP grouped FP8 GEMMs (m-grouped contiguous, PSUM and
masked layouts, and K-grouped GEMMs).

The inputs come from the CUDA test generators with UE8M0 scales (the CDNA4 scaled MFMA format).
The default shapes are reduced; set DG_TEST_FULL=1 for the shapes of tests/test_fp8_fp4.py.
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
from deep_gemm.utils import align, pack_ue8m0_to_int
from generators import (MajorTypeAB, QuantConfig, generate_k_grouped_contiguous,
                        generate_m_grouped_contiguous, generate_m_grouped_masked)

FULL = os.environ.get("DG_TEST_FULL") == "1"
LEGACY = QuantConfig()                          # A 1x128, B 128x128 scale blocks
MX = QuantConfig((32, 32, False, False))        # A 1x32, B 32x32 scale blocks
MAX_DIFF = 1e-3


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
        if mask is None:
            assert torch.equal(out, first)
        else:
            assert torch.equal(out[mask], first[mask])


CONTIGUOUS_SHAPES = ([(4, 8192, 6144, 7168), (8, 4096, 7168, 3072), (8, 4096, 4096, 4096), (4, 8192, 4096, 2048)]
                     if FULL else [(4, 1024, 4096, 2048), (3, 700, 576, 1536)])


@pytest.mark.parametrize("groups,expected_m,n,k", CONTIGUOUS_SHAPES)
@pytest.mark.parametrize("use_psum_layout", [False, True])
@pytest.mark.parametrize("major_b", [MajorTypeAB.KMajor, MajorTypeAB.MNMajor])
@pytest.mark.parametrize("quant", [LEGACY, MX], ids=["legacy", "mx"])
def test_m_grouped_contiguous(groups, expected_m, n, k, use_psum_layout, major_b, quant):
    m, a, b, layout, d, ref, valid = generate_m_grouped_contiguous(
        groups, expected_m, n, k, MajorTypeAB.KMajor, major_b, use_ue8m0=True,
        use_psum_layout=use_psum_layout, quant_config=quant)
    recipe, recipe_a, recipe_b = quant.get_recipes()
    run = lambda: deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
        a, b, d, layout, recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b,
        use_psum_layout=use_psum_layout)
    run()
    assert calc_diff(d[valid], ref[valid]) < MAX_DIFF
    _repeat_equal(run, d, valid)
    if use_psum_layout:
        # zero A padding rows give zero D padding rows (ensure_zero_padding=True)
        for end in layout.tolist():
            assert torch.equal(d[end:align(end, 256)], torch.zeros_like(d[end:align(end, 256)]))
    # The NN alias takes B (and its scales) as stored, [G, K, N] for an MN-major operand
    if major_b == MajorTypeAB.MNMajor:
        d_nn = torch.empty_like(d)
        deep_gemm.m_grouped_fp8_gemm_nn_contiguous(
            a, (b[0].mT, b[1].mT), d_nn, layout, recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b,
            use_psum_layout=use_psum_layout)
        assert torch.equal(d_nn[valid], d[valid])


def test_m_grouped_contiguous_ragged_k_uses_data_parallel_path():
    # K % 128 != 0 rules out the persistent kernel
    for psum in (False, True):
        m, a, b, layout, d, ref, valid = generate_m_grouped_contiguous(
            3, 600, 1024, 1000, MajorTypeAB.KMajor, MajorTypeAB.KMajor, use_ue8m0=True,
            use_psum_layout=psum)
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(a, b, d, layout, use_psum_layout=psum)
        assert calc_diff(d[valid], ref[valid]) < MAX_DIFF


@pytest.mark.parametrize("alignment", [128, 64])
@pytest.mark.parametrize("use_psum_layout", [False, True])
def test_m_grouped_contiguous_small_alignment(alignment, use_psum_layout):
    # alignments the 256-row tiles do not divide run the small-M kernel's shorter tiles
    deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
    m, a, b, layout, d, ref, valid = generate_m_grouped_contiguous(
        5, 300, 1024, 1536, MajorTypeAB.KMajor, MajorTypeAB.KMajor, use_ue8m0=True,
        use_psum_layout=use_psum_layout)
    run = lambda: deep_gemm.m_grouped_fp8_gemm_nt_contiguous(a, b, d, layout, use_psum_layout=use_psum_layout)
    run()
    assert calc_diff(d[valid], ref[valid]) < MAX_DIFF
    _repeat_equal(run, d, valid)


def test_m_grouped_contiguous_rejects_unsupported_alignment():
    m, a, b, layout, d, _, _ = generate_m_grouped_contiguous(
        2, 256, 256, 256, MajorTypeAB.KMajor, MajorTypeAB.KMajor, use_ue8m0=True)
    deep_gemm.set_mk_alignment_for_contiguous_layout(8)
    with pytest.raises(RuntimeError, match="must be a multiple of 16"):
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(a, b, d, layout)


MASKED_SHAPES = ([(32, 4096, 192), (6, 4096, 1024), (32, 4096, 20), (6, 4096, 20)] if FULL else
                 [(8, 1024, 192), (6, 1024, 700), (12, 512, 20)])
MASKED_NK = [(6144, 7168), (7168, 3072), (4096, 4096), (4096, 2048)] if FULL else [(4096, 2048), (576, 1000)]


@pytest.mark.parametrize("groups,max_m,expected_m", MASKED_SHAPES)
@pytest.mark.parametrize("n,k", MASKED_NK)
@pytest.mark.parametrize("use_psum_layout", [False, True])
def test_m_grouped_masked(groups, max_m, expected_m, n, k, use_psum_layout):
    a, b, layout, d, ref, valid = generate_m_grouped_masked(
        groups, max_m, expected_m, n, k, use_ue8m0=True, use_psum_layout=use_psum_layout)
    if use_psum_layout:
        run = lambda: deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            a, b, d, layout, use_psum_layout=True, expected_m_for_psum_layout=int(expected_m * 1.2))
    else:
        run = lambda: deep_gemm.m_grouped_fp8_gemm_nt_masked(a, b, d, layout, int(expected_m * 1.2))
    run()
    assert calc_diff(d[valid], ref[valid]) < MAX_DIFF
    _repeat_equal(run, d, valid)
    if use_psum_layout:
        # the padding rows of the PSUM groups (uninitialized A rows) are stored as zeros
        alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()
        for end in layout.tolist():
            gap = d[end:align(end, alignment)]
            if expected_m * 1.2 <= 64:
                assert torch.equal(gap, torch.zeros_like(gap))


@pytest.mark.parametrize("expected_m", [100, 20])
def test_m_grouped_masked_leaves_rows_past_mask_untouched(expected_m):
    a, b, masked_m, d, ref, valid = generate_m_grouped_masked(4, 512, expected_m, 512, 512, use_ue8m0=True)
    d.fill_(7)
    deep_gemm.m_grouped_fp8_gemm_nt_masked(a, b, d, masked_m, int(expected_m * 1.2))
    assert bool((d[~valid] == 7).all())
    assert calc_diff(d[valid], ref[valid]) < MAX_DIFF


K_GROUPED_SHAPES = ([(8, 768, 2048, 128), (4, 4096, 7168, 8192), (4, 7168, 2048, 8192), (8, 4096, 7168, 4096),
                     (16, 4096, 7168, 2048)] if FULL else
                    [(8, 768, 2048, 128), (4, 1024, 1536, 2048), (16, 512, 1024, 384)])
K_GROUPED_SF = [(32, 128), (128, 128), (32, 256), (128, 256), (32, 384), (128, 384)]


def _k_variants(real_ks):
    yield list(real_ks)
    empty = [0 if random.random() < 0.25 else k for k in real_ks]
    empty[0], empty[-1] = 0, real_ks[-1]
    yield empty
    yield [k if i == len(real_ks) // 2 else 0 for i, k in enumerate(real_ks)]


@pytest.mark.parametrize("groups,m,n,expected_k", K_GROUPED_SHAPES)
@pytest.mark.parametrize("gran_k,k_alignment", K_GROUPED_SF)
@pytest.mark.parametrize("use_psum_layout", [False, True])
@pytest.mark.parametrize("accumulate,out_dtype", [(True, torch.float), (False, torch.float), (False, torch.bfloat16)])
def test_k_grouped_tn(groups, m, n, expected_k, gran_k, k_alignment, use_psum_layout, accumulate, out_dtype):
    real_ks = [max(1, int(expected_k * random.uniform(0.7, 1.3))) for _ in range(groups)]
    for ks in _k_variants(real_ks):
        total_k, a, b, c, d, ref, layout, host_ks = generate_k_grouped_contiguous(
            groups, m, n, MajorTypeAB.MNMajor, MajorTypeAB.MNMajor, ks, use_ue8m0=True, gran_k=gran_k,
            use_psum_layout=use_psum_layout, k_alignment=k_alignment, accumulate=accumulate,
            out_dtype=out_dtype)
        deep_gemm.set_mk_alignment_for_contiguous_layout(k_alignment)
        initial = d.clone() if accumulate else torch.full_like(d, float("nan"))
        run = lambda out, acc, ks_arg=host_ks, sfs=(a, b): deep_gemm.k_grouped_fp8_gemm_tn_contiguous(
            sfs[0], sfs[1], out, ks_arg, layout, acc, recipe=(1, 1, gran_k), use_psum_layout=use_psum_layout)
        out = initial.clone()
        run(out, out if accumulate else None)
        if accumulate:
            assert calc_diff(out, ref) < MAX_DIFF
        else:
            # direct output equals accumulation into zeros, after one cast for BF16
            fp32 = torch.zeros_like(out, dtype=torch.float)
            run(fp32, fp32)
            assert torch.equal(out, fp32.to(out.dtype))
        host_ks_options = (host_ks, None, []) if use_psum_layout else (host_ks,)
        for ks_arg in host_ks_options:
            again = initial.clone()
            run(again, again if accumulate else None, ks_arg)
            assert torch.equal(again, out)
        if gran_k == 32:
            # pre-packed INT32 scales give the same result
            packed = [(x[0], deep_gemm.get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(
                x[1], layout, host_ks, gran_k, k_alignment, use_psum_layout)) for x in (a, b)]
            sf_ks = [k // gran_k for k in host_ks]
            for x, p in zip((a, b), packed):
                ref_packed = [pack_ue8m0_to_int(torch.nn.functional.pad(
                    sf.T, (0, align(sk, 4) - sk)).contiguous()).T for sf, sk in zip(x[1].split(sf_ks), sf_ks)
                    if sk > 0]
                assert torch.equal(p[1], torch.cat(ref_packed))
            again = initial.clone()
            run(again, again if accumulate else None, host_ks, packed)
            assert torch.equal(again, out)


@pytest.mark.parametrize("groups,m,n,expected_k", [(4, 512, 1024, 1024), (6, 256, 768, 384)])
def test_k_grouped_nt(groups, m, n, expected_k):
    deep_gemm.set_mk_alignment_for_contiguous_layout(128)
    ks = [max(1, int(expected_k * random.uniform(0.7, 1.3))) for _ in range(groups)]
    total_k, a, b, c, d, ref, layout, host_ks = generate_k_grouped_contiguous(
        groups, m, n, MajorTypeAB.KMajor, MajorTypeAB.KMajor, ks, use_ue8m0=True, gran_k=128,
        k_alignment=128, accumulate=True, out_dtype=torch.float)
    deep_gemm.k_grouped_fp8_gemm_nt_contiguous(a, b, d, host_ks, layout, d)
    assert calc_diff(d, ref) < MAX_DIFF
