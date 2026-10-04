"""Correctness tests for the CDNA4 HIP FP8 GEMM backend."""
import pytest

try:
    import torch
except Exception as exc:  # pragma: no cover - collection-only environments
    pytest.skip(f"PyTorch is required: {exc}", allow_module_level=True)

if torch.version.hip is None or not torch.cuda.is_available():
    pytest.skip("CDNA4 tests require a ROCm device", allow_module_level=True)

import deep_gemm


def _inputs(m, n, k):
    total_a = max(1, m * k)
    total_b = max(1, n * k)
    xa = (torch.arange(total_a, device="cuda", dtype=torch.float32).reshape(m, k) % 257 - 128) / 64
    xb = (torch.arange(total_b, device="cuda", dtype=torch.float32).reshape(n, k) % 193 - 96) / 48
    return deep_gemm.per_token_cast_to_fp8(xa.bfloat16(), use_ue8m0=True, gran_k=32), \
        deep_gemm.per_token_cast_to_fp8(xb.bfloat16(), use_ue8m0=True, gran_k=32)


def _dequant(q, scale):
    return q.float() * scale.repeat_interleave(32, dim=1)[:, :q.size(1)]


@pytest.mark.parametrize("m,n,k", [(1, 33, 64), (17, 65, 96), (128, 128, 128)])
def test_fp8_gemm_nt_matches_pytorch(m, n, k):
    a, b = _inputs(m, n, k)
    ref = _dequant(a[0], a[1]) @ _dequant(b[0], b[1]).transpose(0, 1)
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    deep_gemm.fp8_gemm_nt(a, b, out, recipe_a=(1, 32), recipe_b=(1, 32))
    torch.testing.assert_close(out.float(), ref, atol=1e-2, rtol=1e-2)
    again = torch.empty_like(out)
    deep_gemm.fp8_gemm_nt(a, b, again, recipe_a=(1, 32), recipe_b=(1, 32))
    assert torch.equal(out, again)


def test_fp8_gemm_nt_accumulate_alpha_and_packed_scales():
    m, n, k = 17, 65, 96
    a, b = _inputs(m, n, k)
    packed_a = deep_gemm.transform_sf_into_required_layout(a[1], m, k, (1, 32))
    packed_b = deep_gemm.transform_sf_into_required_layout(b[1], n, k, (1, 32))
    c = torch.full((m, n), 0.25, device="cuda", dtype=torch.float32)
    c0 = c.clone()
    out = c0.clone()
    alpha = 0.75
    deep_gemm.fp8_gemm_nt((a[0], packed_a), (b[0], packed_b), out, c=out,
                          recipe_a=(1, 32), recipe_b=(1, 32), alpha=alpha)
    ref = alpha * (_dequant(a[0], a[1]) @ _dequant(b[0], b[1]).transpose(0, 1)) + c0
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


def test_fp8_gemm_fp32_output():
    a, b = _inputs(17, 65, 96)
    out = torch.empty((17, 65), device="cuda", dtype=torch.float32)
    deep_gemm.fp8_gemm_nt(a, b, out, recipe_a=(1, 32), recipe_b=(1, 32))
    ref = _dequant(a[0], a[1]) @ _dequant(b[0], b[1]).transpose(0, 1)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


def test_fp32_multi_tile_launch():
    # Exact products expose extra blocks overwriting the first output tile.
    # These dimensions exercise several rows and columns of full output tiles.
    m, n, k = 4096, 4096, 256
    a = torch.ones((m, k), device="cuda", dtype=torch.float8_e4m3fn)
    b = torch.ones((n, k), device="cuda", dtype=torch.float8_e4m3fn)
    sa = torch.ones((m, k // 32), device="cuda", dtype=torch.float32)
    sb = torch.ones((n, k // 32), device="cuda", dtype=torch.float32)
    out = torch.empty((m, n), device="cuda", dtype=torch.float32)
    for _ in range(2):
        out.fill_(float('nan'))
        deep_gemm.fp8_gemm_nt((a, sa), (b, sb), out, recipe=(1, 1, 32))
        assert bool((out == k).all()), 'incorrect output from a multi-tile FP32 launch'


def test_fp8_gemm_aliases_and_empty_shapes():
    m, n, k = 17, 65, 96
    a, b = _inputs(m, n, k)
    out = torch.empty((17, 65), device="cuda", dtype=torch.bfloat16)
    ref = _dequant(a[0], a[1]) @ _dequant(b[0], b[1]).transpose(0, 1)
    b_nn = (b[0].transpose(0, 1), b[1].transpose(0, 1))
    a_t = (a[0].transpose(0, 1), a[1].transpose(0, 1))
    b_t = (b[0].transpose(0, 1), b[1].transpose(0, 1))
    for fn, lhs, rhs in ((deep_gemm.fp8_gemm_nn, a, b_nn),
                         (deep_gemm.fp8_gemm_tn, a_t, b_t),
                         (deep_gemm.fp8_gemm_tt, a_t, b)):
        fn(lhs, rhs, out, recipe_a=(1, 32), recipe_b=(1, 32))
        torch.testing.assert_close(out.float(), ref, atol=1e-2, rtol=1e-2)

    empty = torch.empty((0, 65), device="cuda", dtype=torch.bfloat16)
    deep_gemm.fp8_gemm_nt((a[0][:0], a[1][:0]), b, empty,
                          recipe_a=(1, 32), recipe_b=(1, 32))
    assert empty.numel() == 0
    empty_n = torch.empty((m, 0), device="cuda", dtype=torch.bfloat16)
    deep_gemm.fp8_gemm_nt(a, (b[0][:0], b[1][:0]), empty_n,
                          recipe_a=(1, 32), recipe_b=(1, 32))
    assert empty_n.numel() == 0

    zero_k_a = torch.empty((m, 0), device="cuda", dtype=torch.float8_e4m3fn)
    zero_k_b = torch.empty((n, 0), device="cuda", dtype=torch.float8_e4m3fn)
    zero_k_d = torch.full((m, n), 0.5, device="cuda", dtype=torch.bfloat16)
    zero_k_c = torch.full_like(zero_k_d, 0.25)
    deep_gemm.fp8_gemm_nt((zero_k_a, torch.empty((m, 0), device="cuda", dtype=torch.float32)),
                          (zero_k_b, torch.empty((n, 0), device="cuda", dtype=torch.float32)),
                          zero_k_d, c=zero_k_c, recipe_a=(1, 32), recipe_b=(1, 32))
    assert torch.equal(zero_k_d, zero_k_c)
