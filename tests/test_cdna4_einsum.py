"""Correctness tests for the CDNA4 einsum and fp8_einsum expressions, with the inputs and
tolerances of tests/test_einsum.py.
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

from utils import assert_direct_output_matches_fp32_accumulation

import deep_gemm
from deep_gemm.testing import calc_diff
from deep_gemm.utils.math import (
    ceil_div,
    per_block_cast_to_fp8,
    per_channel_cast_to_fp8,
    per_token_cast_to_fp8,
)

FULL = os.environ.get("DG_TEST_FULL") == "1"
BATCHES = (4, 32, 128, 4096, 8192) if FULL else (4, 128, 1024)
HRD = [(128, 512, 128), (8, 4096, 1024)]


@pytest.fixture(autouse=True)
def _seed():
    random.seed(0)
    torch.manual_seed(0)


@pytest.mark.parametrize("s", (129, 4096, 8192) if FULL else (129, 1024))
@pytest.mark.parametrize("m,n,k", [(128, 384, 128), (256, 256, 256), (384, 128, 384)])
@pytest.mark.parametrize(
    "dtype", [torch.float, torch.bfloat16], ids=["fp32_acc", "bf16"]
)
def test_bmk_bnk_mn(s, m, n, k, dtype):
    a = torch.randn((s, m, k), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((s, n, k), dtype=torch.bfloat16, device="cuda")
    d = torch.randn((m, n), dtype=dtype, device="cuda")
    c = d if dtype == torch.float else None
    ref = (c if c is not None else 0) + torch.bmm(a.float(), b.float().mT).sum(0)
    deep_gemm.einsum("bmk,bnk->mn", a, b, d, c=c)
    assert calc_diff(d, ref) < 1e-5


def _padded_hdr(h, d, r):
    return torch.randn((h, d, r + 128), device="cuda", dtype=torch.bfloat16)[:, :, :r]


def _check_bf16_einsum(expr, x, y, out_shape, label):
    """The FP32 result (accumulated from zero) matches an FP32 reference, and the direct BF16
    output is that result rounded once. (A BF16 reference from another library rounds differently
    accumulated sums, which the CUDA test's 1e-10 bound only tolerates for matching orders.)"""
    launch = lambda out, acc: deep_gemm.einsum(expr, x, y, out, c=acc)
    fp32 = torch.zeros(out_shape, device="cuda", dtype=torch.float)
    launch(fp32, fp32)
    assert calc_diff(fp32, torch.einsum(expr, x.float(), y.float())) < 1e-10
    z = torch.empty(out_shape, device="cuda", dtype=torch.bfloat16)
    deep_gemm.einsum(expr, x, y, z)
    assert_direct_output_matches_fp32_accumulation(z, launch, label)


@pytest.mark.parametrize("h,r,d", HRD)
@pytest.mark.parametrize("b", BATCHES)
def test_bhr_hdr_bhd(h, r, d, b):
    x = torch.randn((b, h, r), device="cuda", dtype=torch.bfloat16)
    _check_bf16_einsum(
        "bhr,hdr->bhd", x, _padded_hdr(h, d, r), (b, h, d), f"{b=}, {h=}, {r=}, {d=}"
    )


@pytest.mark.parametrize("h,r,d", HRD)
@pytest.mark.parametrize("b", BATCHES)
def test_bhd_hdr_bhr(h, r, d, b):
    x = torch.randn((b, h, d), device="cuda", dtype=torch.bfloat16)
    _check_bf16_einsum(
        "bhd,hdr->bhr", x, _padded_hdr(h, d, r), (b, h, r), f"{b=}, {h=}, {r=}, {d=}"
    )


@pytest.mark.parametrize("h,r,d", HRD)
@pytest.mark.parametrize("b", BATCHES)
def test_bhd_bhr_hdr(h, r, d, b):
    x = torch.randn((b, h, d), device="cuda", dtype=torch.bfloat16)
    y = torch.randn((b, h, r), device="cuda", dtype=torch.bfloat16)
    z0 = torch.randn((h, d, r), device="cuda", dtype=torch.float32) * 10
    ref = z0 + torch.einsum("bhd,bhr->hdr", x.float(), y.float())
    z = z0.clone()
    deep_gemm.einsum("bhd,bhr->hdr", x, y, z, z)
    assert calc_diff(z, ref) < 1e-10


def _fp8_tokens(x, gran_dim):
    rows = x.view(-1, x.size(-1))
    data, sf = per_token_cast_to_fp8(rows, use_ue8m0=True)
    return data.view(x.shape), sf.view(*x.shape[:-1], ceil_div(x.size(-1), 128))


def _fp8_blocks(y):
    data = torch.empty_like(y, dtype=torch.float8_e4m3fn)
    sf = torch.empty(
        (y.size(0), ceil_div(y.size(1), 128), ceil_div(y.size(2), 128)),
        device="cuda",
        dtype=torch.float,
    )
    for i in range(y.size(0)):
        data[i], sf[i] = per_block_cast_to_fp8(y[i], use_ue8m0=True)
    return data, sf


@pytest.mark.parametrize("b", BATCHES)
def test_fp8_bhr_hdr_bhd(b):
    h, r, d = 8, 4096, 1024
    x = torch.randn((b, h, r), device="cuda", dtype=torch.bfloat16)
    y = torch.randn((h, d, r), device="cuda", dtype=torch.bfloat16)
    z = torch.empty((b, h, d), device="cuda", dtype=torch.bfloat16)
    deep_gemm.fp8_einsum("bhr,hdr->bhd", _fp8_tokens(x, r), _fp8_blocks(y), z)
    assert calc_diff(z, torch.einsum("bhr,hdr->bhd", x, y)) < 1e-3


@pytest.mark.parametrize("b", BATCHES)
def test_fp8_bhd_hdr_bhr(b):
    h, r, d = 8, 4096, 1024
    x = torch.randn((b, h, d), device="cuda", dtype=torch.bfloat16)
    y = torch.randn((h, d, r), device="cuda", dtype=torch.bfloat16)
    z = torch.empty((b, h, r), device="cuda", dtype=torch.bfloat16)
    deep_gemm.fp8_einsum("bhd,hdr->bhr", _fp8_tokens(x, d), _fp8_blocks(y), z)
    assert calc_diff(z, torch.einsum("bhd,hdr->bhr", x, y)) < 1e-3


@pytest.mark.parametrize("b", (4096, 8192) if FULL else (1024, 4096))
def test_fp8_bhd_bhr_hdr(b):
    h, r, d = 8, 4096, 1024
    x = torch.randn((b, h, d), device="cuda", dtype=torch.bfloat16)
    y = torch.randn((b, h, r), device="cuda", dtype=torch.bfloat16)
    z0 = torch.randn((h, d, r), device="cuda", dtype=torch.float32) * 10
    ref = z0 + torch.einsum("bhd,bhr->hdr", x, y)
    x_fp8 = per_channel_cast_to_fp8(x.view(b, -1), use_ue8m0=True)
    y_fp8 = per_channel_cast_to_fp8(y.view(b, -1), use_ue8m0=True)
    x_fp8 = (x_fp8[0].view(b, h, d), x_fp8[1].view(ceil_div(b, 128), h, d))
    y_fp8 = (y_fp8[0].view(b, h, r), y_fp8[1].view(ceil_div(b, 128), h, r))
    z = z0.clone()
    deep_gemm.fp8_einsum("bhd,bhr->hdr", x_fp8, y_fp8, z, c=z, recipe=(1, 1, 128))
    assert calc_diff(z, ref) < 1e-3


@pytest.mark.parametrize("batch,width,positions", [(8, 3, 3), (8, 5, 6), (13, 3, 3)])
def test_short_packed_wv_output_and_scale_guards(batch, width, positions, monkeypatch):
    monkeypatch.setenv("DG_CDNA4_DECODE_BATCH", str(batch))
    monkeypatch.setenv("DG_BENCH_SPEC_WIDTH", str(width))
    monkeypatch.setenv("DG_CDNA4_PACKED_WV_TILE", "64x128")
    rows, heads, k, n = batch * positions, 8, 4096, 1024

    def quantized(shape):
        data = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        fp8, scales = per_token_cast_to_fp8(
            data.reshape(-1, k), use_ue8m0=True, gran_k=32
        )
        return fp8.view(shape), scales.view(*shape[:-1], k // 32)

    x = quantized((rows, heads, k))
    weight = quantized((heads, n, k))
    bf16 = torch.empty((rows, heads, n), device="cuda", dtype=torch.bfloat16)
    deep_gemm.fp8_einsum("bhr,hdr->bhd", x, weight, bf16, recipe=(1, 1, 32))
    ref, sf = per_token_cast_to_fp8(
        bf16.flatten(1), use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
    )
    storage = torch.full(
        (rows * heads * n + 32,), 179, device="cuda", dtype=torch.uint8
    )
    packed = storage[16:-16].view(torch.float8_e4m3fn).reshape(rows, heads, n)
    scale_storage = torch.full(
        (heads * n // 128 + 2, (rows + 3) // 4 * 4 + 4),
        -1,
        device="cuda",
        dtype=torch.int32,
    )
    actual = scale_storage[1:-1, :rows].T
    deep_gemm.fp8_einsum("bhr,hdr->bhd", x, weight, (packed, actual), recipe=(1, 1, 32))
    assert torch.equal(ref.view(torch.uint8), packed.flatten(1).view(torch.uint8))
    assert torch.equal(sf, actual)
    assert (scale_storage[[0, -1]] == -1).all() and (
        scale_storage[:, rows:] == -1
    ).all()
    assert (storage[:16] == 179).all() and (storage[-16:] == 179).all()
