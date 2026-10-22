"""ROCm numerical and storage tests for native FP8/FP4 NT GEMMs."""

import pytest
import torch

import deep_gemm

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.version.hip, reason="requires ROCm"
)


def _operand(rows, k, bits, gran_mn, gran_k, packed_sf):
    scales = torch.exp2(
        torch.randint(
            -4,
            2,
            ((rows + gran_mn - 1) // gran_mn, (k + gran_k - 1) // gran_k),
            device="cuda",
        ).float()
    )
    if bits == 8:
        data = torch.randn(rows, k, device="cuda").to(torch.float8_e4m3fn)
        decoded = data.float()
    else:
        codes = torch.randint(0, 16, (rows, k), dtype=torch.uint8, device="cuda")
        data = (codes[:, ::2] | (codes[:, 1::2] << 4)).view(torch.int8)
        table = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device="cuda")
        decoded = table[(codes & 7).long()] * torch.where((codes & 8) != 0, -1.0, 1.0)
        assert data.numel() * data.element_size() == rows * k // 2
    expanded = scales.repeat_interleave(gran_mn, 0)[:rows].repeat_interleave(gran_k, 1)[
        :, :k
    ]
    ref = decoded * expanded
    if packed_sf:
        scales = deep_gemm.transform_sf_into_required_layout(
            scales, rows, k, (gran_mn, gran_k)
        )
        # Canonical native ABI: packed SF is already expanded along MN and
        # needs a column pitch aligned to four words. Keep nontrivial padding.
        pitch = (rows + 3) // 4 * 4 + 4
        padded = torch.empty_strided(scales.shape, (1, pitch), dtype=scales.dtype, device=scales.device)
        scales = padded.copy_(scales)
    return (data, scales), ref


@pytest.mark.parametrize("bits_a,bits_b", [(8, 8), (8, 4), (4, 8), (4, 4)])
@pytest.mark.parametrize("m,n,k", [(1, 1, 32), (17, 65, 160), (32, 64, 512)])
@pytest.mark.parametrize("packed_sf", [False, True])
def test_scaled_gemm(bits_a, bits_b, m, n, k, packed_sf):
    torch.manual_seed(15)
    a, ar = _operand(m, k, bits_a, 1, 32, packed_sf)
    b, br = _operand(n, k, bits_b, 32, 32, packed_sf)
    storage = torch.full((m, n + 16), float("nan"), device="cuda")
    out = storage[:, :n]
    c = torch.randn(m, n, device="cuda")
    deep_gemm.fp8_fp4_gemm_nt(a, b, out, c, recipe=(1, 1 if packed_sf else 32, 32), alpha=0.75)
    ref = (0.75 * (ar.double() @ br.double().T) + c.double()).float()
    # Native MX accumulation is not bitwise equal to FP64, especially after
    # cancellation. Bound the relative L2 error against dequantized operands,
    # substantially more tightly than upstream's quantized-vs-original test.
    error = (out.double() - ref.double()).square().sum().sqrt()
    magnitude = ref.double().square().sum().sqrt().clamp_min(1e-12)
    assert error / magnitude < 5e-5
    assert (out - ref).abs().max() < ref.abs().max().clamp_min(1) * 1e-4
    assert torch.isnan(storage[:, n:]).all()


def test_distinct_k_granularities_and_graph():
    torch.manual_seed(16)
    a, ar = _operand(33, 640, 8, 1, 128, True)
    b, br = _operand(72, 640, 4, 1, 32, True)
    out = torch.empty(33, 72, dtype=torch.bfloat16, device="cuda")

    def run():
        deep_gemm.fp8_fp4_gemm_nt(a, b, out, recipe_a=(1, 128), recipe_b=(1, 32))

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.testing.assert_close(out, (ar @ br.T).bfloat16(), atol=0.03, rtol=0.008)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_bf16_strides_and_inplace_c(dtype):
    torch.manual_seed(17)
    # Row padding and MN-major operands exercise native supported strides.
    # Arbitrary step-two K strides are outside the upstream GEMM ABI.
    a = torch.randn(19, 100, device="cuda", dtype=torch.bfloat16)[:, :97]
    b = torch.randn(97, 72, device="cuda", dtype=torch.bfloat16)[:, :71].T
    storage = torch.randn(19, 72, device="cuda", dtype=dtype)
    out = storage[:, :71]
    guard = storage[:, 71:].clone()
    original = out.clone()
    deep_gemm.bf16_gemm_nt(a, b, out, out)
    ref = a.float() @ b.float().T + original.float()
    torch.testing.assert_close(
        out,
        ref.to(dtype),
        atol=0.03 if dtype == torch.bfloat16 else 1e-4,
        rtol=0.008 if dtype == torch.bfloat16 else 1e-4,
    )
    assert torch.equal(storage[:, 71:], guard)
    unsupported = torch.randn(19, 194, device="cuda", dtype=torch.bfloat16)[:, ::2]
    with pytest.raises((RuntimeError, ValueError), match="K-major|MN-major"):
        deep_gemm.bf16_gemm_nt(unsupported, b, out)


def test_grouped_scale_broadcast():
    scales = torch.exp2(torch.randint(-7, 7, (3, 2, 5), device="cuda").float())
    packed = deep_gemm.transform_sf_into_required_layout(
        scales, 37, 160, (32, 32), num_groups=3
    )
    bytes_ = packed.contiguous().view(torch.uint8)
    expected = (scales.repeat_interleave(32, 1)[:, :37].log2() + 127).to(torch.uint8)
    torch.testing.assert_close(bytes_[..., :5], expected, rtol=0, atol=0)
    # The canonical native layout fills unused scale bytes with zero. They
    # represent no valid K elements; do not impose the retired baseline's 127.
    assert (bytes_[..., 5:] == 0).all()


def test_native_float4_storage_dtype():
    if not hasattr(torch, "float4_e2m1fn_x2"):
        pytest.skip("PyTorch does not expose the native FP4 storage dtype")
    a, _ = _operand(9, 128, 8, 1, 32, True)
    b, _ = _operand(67, 128, 4, 1, 32, True)
    ref = torch.empty(9, 67, device="cuda")
    out = torch.empty_like(ref)
    deep_gemm.fp8_fp4_gemm_nt(a, b, ref, recipe=(1, 1, 32))
    deep_gemm.fp8_fp4_gemm_nt(
        a, (b[0].view(torch.float4_e2m1fn_x2), b[1]), out, recipe=(1, 1, 32)
    )
    assert torch.equal(out, ref)


def test_legacy_default_recipe():
    a, ar = _operand(19, 256, 8, 1, 128, False)
    b, br = _operand(63, 256, 8, 128, 128, False)
    out = torch.empty(19, 63, device="cuda")
    deep_gemm.fp8_fp4_gemm_nt(a, b, out)
    ref = (ar.double() @ br.double().T).float()
    error = (out.double() - ref.double()).square().sum().sqrt()
    assert error / ref.double().square().sum().sqrt() < 5e-5
