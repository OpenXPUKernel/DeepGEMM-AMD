"""Exact AMD fast-path comparisons, including graph replay and input mutation."""

import pytest

try:
    import torch
except ImportError:
    pytest.skip("PyTorch is required", allow_module_level=True)

if not torch.version.hip or not torch.cuda.is_available():
    pytest.skip("requires an AMD GPU", allow_module_level=True)

from test_cdna4_mega_mhc import make_inputs, make_outputs, run_mega_mhc

import deep_gemm


def _equal(left, right):
    return torch.equal(
        left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8)
    )


@pytest.mark.parametrize("rows,experts,topk", [(400, 128, 3), (500, 384, 6)])
@pytest.mark.parametrize("controls", [False, True])
@torch.inference_mode()
def test_gate_partial_reduction_is_exact(rows, experts, topk, controls, monkeypatch):
    torch.manual_seed(1917)
    monkeypatch.delenv("DG_CDNA4_GEMM_TUNE", raising=False)
    monkeypatch.delenv("DG_CDNA4_GEMM_TUNE_MAP", raising=False)
    x = torch.randn(rows, 5120, device="cuda", dtype=torch.bfloat16) * 0.1
    weight = torch.randn(experts, 5120, device="cuda", dtype=torch.bfloat16) * 0.01
    bias = torch.randn(experts, device="cuda") * 0.01
    count = torch.ones(experts, dtype=torch.int32, device="cuda")
    count[:8] = 2
    physical = torch.full((experts, 2), -1, dtype=torch.int32, device="cuda")
    physical[:, 0] = torch.arange(experts, dtype=torch.int32, device="cuda")
    physical[:8, 1] = torch.arange(
        experts, experts + 8, dtype=torch.int32, device="cuda"
    )
    kwargs = dict(
        num_topk=topk,
        use_shared_as_routed=False,
        num_shared_experts=0,
        routed_scaling_factor=1.5,
        ep_rank=3,
        bias=bias,
        to_physical_map=physical,
        logical_count=count,
    )
    if controls:
        token = torch.arange(rows, device="cuda")
        kwargs.update(
            mask=token % 7 != 0,
            image_token_mask=token % 5 == 0,
            image_bias=-bias,
            force_random=token % 13 == 0,
        )
    outputs = [
        (
            torch.empty(rows, topk, dtype=torch.int64, device="cuda"),
            torch.empty(rows, topk, dtype=torch.float32, device="cuda"),
        )
        for _ in range(2)
    ]
    monkeypatch.setenv("DG_GATE_FUSED_REDUCTION", "0")
    deep_gemm.bf16_mega_gate(x, weight, out=outputs[0], **kwargs)
    monkeypatch.setenv("DG_GATE_FUSED_REDUCTION", "1")
    for plan in ("off", "2,4,1", "8,4,2"):
        monkeypatch.setenv("DG_GATE_PARTIAL_TILE", plan)
        partial = deep_gemm._C.cdna4_bf16_gate_partials(x, weight)
        assert partial is not None
        actual = partial[0].clone()
        for item in partial[1:]:
            actual.add_(item)
        expected = torch.empty_like(actual)
        deep_gemm.bf16_gemm_nt(x, weight, expected)
        assert _equal(actual, expected)
        deep_gemm.bf16_mega_gate(x, weight, out=outputs[1], **kwargs)
        assert all(_equal(a, b) for a, b in zip(*outputs))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        deep_gemm.bf16_mega_gate(x, weight, out=outputs[1], **kwargs)
    x.add_(0.03125)
    monkeypatch.setenv("DG_GATE_FUSED_REDUCTION", "0")
    deep_gemm.bf16_mega_gate(x, weight, out=outputs[0], **kwargs)
    graph.replay()
    assert all(_equal(a, b) for a, b in zip(*outputs))


@pytest.mark.parametrize("rows", [33, 400, 500])
@pytest.mark.parametrize("shifted", [False, True])
@pytest.mark.parametrize("layout", ["col", "moe"])
@torch.inference_mode()
def test_single_launch_mhc_finish_preserves_every_output(
    rows, shifted, layout, monkeypatch
):
    inputs, _ = make_inputs(rows, 5120, 1917, shifted)
    expected, actual = [make_outputs(inputs, layout) for _ in range(2)]
    monkeypatch.setenv("DG_MHC_MIX_THREADS", "128")
    monkeypatch.setenv("DG_MHC_FUSED_FINISH", "0")
    run_mega_mhc(inputs, expected)
    monkeypatch.setenv("DG_MHC_FUSED_FINISH", "1")
    run_mega_mhc(inputs, actual)

    def compare():
        for key, value in expected.items():
            if isinstance(value, torch.Tensor) and key != "y_shared_sf_storage":
                assert _equal(value, actual[key]), key

    compare()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_mega_mhc(inputs, actual)
    inputs["x"].add_(0.03125)
    monkeypatch.setenv("DG_MHC_FUSED_FINISH", "0")
    run_mega_mhc(inputs, expected)
    graph.replay()
    compare()


@pytest.mark.parametrize("rows,cols", [(17, 96), (33, 256)])
@torch.inference_mode()
def test_shape_specialization_preserves_accumulation_and_strides(
    rows, cols, monkeypatch
):
    torch.manual_seed(1917)
    x = torch.randn(rows, 1024, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(cols, 1024, device="cuda", dtype=torch.bfloat16)
    addend = torch.randn(rows, cols, device="cuda", dtype=torch.float32)
    expected = torch.empty(rows, cols + 16, device="cuda")[:, :cols]
    actual = torch.empty_like(expected)
    monkeypatch.setenv("DG_CDNA4_SPECIALIZE_GEMM", "0")
    deep_gemm.bf16_gemm_nt(x, weight, expected, addend)
    monkeypatch.setenv("DG_CDNA4_SPECIALIZE_GEMM", "1")
    deep_gemm.bf16_gemm_nt(x, weight, actual, addend)
    assert _equal(expected, actual)
    x = x.clone() * 0.5
    monkeypatch.setenv("DG_CDNA4_SPECIALIZE_GEMM", "0")
    deep_gemm.bf16_gemm_nt(x, weight, expected)
    monkeypatch.setenv("DG_CDNA4_SPECIALIZE_GEMM", "1")
    deep_gemm.bf16_gemm_nt(x, weight, actual)
    assert _equal(expected, actual)
