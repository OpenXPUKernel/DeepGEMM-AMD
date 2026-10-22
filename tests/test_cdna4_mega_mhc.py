"""Correctness tests for the CDNA4 mega_mhc with the reference and the checks of tests/test_mega_mhc.py:
shifted and normal mHC, BF16 and FP8 outputs in the column-major and Mega MoE scale layouts,
bitwise repeatability, bitwise equal outputs of equal tokens, deterministic mode, CUDA graphs and
concurrent streams.
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
from deep_gemm.utils import align, per_token_cast_to_fp8


def make_inputs(num_tokens, hidden, seed, is_shifted):
    hc_mult = 4
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def randn(shape, dtype):
        return torch.randn(shape, dtype=dtype, device="cuda", generator=generator)

    inputs = dict(
        x=randn((num_tokens, hidden), torch.bfloat16),
        residual=randn((num_tokens, hc_mult, hidden), torch.bfloat16),
        post_mix=randn((num_tokens, hc_mult, 1), torch.float).sigmoid(),
        comb_res_mix=torch.rand((num_tokens, hc_mult, hc_mult), device="cuda", generator=generator),
        shifted_prev_mix=randn((num_tokens, hc_mult, 1), torch.float).sigmoid() if is_shifted else None,
        fn=randn((hc_mult * (hc_mult + 2), hc_mult * hidden), torch.float).mul_(1e-2),
        mix_scales=randn((3, ), torch.float).mul_(0.1),
        mix_bases=randn((hc_mult * (hc_mult + 2), ), torch.float).mul_(0.1),
        rmsnorm_weight=randn((hidden, ), torch.float).mul_(0.1).add_(1).bfloat16(),
        hc_mult=hc_mult, hc_norm_eps=2e-5, hc_pre_eps=3e-4, hc_post_scale=1.25, sinkhorn_eps=2e-6,
        num_sinkhorn_iters=10, rmsnorm_eps=7e-6, rmsnorm_scale=1.25,
    )
    for _ in range(3):
        inputs["comb_res_mix"] /= inputs["comb_res_mix"].sum(-1, keepdim=True)
        inputs["comb_res_mix"] /= inputs["comb_res_mix"].sum(-2, keepdim=True)
    duplicates = tuple(i for i in (63, 64, 511, 512, num_tokens - 1) if 0 < i < num_tokens)
    for name in ("x", "residual", "post_mix", "comb_res_mix", "shifted_prev_mix"):
        if inputs[name] is not None:
            inputs[name][list(set(duplicates))] = inputs[name][0].clone()
    return inputs, duplicates


def make_outputs(inputs, sf_layout, shared_sf_block_m=224):
    num_tokens, hidden = inputs["x"].shape
    is_shifted = inputs["shifted_prev_mix"] is not None
    outputs = dict(
        new_residual=torch.empty_like(inputs["residual"]),
        new_prev_mix=torch.empty_like(inputs["shifted_prev_mix"]) if is_shifted else None,
        new_post_mix=torch.empty_like(inputs["post_mix"]),
        new_comb_res_mix=torch.empty_like(inputs["comb_res_mix"]),
        y_bf16=torch.empty_like(inputs["x"]),
    )
    if sf_layout == "bf16":
        return outputs
    sf_shape = (num_tokens, hidden // 128)
    outputs["y_fp8"] = torch.empty((num_tokens, hidden), dtype=torch.float8_e4m3fn, device="cuda")
    if sf_layout == "col":
        outputs["y_gemm_sf"] = torch.empty_strided(sf_shape, (1, align(num_tokens, 4)), dtype=torch.int32, device="cuda")
    else:
        outputs["y_routed_sf"] = torch.empty(sf_shape, dtype=torch.int32, device="cuda")
        num_rows = align(num_tokens, 4)
        storage = torch.empty_strided((num_rows, hidden // 128), (1, num_rows), dtype=torch.int32, device="cuda")
        outputs.update(y_shared_sf=storage[:num_tokens], y_shared_sf_storage=storage,
                       shared_sf_block_m=shared_sf_block_m)
    return outputs


def run_mega_mhc(inputs, outputs, **overrides):
    kwargs = {**inputs, **outputs}
    kwargs.pop("y_shared_sf_storage", None)
    kwargs.update(overrides)
    deep_gemm.mega_mhc(**kwargs)


def extra_sf_rows(num_tokens, block_m):
    # Native MegaMoE's finish/scatter and grouped GEMMs consume ordinary rows.
    return torch.arange(num_tokens, device="cuda")


def sinkhorn_reference(value, repeat, eps):
    value = value.softmax(dim=-1) + eps
    value = value / (value.sum(dim=-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        value = value / (value.sum(dim=-1, keepdim=True) + eps)
        value = value / (value.sum(dim=-2, keepdim=True) + eps)
    return value


@torch.no_grad()
def mhc_reference(x, residual, post_mix, comb_res_mix, fn, mix_scales, mix_bases, rmsnorm_weight, hc_mult, hc_norm_eps,
                  hc_pre_eps, hc_post_scale, sinkhorn_eps, num_sinkhorn_iters, rmsnorm_eps, rmsnorm_scale,
                  sf_layout="bf16", shifted_prev_mix=None, mix_dtype=torch.double):
    new_residual = x.float().unsqueeze(1) * post_mix
    new_residual += torch.einsum("tij,tih->tjh", comb_res_mix, residual.float())
    new_residual = new_residual.bfloat16()
    norm_input = None if shifted_prev_mix is None else (new_residual.float() * shifted_prev_mix).sum(dim=1).bfloat16()
    mixes = (new_residual.flatten(1).to(mix_dtype) @ fn.to(mix_dtype).mT).float()
    hc_sqr_sum = new_residual.float().square().sum((1, 2))
    mixes *= torch.rsqrt(hc_sqr_sum / (hc_mult * x.size(1)) + hc_norm_eps).unsqueeze(1)
    scales = torch.cat((mix_scales[0].expand(hc_mult), mix_scales[1].expand(hc_mult),
                        mix_scales[2].expand(hc_mult * hc_mult)))
    mixes = mixes * scales + mix_bases
    new_prev_mix = mixes[:, :hc_mult].sigmoid().unsqueeze(2) + hc_pre_eps
    new_post_mix = (mixes[:, hc_mult:2 * hc_mult].sigmoid() * hc_post_scale).unsqueeze(2)
    new_comb_res_mix = sinkhorn_reference(mixes[:, 2 * hc_mult:].view(-1, hc_mult, hc_mult), num_sinkhorn_iters,
                                          sinkhorn_eps)
    if norm_input is None:
        norm_input = (new_residual.float() * new_prev_mix).sum(dim=1).bfloat16()
    norm_input_float = norm_input.float()
    y_bf16 = (norm_input_float * torch.rsqrt(norm_input_float.square().mean(1) + rmsnorm_eps).unsqueeze(1) *
              (rmsnorm_weight.float() * rmsnorm_scale)).bfloat16()
    result = dict(new_residual=new_residual, new_post_mix=new_post_mix, new_comb_res_mix=new_comb_res_mix,
                  y_bf16=y_bf16)
    if shifted_prev_mix is not None:
        result["new_prev_mix"] = new_prev_mix
    if sf_layout != "bf16":
        y_fp8, y_fp8_sf = per_token_cast_to_fp8(y_bf16, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
        result.update(y_fp8=y_fp8, y_sf=y_fp8_sf)
    return result


def logical_outputs(outputs):
    names = ("new_residual", "new_post_mix", "new_comb_res_mix", "y_bf16", "new_prev_mix", "y_fp8")
    values = {name: outputs[name] for name in names if outputs.get(name) is not None}
    for name in ("y_gemm_sf", "y_routed_sf"):
        if name in outputs:
            values["y_sf"] = outputs[name]
    if "y_shared_sf_storage" in outputs:
        rows = extra_sf_rows(outputs["y_bf16"].shape[0], outputs["shared_sf_block_m"])
        values["y_shared_sf"] = outputs["y_shared_sf_storage"][rows]
    return values


def comparable(tensor):
    return tensor.contiguous().view(torch.uint8) if tensor.dtype == torch.int32 else tensor


def check_correctness(actual, reference, case):
    # tighter than the CUDA test's 5e-5 / 2e-4: the CDNA4 outputs differ from the reference by
    # BF16 roundings and the 16-bit mantissas of fn
    for name, tensor in actual.items():
        expected = reference["y_sf"] if name == "y_shared_sf" else reference[name]
        diff = float(calc_diff(comparable(tensor), comparable(expected)))
        assert diff < (1e-6 if name in ("y_fp8", "y_sf", "y_shared_sf") else 1e-8), f"{case}, {name=}, {diff=}"


@pytest.mark.parametrize("is_shifted", [True, False])
@pytest.mark.parametrize("sf_layout", ["col", "extra", "bf16"])
@pytest.mark.parametrize("hidden", [4096, 7168])
@pytest.mark.parametrize("num_tokens", [1, 64, 65, 96, 127, 128, 129, 200, 255, 256, 257, 1025, 4096])
@torch.no_grad()
def test_mega_mhc(is_shifted, sf_layout, hidden, num_tokens):
    inputs, duplicates = make_inputs(num_tokens, hidden, 2026 + num_tokens + hidden, is_shifted)
    outputs = make_outputs(inputs, sf_layout)
    reference = mhc_reference(**inputs, sf_layout=sf_layout)
    run_mega_mhc(inputs, outputs)
    actual = logical_outputs(outputs)
    check_correctness(actual, reference, (is_shifted, sf_layout, num_tokens, hidden))
    expected = {name: tensor.clone() for name, tensor in actual.items()}
    for _ in range(3):
        run_mega_mhc(inputs, outputs)
        for name, tensor in logical_outputs(outputs).items():
            assert torch.equal(comparable(tensor), comparable(expected[name])), name
    for token_idx in duplicates:
        for name, tensor in actual.items():
            assert torch.equal(comparable(tensor[token_idx]), comparable(tensor[0])), f"{name}, token={token_idx}"


@torch.no_grad()
def test_mega_mhc_large():
    inputs, _ = make_inputs(32768, 7168, 7, True)
    outputs = make_outputs(inputs, "extra")
    reference = mhc_reference(**inputs, sf_layout="extra")
    run_mega_mhc(inputs, outputs)
    check_correctness(logical_outputs(outputs), reference, "large")


@torch.no_grad()
def test_mega_mhc_api_contract():
    num_tokens, hidden = 64, 4096
    inputs, _ = make_inputs(num_tokens, hidden, 2026, False)

    # deterministic mode: fixed splits, results independent of the number of tokens
    deep_gemm.use_deterministic_algorithms(True)
    try:
        outputs = make_outputs(inputs, "bf16")
        run_mega_mhc(inputs, outputs)
        extra_inputs, _ = make_inputs(4096 - num_tokens, hidden, 99, False)
        large_inputs = dict(inputs)
        for name in ("x", "residual", "post_mix", "comb_res_mix"):
            large_inputs[name] = torch.cat([inputs[name], extra_inputs[name]])
        large_outputs = make_outputs(large_inputs, "bf16")
        run_mega_mhc(large_inputs, large_outputs)
        for name, tensor in logical_outputs(outputs).items():
            assert torch.equal(tensor, logical_outputs(large_outputs)[name][:num_tokens]), name
    finally:
        deep_gemm.use_deterministic_algorithms(False)

    # CUDA graphs replay the eager results
    expected_outputs = make_outputs(inputs, "bf16")
    run_mega_mhc(inputs, expected_outputs)
    graph_outputs = make_outputs(inputs, "bf16")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_mega_mhc(inputs, graph_outputs)
    graph.replay()
    torch.cuda.synchronize()
    for name, tensor in logical_outputs(graph_outputs).items():
        assert torch.equal(tensor, logical_outputs(expected_outputs)[name]), f"CUDA graph: {name}"

    # FP8-only calls match the FP8 outputs of calls with BF16 outputs
    for is_shifted in (False, True):
        inputs, _ = make_inputs(num_tokens, hidden, 2026, is_shifted)
        outputs = make_outputs(inputs, "extra")
        run_mega_mhc(inputs, outputs)
        names = ("y_fp8", "y_routed_sf", "y_shared_sf_storage")
        expected = {name: outputs[name].clone() for name in names}
        small_inputs, _ = make_inputs(1, hidden, 2027, is_shifted)
        run_mega_mhc(small_inputs, make_outputs(small_inputs, "extra"))
        run_mega_mhc(inputs, outputs, y_bf16=None)
        for name in names:
            assert torch.equal(comparable(outputs[name]), comparable(expected[name])), f"FP8-only: {name}"

    # concurrent streams
    inputs, _ = make_inputs(num_tokens, hidden, 2028, False)
    expected_outputs = make_outputs(inputs, "bf16")
    run_mega_mhc(inputs, expected_outputs)
    torch.cuda.synchronize()
    streams = (torch.cuda.Stream(), torch.cuda.Stream())
    stream_outputs = []
    for stream in streams:
        with torch.cuda.stream(stream):
            outputs = make_outputs(inputs, "bf16")
            run_mega_mhc(inputs, outputs)
            stream_outputs.append(outputs)
    for stream in streams:
        stream.synchronize()
    for outputs in stream_outputs:
        for name, tensor in logical_outputs(outputs).items():
            assert torch.equal(tensor, logical_outputs(expected_outputs)[name]), f"multi-stream: {name}"
