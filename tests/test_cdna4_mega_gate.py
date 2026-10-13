"""Correctness tests for the CDNA4 bf16_mega_gate with the checks of tests/test_mega_gate.py: top-k
selection up to numerical ties, weights, shared experts, physical replicas, masked, fixed and random
routings, and bitwise repeatability. The reference logits come from a double-precision matmul.
"""
import pytest

try:
    import torch
    import torch.nn.functional as F
except Exception as exc:  # pragma: no cover - collection-only environments
    pytest.skip(f"PyTorch is required: {exc}", allow_module_level=True)

if torch.version.hip is None or not torch.cuda.is_available():
    pytest.skip("CDNA4 tests require a ROCm device", allow_module_level=True)

import deep_gemm
from deep_gemm.testing import calc_diff

NUM_TOPK = 6
NUM_SHARED_EXPERTS = 1
ROUTED_SCALING_FACTOR = 1.5
NUM_EXTRA_EXPERTS = 32

GATE_CONFIGS = (
    (64, 2, 6), (72, 1, 6), (96, 2, 6), (108, 2, 6), (128, 2, 6), (144, 2, 6),
    (256, 1, 6), (256, 2, 8), (384, 1, 6), (384, 1, 9), (512, 1, 6),
)


def make_physical_map(num_logical_experts):
    to_physical_map = torch.full((num_logical_experts, NUM_EXTRA_EXPERTS + 1), -1, dtype=torch.int32, device="cuda")
    logical_count = torch.ones((num_logical_experts, ), dtype=torch.int32, device="cuda")
    to_physical_map[:, 0] = torch.arange(num_logical_experts, dtype=torch.int32, device="cuda")
    num_duplicated = min(num_logical_experts, NUM_EXTRA_EXPERTS)
    to_physical_map[:num_duplicated, 1] = num_logical_experts + torch.arange(num_duplicated, dtype=torch.int32,
                                                                             device="cuda")
    logical_count[:num_duplicated] = 2
    return to_physical_map, logical_count


def make_inputs(num_tokens, hidden, num_routed_experts, use_shared_as_routed=False, with_bias=True,
                with_image_bias=False, with_mask=False, with_physical_map=True, num_shared_experts=NUM_SHARED_EXPERTS):
    num_logical_experts = num_routed_experts + (num_shared_experts if use_shared_as_routed else 0)
    mask = torch.rand((num_tokens, ), device="cuda") >= 0.25 if with_mask else None
    if mask is not None:
        mask[0] = True
    to_physical_map, logical_count = make_physical_map(num_logical_experts) if with_physical_map else (None, None)
    return dict(
        x=torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda"),
        weight=torch.randn((num_routed_experts, hidden), dtype=torch.bfloat16, device="cuda").mul_(hidden ** -0.5),
        mask=mask,
        bias=torch.randn((num_routed_experts, ), dtype=torch.float32, device="cuda") if with_bias else None,
        image_bias=torch.randn((num_routed_experts, ), dtype=torch.float32, device="cuda") if with_image_bias else None,
        image_token_mask=torch.rand((num_tokens, ), device="cuda") < 0.5 if with_image_bias else None,
        to_physical_map=to_physical_map,
        logical_count=logical_count,
    )


def make_outputs(inputs, use_shared_as_routed=False, with_unmapped_topk_idx=False, num_topk=NUM_TOPK,
                 num_shared_experts=NUM_SHARED_EXPERTS):
    shape = (inputs["x"].shape[0], num_topk + (num_shared_experts if use_shared_as_routed else 0))
    return dict(
        topk_idx=torch.empty(shape, dtype=torch.int64, device="cuda"),
        topk_weights=torch.empty(shape, dtype=torch.float32, device="cuda"),
        unmapped_topk_idx=(torch.empty((shape[0], num_topk), dtype=torch.int64, device="cuda")
                           if with_unmapped_topk_idx else None),
    )


def run_mega_gate(inputs, outputs, use_shared_as_routed=False, num_topk=NUM_TOPK,
                  num_shared_experts=NUM_SHARED_EXPERTS, **overrides):
    kwargs = {**inputs, **overrides}
    x, weight = kwargs.pop("x"), kwargs.pop("weight")
    return deep_gemm.bf16_mega_gate(
        x, weight, num_topk, use_shared_as_routed, num_shared_experts, ROUTED_SCALING_FACTOR, 0,
        unmapped_topk_idx=outputs["unmapped_topk_idx"],
        out=(outputs["topk_idx"], outputs["topk_weights"]), **kwargs)


def reference_logits(inputs):
    return (inputs["x"].double() @ inputs["weight"].double().T).float()


def check_topk_weights(inputs, outputs, logits, mask=None, force_random=None, fix_routing_mask=None,
                       num_topk=NUM_TOPK):
    fused_idx, fused_weights = outputs["topk_idx"], outputs["topk_weights"]
    num_tokens, num_routed_experts = logits.shape
    active = mask if mask is not None else torch.ones((num_tokens, ), dtype=torch.bool, device="cuda")
    score_based = active if force_random is None else active & ~force_random

    if score_based.any():
        scores = F.softplus(logits).sqrt()
        logical_topk_idx = outputs["unmapped_topk_idx"]
        if logical_topk_idx is None:
            logical_topk_idx = fused_idx[:, :num_topk]
            to_physical_map = inputs["to_physical_map"]
            if to_physical_map is not None:
                logical_topk_idx = torch.where(logical_topk_idx < to_physical_map.shape[0], logical_topk_idx,
                                               logical_topk_idx - to_physical_map.shape[0])

        selected_idx = logical_topk_idx[score_based]
        assert torch.all((selected_idx >= 0) & (selected_idx < num_routed_experts)).item()

        ranking_bias = (inputs["bias"] if inputs["bias"] is not None else
                        torch.zeros((num_routed_experts, ), dtype=torch.float32, device="cuda"))
        if inputs["image_token_mask"] is not None:
            ranking_bias = torch.where(inputs["image_token_mask"][:, None], inputs["image_bias"], ranking_bias)
        ranking = scores + ranking_bias

        def reference_weights(row_mask, topk_idx):
            topk_scores = scores[row_mask].gather(1, topk_idx)
            return topk_scores / (topk_scores.sum(dim=1, keepdim=True) + 1e-20) * ROUTED_SCALING_FACTOR

        weight_diff = float(calc_diff(fused_weights[score_based, :num_topk],
                                      reference_weights(score_based, selected_idx)))
        assert weight_diff < 1e-11, f"{weight_diff=}"

        ranked = score_based if fix_routing_mask is None else score_based & ~fix_routing_mask
        if ranked.any():
            ranked_idx = logical_topk_idx[ranked]
            sorted_idx = ranked_idx.sort(dim=1).values
            assert torch.all(sorted_idx[:, 1:] != sorted_idx[:, :-1]).item()

            selected_ranking = ranking[ranked].gather(1, ranked_idx)
            unselected_max = ranking[ranked].scatter(1, ranked_idx, float("-inf")).max(dim=1).values
            crossing = unselected_max - selected_ranking.min(dim=1).values
            tolerance = 1e-5 + 1e-5 * ranking[ranked].abs().amax(dim=1)
            assert torch.all(crossing <= tolerance).item(), f"top-k beyond numerical ties: {crossing.max().item()=}"

            # the selected set is unique when the k-th and (k+1)-th rankings are apart beyond the tolerance
            separable = torch.ones_like(tolerance, dtype=torch.bool)
            if num_topk < num_routed_experts:
                values = ranking[ranked].topk(num_topk + 1, dim=1).values
                separable = values[:, num_topk - 1] - values[:, num_topk] > tolerance
            if separable.any():
                ref_idx = ranking[ranked].topk(num_topk, dim=1).indices
                assert torch.equal(sorted_idx[separable], ref_idx.sort(dim=1).values[separable])

        if fused_weights.shape[1] > num_topk:
            assert torch.all(fused_weights[score_based, num_topk:] == 1.0).item()

    if (~active).any():
        assert torch.all(fused_idx[~active] == -1).item()
        assert torch.all(fused_weights[~active] == 0).item()
        if outputs["unmapped_topk_idx"] is not None:
            assert torch.all(outputs["unmapped_topk_idx"][~active] == -1).item()


def check_physical_map(inputs, outputs, use_shared_as_routed, num_topk, num_shared_experts):
    to_physical_map = inputs["to_physical_map"]
    if to_physical_map is None or outputs["unmapped_topk_idx"] is None:
        return
    active = inputs["mask"] if inputs["mask"] is not None else torch.ones_like(outputs["topk_idx"][:, 0], dtype=torch.bool)
    logical = outputs["unmapped_topk_idx"][active]
    num_routed_experts = inputs["weight"].shape[0]
    if use_shared_as_routed:
        shared = num_routed_experts + torch.arange(num_shared_experts, device="cuda")
        logical = torch.cat([logical, shared.expand(logical.shape[0], -1)], dim=1)
    tokens = torch.arange(outputs["topk_idx"].shape[0], device="cuda")[active]
    replica = tokens[:, None] * 23333 % inputs["logical_count"][logical]
    assert torch.equal(outputs["topk_idx"][active], to_physical_map[logical, replica].long())


def check_case(num_tokens, hidden, num_routed_experts, use_shared_as_routed=False, num_topk=NUM_TOPK,
               num_shared_experts=NUM_SHARED_EXPERTS, with_unmapped_topk_idx=False, num_repeats=3, **input_options):
    inputs = make_inputs(num_tokens, hidden, num_routed_experts, use_shared_as_routed,
                         num_shared_experts=num_shared_experts, **input_options)
    outputs = make_outputs(inputs, use_shared_as_routed, with_unmapped_topk_idx, num_topk, num_shared_experts)
    run_mega_gate(inputs, outputs, use_shared_as_routed, num_topk, num_shared_experts)
    check_topk_weights(inputs, outputs, reference_logits(inputs), inputs["mask"], num_topk=num_topk)
    check_physical_map(inputs, outputs, use_shared_as_routed, num_topk, num_shared_experts)
    expected = {name: tensor.clone() for name, tensor in outputs.items() if tensor is not None}
    for _ in range(num_repeats):
        run_mega_gate(inputs, outputs, use_shared_as_routed, num_topk, num_shared_experts)
        for name, tensor in expected.items():
            assert torch.equal(outputs[name], tensor), name


@pytest.mark.parametrize("num_tokens", [1, 3, 128, 4096])
@pytest.mark.parametrize("hidden", [4096, 7168])
@pytest.mark.parametrize("num_routed_experts", [128, 256, 264, 384, 512])
def test_mega_gate(num_tokens, hidden, num_routed_experts):
    torch.manual_seed(0)
    check_case(num_tokens, hidden, num_routed_experts, with_unmapped_topk_idx=True)


@pytest.mark.parametrize("num_routed_experts,num_shared_experts,num_topk", GATE_CONFIGS)
@pytest.mark.parametrize("use_shared_as_routed", [False, True])
def test_mega_gate_configs(num_routed_experts, num_shared_experts, num_topk, use_shared_as_routed):
    if use_shared_as_routed and (num_topk % num_shared_experts or num_routed_experts % (num_topk // num_shared_experts)):
        pytest.skip("shared experts cannot be routed for this configuration")
    torch.manual_seed(2)
    check_case(16, 4096, num_routed_experts, use_shared_as_routed=use_shared_as_routed, num_topk=num_topk,
               num_shared_experts=num_shared_experts, with_unmapped_topk_idx=True)


def test_mega_gate_options():
    torch.manual_seed(2)
    check_case(16, 4096, 128, with_bias=False, with_physical_map=False)
    check_case(129, 4352, 260, with_image_bias=True, with_mask=True, with_unmapped_topk_idx=True)

    inputs = make_inputs(128, 4096, 384, with_image_bias=True)
    inputs["bias"].fill_(1e8)
    inputs["image_bias"].fill_(-1e8)
    outputs = make_outputs(inputs)
    run_mega_gate(inputs, outputs)
    check_topk_weights(inputs, outputs, reference_logits(inputs))

    num_topk = 8
    inputs = make_inputs(8, 4096, 128, with_bias=False, with_physical_map=False)
    inputs["x"].zero_()
    inputs["weight"].zero_()
    topk_idx, _ = run_mega_gate(inputs, make_outputs(inputs, num_topk=num_topk), num_topk=num_topk)
    assert torch.equal(topk_idx, torch.arange(num_topk, dtype=torch.int64, device="cuda").expand(8, -1))

    inputs = make_inputs(16, 4096, 128, with_physical_map=False)
    outputs = make_outputs(inputs, with_unmapped_topk_idx=True, num_topk=num_topk)
    outputs["unmapped_topk_idx"] = torch.empty((16, num_topk + 3), dtype=torch.int64, device="cuda")[:, :num_topk]
    run_mega_gate(inputs, outputs, num_topk=num_topk)
    check_topk_weights(inputs, outputs, reference_logits(inputs), num_topk=num_topk)

    inputs = make_inputs(0, 4096, 128, with_physical_map=False)
    topk_idx, topk_weights = run_mega_gate(inputs, make_outputs(inputs, num_topk=num_topk), num_topk=num_topk)
    assert topk_idx.shape == (0, num_topk) and topk_weights.shape == (0, num_topk)


def test_mega_gate_routing_controls():
    torch.manual_seed(2)
    num_tokens, hidden, num_routed_experts = 512, 4096, 384
    inputs = make_inputs(num_tokens, hidden, num_routed_experts, use_shared_as_routed=True, with_image_bias=True)
    outputs = make_outputs(inputs, use_shared_as_routed=True, with_unmapped_topk_idx=True)
    token_idx = torch.arange(num_tokens, device="cuda")
    mask, fix_routing_mask, force_random = token_idx % 11 != 0, token_idx % 3 == 0, token_idx % 5 == 0
    fixed_topk_idx = torch.randint(0, num_routed_experts, (num_tokens, NUM_TOPK), dtype=torch.int64, device="cuda")
    unmapped_topk_idx = outputs["unmapped_topk_idx"]
    unmapped_topk_idx.copy_(fixed_topk_idx)

    fused_idx, fused_weights = run_mega_gate(inputs, outputs, True, mask=mask, fix_routing_mask=fix_routing_mask,
                                             force_random=force_random)
    check_topk_weights(inputs, outputs, reference_logits(inputs), mask, force_random, fix_routing_mask)

    active_force_random = force_random & mask
    fixed = fix_routing_mask & mask & ~force_random
    assert torch.equal(unmapped_topk_idx[fixed], fixed_topk_idx[fixed])
    random_idx, random_weights = fused_idx[active_force_random], fused_weights[active_force_random]
    assert torch.all((random_idx >= 0) & (random_idx < inputs["logical_count"].sum())).item()
    assert torch.all(random_weights > 0).item()
    sorted_random_idx = random_idx.sort(dim=1).values
    assert torch.all(sorted_random_idx[:, 1:] != sorted_random_idx[:, :-1]).item()
    assert torch.all(unmapped_topk_idx[active_force_random] == -1).item()

    num_topk = 8
    inputs = make_inputs(32, 4096, 128, with_physical_map=False)
    outputs = make_outputs(inputs, with_unmapped_topk_idx=True, num_topk=num_topk)
    force_random = torch.ones((32, ), dtype=torch.bool, device="cuda")
    topk_idx, topk_weights = run_mega_gate(inputs, outputs, num_topk=num_topk, force_random=force_random)
    assert torch.all((topk_idx >= 0) & (topk_idx < 128)).item()
    assert torch.all(topk_weights > 0).item()
    sorted_topk_idx = topk_idx.sort(dim=1).values
    assert torch.all(sorted_topk_idx[:, 1:] != sorted_topk_idx[:, :-1]).item()
    assert torch.all(outputs["unmapped_topk_idx"] == -1).item()
