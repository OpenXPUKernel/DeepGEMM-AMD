"""Correctness and storage-contract tests for the CDNA4 HIP FP8 GEMM backend.

Set DG_TEST_LARGE=1 to also exercise tensor addresses beyond 4 GiB. Those tests
use sparse strided storage, so their arithmetic stays small despite the allocation.
"""
import os

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


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("layout", ["kk", "km", "mk", "mm"])
def test_multi_tile_launch(dtype, layout):
    # Exact products expose extra blocks overwriting the first output tile.
    # These dimensions exercise several rows and columns of full output tiles.
    m, n, k = 4096, 4096, 256
    a = torch.ones((m, k), device="cuda", dtype=torch.float8_e4m3fn)
    b = torch.ones((n, k), device="cuda", dtype=torch.float8_e4m3fn)
    a, _ = _place_operand(a, layout[0])
    b, _ = _place_operand(b, layout[1])
    sa = torch.ones((m, k // 32), device="cuda", dtype=torch.float32)
    sb = torch.ones((n, k // 32), device="cuda", dtype=torch.float32)
    out = torch.empty((m, n), device="cuda", dtype=dtype)
    for _ in range(2):
        out.fill_(float('nan'))
        deep_gemm.fp8_gemm_nt((a, sa), (b, sb), out, recipe=(1, 1, 32))
        assert bool((out == k).all()), 'incorrect output from a multi-tile launch'


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
    deep_gemm.fp8_gemm_nt((zero_k_a, torch.empty((m, 0), device="cuda", dtype=torch.float32)),
                          (zero_k_b, torch.empty((n, 0), device="cuda", dtype=torch.float32)),
                          zero_k_d, recipe_a=(1, 32), recipe_b=(1, 32))
    assert bool((zero_k_d == 0).all())


RECIPES = {
    "32": ((1, 32), (1, 32)),
    "128": ((1, 128), (1, 128)),
    "mixed": ((1, 32), (1, 128)),
    "block": ((1, 128), (128, 128)),
}
LAYOUTS = ("kk", "km", "mk", "mm")


def _scaled_operand(rows, k, recipe):
    """Small exact FP8 values and varying powers of two isolate indexing errors."""
    values = torch.arange(rows * k, device="cuda", dtype=torch.float32).reshape(rows, k)
    q = ((values % 7 - 3) / 4).to(torch.float8_e4m3fn)
    scale_rows = deep_gemm.ceil_div(rows, recipe[0])
    scale_cols = deep_gemm.ceil_div(k, recipe[1])
    exponents = torch.arange(scale_rows * scale_cols, device="cuda").reshape(scale_rows, scale_cols)
    sf = torch.exp2((exponents % 4 - 2).float())
    return q, sf


def _dequant_recipe(q, sf, recipe):
    scales = sf.repeat_interleave(recipe[0], dim=0).repeat_interleave(recipe[1], dim=1)
    return q.float() * scales[:q.size(0), :q.size(1)]


def _place_operand(q, major, offset=0, row_stride=None):
    """Keep NaN codes in every byte outside the logical operand."""
    physical = q.T if major == "m" else q
    rows, cols = physical.shape
    ld = deep_gemm.align(cols, 4) if row_stride is None else row_stride
    backing = torch.full((offset + rows * ld,), 0x7F, device="cuda", dtype=torch.uint8)
    view = backing[offset:].view(rows, ld)[:, :cols].view(torch.float8_e4m3fn)
    view.copy_(physical)
    return (view.T if major == "m" else view), backing


def _pack_operand(operand, recipe):
    q, sf = operand
    return q, deep_gemm.transform_sf_into_required_layout(sf, q.size(0), q.size(1), recipe)


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("recipe_name", RECIPES)
@pytest.mark.parametrize("packed", [False, True], ids=["fp32_scales", "packed_scales"])
def test_native_layouts_and_scale_recipes(layout, recipe_name, packed):
    m, n, k = 33, 65, 521
    recipe_a, recipe_b = RECIPES[recipe_name]
    a, b = _scaled_operand(m, k, recipe_a), _scaled_operand(n, k, recipe_b)
    ref = _dequant_recipe(*a, recipe_a) @ _dequant_recipe(*b, recipe_b).T
    qa, storage_a = _place_operand(a[0], layout[0])
    qb, storage_b = _place_operand(b[0], layout[1])
    snapshots = storage_a.clone(), storage_b.clone()
    a, b = (qa, a[1]), (qb, b[1])
    if packed:
        a, b = _pack_operand(a, recipe_a), _pack_operand(b, recipe_b)
        recipe_a, recipe_b = (1, recipe_a[1]), (1, recipe_b[1])
    out = torch.full((m, n), float("nan"), device="cuda", dtype=torch.float32)
    kwargs = dict(recipe_a=recipe_a, recipe_b=recipe_b)
    # The default FP32 recipe is per-token A and per-block B.
    if recipe_name == "block" and not packed:
        kwargs = {}
    deep_gemm.fp8_gemm_nt(a, b, out, **kwargs)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
    again = torch.empty_like(out)
    deep_gemm.fp8_gemm_nt(a, b, again, **kwargs)
    assert torch.equal(out, again)
    assert torch.equal(storage_a, snapshots[0])
    assert torch.equal(storage_b, snapshots[1])


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("shape,offset", [
    ((4, 8, 132), 1),
    ((4, 8, 132), 2),
    ((1, 1, 1), 4),
    ((3, 5, 7), 4),
    ((33, 65, 1001), 4),
    ((257, 255, 2001), 0),
])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_ragged_padding_and_unaligned_bases(layout, shape, offset, dtype):
    m, n, k = shape
    recipe = (1, 32)
    a, b = _scaled_operand(m, k, recipe), _scaled_operand(n, k, recipe)
    ref = _dequant_recipe(*a, recipe) @ _dequant_recipe(*b, recipe).T
    qa, _ = _place_operand(a[0], layout[0], offset)
    qb, _ = _place_operand(b[0], layout[1], offset)
    a, b = _pack_operand((qa, a[1]), recipe), _pack_operand((qb, b[1]), recipe)
    ld = deep_gemm.align(n + 8, 8)
    storage = torch.full((m + 2, ld), -123.0, device="cuda", dtype=dtype)
    out = storage[1:m + 1, 4:n + 4]
    out.fill_(float("nan"))
    deep_gemm.fp8_gemm_nt(a, b, out, recipe_a=recipe, recipe_b=recipe)
    torch.testing.assert_close(out.float(), ref, atol=1e-2, rtol=1e-2)
    assert bool((storage[0] == -123).all())
    assert bool((storage[-1] == -123).all())
    assert bool((storage[1:-1, :4] == -123).all())
    assert bool((storage[1:-1, n + 4:] == -123).all())


@pytest.mark.parametrize("layout,alias", [("km", "nn"), ("mm", "tn"), ("mk", "tt")])
@pytest.mark.parametrize("recipe_name", ["128", "mixed", "block"])
def test_aliases_preserve_native_layout_and_scale_blocks(layout, alias, recipe_name):
    m, n, k = 33, 65, 521
    recipe_a, recipe_b = RECIPES[recipe_name]
    a, b = _scaled_operand(m, k, recipe_a), _scaled_operand(n, k, recipe_b)
    ref = _dequant_recipe(*a, recipe_a) @ _dequant_recipe(*b, recipe_b).T
    qa, _ = _place_operand(a[0], layout[0])
    qb, _ = _place_operand(b[0], layout[1])
    a, b = _pack_operand((qa, a[1]), recipe_a), _pack_operand((qb, b[1]), recipe_b)
    recipe_a, recipe_b = (1, recipe_a[1]), (1, recipe_b[1])
    if layout[0] == "m":
        a = a[0].T, a[1].T
    if layout[1] == "m":
        b = b[0].T, b[1].T
    out = torch.empty((m, n), device="cuda", dtype=torch.float32)
    getattr(deep_gemm, f"fp8_gemm_{alias}")(
        a, b, out, recipe_a=recipe_a, recipe_b=recipe_b)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("recipe", [(1, 32), (1, 128), (128, 128)])
def test_packed_scale_transform_preserves_original_blocks(recipe):
    rows, k = 257, 1031
    _, sf = _scaled_operand(rows, k, recipe)
    packed = deep_gemm.transform_sf_into_required_layout(sf, rows, k, recipe)
    assert packed.shape == (rows, deep_gemm.ceil_div(k, recipe[1] * 4))
    assert packed.stride(0) == 1
    assert packed.stride(1) == deep_gemm.align(packed.size(0), 4)
    again = deep_gemm.transform_sf_into_required_layout(packed, rows, k, (1, recipe[1]))
    assert again.data_ptr() == packed.data_ptr()
    assert again.stride() == packed.stride()
    assert torch.equal(again, packed)
    # Decode only in the test: packing must retain one exponent per original block.
    codes = packed.contiguous().view(torch.uint8).reshape(packed.size(0), -1)
    expected = (sf.contiguous().view(torch.int32) >> 23).to(torch.uint8)
    expected = expected.repeat_interleave(recipe[0], dim=0)[:rows]
    assert torch.equal(codes[:, :sf.size(1)], expected)


@pytest.mark.parametrize("major,shape,offset,row_stride", [
    ("k", (33, 137), 0, 137),
    ("m", (33, 136), 0, 33),
    ("k", (33, 137), 1, None),
    ("k", (33, 137), 3, None),
    ("m", (33, 136), 1, None),
    ("m", (33, 136), 2, None),
    ("k", (2, 4), 0, 1 << 24),
    ("m", (4, 2), 0, 1 << 24),
])
def test_rejects_unreadable_operand_storage(major, shape, offset, row_stride):
    m, k = shape
    n, recipe = 8, (1, 32)
    a, b = _scaled_operand(m, k, recipe), _scaled_operand(n, k, recipe)
    qa, _ = _place_operand(a[0], major, offset, row_stride)
    qb, _ = _place_operand(b[0], "k")
    out = torch.empty((m, n), device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError):
        deep_gemm.fp8_gemm_nt((qa, a[1]), (qb, b[1]), out,
                              recipe_a=recipe, recipe_b=recipe)


@pytest.mark.parametrize("kind", ["k_major", "unaligned_column", "overlapping_columns"])
def test_rejects_invalid_packed_scale_strides(kind):
    m, n, k = 33, 65, 521
    recipe = (1, 32)
    a, b = _scaled_operand(m, k, recipe), _scaled_operand(n, k, recipe)
    qa, _ = _place_operand(a[0], "k")
    qb, _ = _place_operand(b[0], "k")
    a, b = _pack_operand((qa, a[1]), recipe), _pack_operand((qb, b[1]), recipe)
    if kind == "k_major":
        bad_sf = a[1].contiguous()
    else:
        ld = deep_gemm.align(m, 4) + 2 if kind == "unaligned_column" else 32
        bad_sf = torch.empty_strided(a[1].shape, (1, ld), device="cuda", dtype=torch.int32)
    out = torch.empty((m, n), device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError):
        deep_gemm.fp8_gemm_nt((a[0], bad_sf), b, out, recipe_a=recipe, recipe_b=recipe)


def test_packed_block_scales_require_expanded_mn_rows():
    # Block-scale broadcasting belongs to FP32 preparation, not packed calls.
    sf = torch.empty_strided((1, 2), (1, 4), device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError):
        deep_gemm.transform_sf_into_required_layout(sf, 33, 521, (128, 128))


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("recipe_name", ["32", "128", "mixed", "block"])
def test_packed_native_call_does_not_copy_or_expand_inputs(layout, recipe_name):
    m, n, k = 33, 65, 521
    recipe_a, recipe_b = RECIPES[recipe_name]
    a, b = _scaled_operand(m, k, recipe_a), _scaled_operand(n, k, recipe_b)
    qa, _ = _place_operand(a[0], layout[0], offset=4)
    qb, _ = _place_operand(b[0], layout[1], offset=4)
    a, b = _pack_operand((qa, a[1]), recipe_a), _pack_operand((qb, b[1]), recipe_b)
    recipe_a, recipe_b = (1, recipe_a[1]), (1, recipe_b[1])
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    kwargs = dict(recipe_a=recipe_a, recipe_b=recipe_b)
    deep_gemm.fp8_gemm_nt(a, b, out, **kwargs)  # Compile outside the profile.
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        deep_gemm.fp8_gemm_nt(a, b, out, **kwargs)
        torch.cuda.synchronize()
    operations = {event.key for event in profile.key_averages()}
    forbidden = {"aten::contiguous", "aten::copy_", "aten::zero_", "aten::repeat_interleave"}
    assert not operations & forbidden, f"unexpected input preparation: {operations & forbidden}"


@pytest.mark.parametrize("c_layout", ["alias", "same_stride", "different_stride"])
def test_accumulation_with_padded_output(c_layout):
    m, n, k = 33, 65, 521
    recipe = (1, 128)
    a, b = _scaled_operand(m, k, recipe), _scaled_operand(n, k, recipe)
    ref = _dequant_recipe(*a, recipe) @ _dequant_recipe(*b, recipe).T
    qa, _ = _place_operand(a[0], "m")
    qb, _ = _place_operand(b[0], "k")
    a, b = _pack_operand((qa, a[1]), recipe), _pack_operand((qb, b[1]), recipe)
    storage = torch.full((m, n + 7), -123.0, device="cuda", dtype=torch.float32)
    out = storage[:, :n]
    if c_layout == "alias":
        c = out
    elif c_layout == "same_stride":
        c = torch.empty_like(storage)[:, :n]
    else:
        c = torch.empty((m, n), device="cuda", dtype=torch.float32)
    c.fill_(0.25)
    alpha = -0.75
    deep_gemm.fp8_gemm_nt(a, b, out, c=c, alpha=alpha, recipe_a=recipe, recipe_b=recipe)
    torch.testing.assert_close(out, alpha * ref + 0.25, atol=1e-2, rtol=1e-2)
    assert bool((storage[:, n:] == -123).all())
    if c_layout != "alias":
        assert bool((c == 0.25).all())


@pytest.mark.parametrize("shape", [(0, 65, 128), (33, 0, 128), (33, 65, 0)])
def test_empty_plan(shape):
    assert deep_gemm.get_plan(*shape) == (0, 0)


def test_singleton_k_ignores_unused_stride():
    # The K dimension has one element, so its stride does not determine layout.
    a = torch.empty_strided((17, 1), (4, 8), device="cuda", dtype=torch.float8_e4m3fn)
    b = torch.empty_strided((33, 1), (4, 12), device="cuda", dtype=torch.float8_e4m3fn)
    a.fill_(1.0)
    b.fill_(2.0)
    sa = torch.ones((17, 1), device="cuda")
    sb = torch.ones((33, 1), device="cuda")
    out = torch.empty((17, 33), device="cuda", dtype=torch.float32)
    deep_gemm.fp8_gemm_nt((a, sa), (b, sb), out, recipe=(1, 1, 32))
    assert bool((out == 2).all())


@pytest.mark.parametrize("shape", [(1, 1, 32), (1, 1, 384), (257, 511, 385),
                                  (33, 65, 1001), (1024, 2048, 8192)])
def test_split_plan_has_no_empty_partitions(shape):
    m, n, k = shape
    tiles, splits = deep_gemm.get_plan(m, n, k)
    assert tiles == deep_gemm.ceil_div(m, 256) * deep_gemm.ceil_div(n, 256)
    k_tiles = deep_gemm.ceil_div(k, 128)
    assert 1 <= splits <= k_tiles
    tiles_per_split = deep_gemm.ceil_div(k_tiles, splits)
    assert deep_gemm.ceil_div(k_tiles, tiles_per_split) == splits
    assert (splits - 1) * tiles_per_split < k_tiles


def test_split_normalization_with_uneven_k_tiles():
    # Leave roughly five workgroups per output tile. Twelve K tiles then need
    # fewer active partitions than the initial occupancy estimate suggests.
    units = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    n = max(1, units // 5) * 256
    _, splits = deep_gemm.get_plan(1, n, 1536)
    assert splits > 1
    assert deep_gemm.ceil_div(12, deep_gemm.ceil_div(12, splits)) == splits


@pytest.mark.skipif(os.environ.get("DG_TEST_LARGE") != "1", reason="set DG_TEST_LARGE=1 for >4 GiB allocations")
@pytest.mark.parametrize("large_tensor", ["a", "b", "a_mn", "b_mn", "d", "scales", "b_scales"])
def test_offsets_beyond_four_gib(large_tensor):
    """Touch a few logical elements beyond 32-bit offsets, without a huge GEMM."""
    recipe = (1, 32)
    m, n, k = (258, 4, 4) if large_tensor == "a" else (4, 258, 4)
    if large_tensor == "d":
        m, n, k = 3, 4, 4
    elif large_tensor in ("scales", "b_scales"):
        m, n, k = 4, 4, 640
    elif large_tensor in ("a_mn", "b_mn"):
        m, n, k = 4, 4, 258
    qa = torch.zeros((m, k), device="cuda", dtype=torch.float8_e4m3fn)
    qb = torch.ones((n, k), device="cuda", dtype=torch.float8_e4m3fn)
    sa = torch.ones((m, deep_gemm.ceil_div(k, 32)), device="cuda")
    sb = torch.ones((n, deep_gemm.ceil_div(k, 32)), device="cuda")
    if large_tensor == "scales":
        sa[:, -1] = 2.0  # A wrapped scale address would incorrectly read one.
    elif large_tensor == "b_scales":
        sb[:, -1] = 2.0
    sa = deep_gemm.transform_sf_into_required_layout(sa, m, k, recipe)
    sb = deep_gemm.transform_sf_into_required_layout(sb, n, k, recipe)
    out = torch.empty((m, n), device="cuda", dtype=torch.float32)
    if large_tensor in ("a_mn", "b_mn"):
        q = qa if large_tensor == "a_mn" else qb
        large = torch.empty_strided(q.shape, (1, (1 << 24) - 4), device="cuda", dtype=q.dtype)
        large.zero_()
        large[:, -1].fill_(1.0)
        assert large.untyped_storage().nbytes() > 1 << 32
        if large_tensor == "a_mn":
            qa = large
            qb, _ = _place_operand(qb, "k")
        else:
            qb = large
            qa.fill_(1.0)
            qa, _ = _place_operand(qa, "k")
    elif large_tensor in ("a", "b"):
        q = qa if large_tensor == "a" else qb
        large = torch.empty_strided(q.shape, ((1 << 24) - 4, 1), device="cuda", dtype=q.dtype)
        large.zero_()
        large[-1].fill_(1.0)
        assert large.untyped_storage().nbytes() > 1 << 32
        if large_tensor == "a":
            qa = large
        else:
            qb = large
            qa.fill_(1.0)
    elif large_tensor == "d":
        out = torch.empty_strided((m, n), (1 << 29, 1), device="cuda", dtype=torch.float32)
        out.fill_(float("nan"))
        qa.fill_(1.0)
        assert out.untyped_storage().nbytes() > 1 << 32
    else:
        scales = sa if large_tensor == "scales" else sb
        large = torch.empty_strided(scales.shape, (1, 1 << 28), device="cuda", dtype=torch.int32)
        large.copy_(scales)
        if large_tensor == "scales":
            sa = large
        else:
            sb = large
        qa[:, -32:].fill_(1.0)
        assert large.untyped_storage().nbytes() > 1 << 32
    deep_gemm.fp8_gemm_nt((qa, sa), (qb, sb), out, recipe_a=recipe, recipe_b=recipe)
    torch.cuda.synchronize()
    if large_tensor == "a":
        assert bool((out[:-1] == 0).all())
        assert bool((out[-1] == k).all())
    elif large_tensor == "b":
        assert bool((out[:, :-1] == 0).all())
        assert bool((out[:, -1] == k).all())
    elif large_tensor in ("a_mn", "b_mn"):
        assert bool((out == 1).all())
    else:
        assert bool((out == (64 if large_tensor in ("scales", "b_scales") else k)).all())
