"""CDNA4 FP8 GEMM backend.

This module provides the same dense FP8 GEMM entry points as DeepGEMM's CUDA
backend.  The HIP kernel accepts OCP FP8 E4M3 operands and UE8M0 scales with
32- or 128-element K granularity and native K/MN-major operands.
The HIP implementation is compiled into
``deep_gemm._C`` by the normal DeepGEMM build flow.

KnowingNothing implementation.
"""
from __future__ import annotations

from typing import List, Optional, Tuple
from pathlib import Path

import torch

__all__ = [
    "ceil_div", "align", "ceil_to_ue8m0", "pack_ue8m0_to_int",
    "per_token_cast_to_fp8", "per_block_cast_to_fp8",
    "get_tma_aligned_size", "get_mn_major_tma_aligned_tensor",
    "get_mn_major_tma_aligned_packed_ue8m0_tensor",
    "get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor",
    "set_mk_alignment_for_contiguous_layout", "get_mk_alignment_for_contiguous_layout",
    "get_theoretical_mk_alignment_for_contiguous_layout",
    "get_m_alignment_for_contiguous_layout", "get_k_alignment_for_contiguous_layout",
    "transform_sf_into_required_layout", "get_plan",
    "fp8_gemm_nt", "fp8_gemm_nn", "fp8_gemm_tn", "fp8_gemm_tt",
    "fp8_fp4_gemm_nt", "fp8_fp4_gemm_nn", "fp8_fp4_gemm_tn", "fp8_fp4_gemm_tt",
    "fp4_gemm_nt",
    "m_grouped_fp8_fp4_gemm_nt_contiguous", "m_grouped_fp8_fp4_gemm_nn_contiguous",
    "m_grouped_fp8_fp4_gemm_nt_masked", "m_grouped_fp4_gemm_nt_contiguous",
    "m_grouped_fp8_gemm_nt_contiguous", "m_grouped_fp8_gemm_nn_contiguous",
    "m_grouped_fp4_gemm_nt_masked", "m_grouped_fp8_gemm_nt_masked",
    "k_grouped_fp4_gemm_nt_contiguous", "k_grouped_fp8_gemm_nt_contiguous",
    "k_grouped_fp8_gemm_tn_contiguous",
]

def _backend():
    # The HIP implementation is part of the normal deep_gemm._C extension.
    from . import _C
    return _C


_backend().cdna4_init(str(Path(__file__).resolve().parent))


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


def ceil_to_ue8m0(x: torch.Tensor) -> torch.Tensor:
    bits = x.abs().float().view(torch.int32)
    exponent = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).bool().to(torch.int32)
    return (exponent.clamp(1, 254) << 23).view(torch.float32)


def pack_ue8m0_to_int(x: torch.Tensor) -> torch.Tensor:
    if x.dtype != torch.float32 or x.dim() == 0 or x.size(-1) % 4:
        raise ValueError("UE8M0 scales must be float32 with a last dimension divisible by four")
    bits = x.contiguous().view(torch.int32)
    if bool((bits < 0).any()) or bool(((bits & 0x7FFFFF) != 0).any()):
        raise ValueError("scales must be positive powers of two")
    return (bits >> 23).to(torch.uint8).view(torch.int32)


def per_token_cast_to_fp8(
    x: torch.Tensor,
    use_ue8m0: bool = True,
    gran_k: int = 128,
    use_packed_ue8m0: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if x.dim() != 2:
        raise ValueError("x must be a 2-D tensor")
    if use_packed_ue8m0 and not use_ue8m0:
        raise ValueError("packed scales require UE8M0")
    m, k = x.shape
    padded_k = align(k, gran_k)
    padded = torch.zeros((m, padded_k), dtype=x.dtype, device=x.device)
    padded[:, :k] = x
    blocks = padded.view(m, padded_k // gran_k, gran_k)
    amax = blocks.abs().float().amax(dim=2).clamp_min(1e-4)
    scales = amax / 448.0
    if use_ue8m0:
        scales = ceil_to_ue8m0(scales)
    fp8 = (blocks * scales.reciprocal().unsqueeze(2)).to(torch.float8_e4m3fn).view(m, padded_k)[:, :k]
    fp8 = fp8.contiguous()
    if use_packed_ue8m0:
        scales = torch.nn.functional.pad(scales, (0, align(scales.size(1), 4) - scales.size(1)), value=1.0)
        return fp8, pack_ue8m0_to_int(scales)
    return fp8, scales


def per_block_cast_to_fp8(
    x: torch.Tensor, use_ue8m0: bool = True, gran_k: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    if x.dim() != 2:
        raise ValueError("x must be a 2-D tensor")
    m, k = x.shape
    padded = torch.zeros((align(m, gran_k), align(k, gran_k)), dtype=x.dtype, device=x.device)
    padded[:m, :k] = x
    blocks = padded.view(-1, gran_k, padded.size(1) // gran_k, gran_k)
    amax = blocks.abs().float().amax(dim=(1, 3), keepdim=True).clamp_min(1e-4)
    scales = amax / 448.0
    if use_ue8m0:
        scales = ceil_to_ue8m0(scales)
    q = (blocks * scales.reciprocal()).to(torch.float8_e4m3fn)
    return q.view_as(padded)[:m, :k].contiguous(), scales.view(blocks.size(0), blocks.size(2))


def _pack_codes_mn_major(codes: torch.Tensor) -> torch.Tensor:
    """Pack [(G,) rows, blocks] UE8M0 codes into int32 words of four K blocks, MN-major per group."""
    batched = codes.dim() == 3
    codes = codes if batched else codes.unsqueeze(0)
    groups, rows, blocks = codes.shape
    padded = torch.zeros((groups, align(rows, 4), align(blocks, 4)), dtype=torch.uint8, device=codes.device)
    padded[:, :rows, :blocks] = codes
    packed = padded.view(torch.int32)
    words, pitch = packed.size(2), align(rows, 4)
    out = torch.empty_strided(
        (groups, rows, words), (pitch * words, 1, pitch), dtype=torch.int32, device=codes.device
    )
    out.copy_(packed[:, :rows])
    return out if batched else out[0]


def _ue8m0_codes(sf: torch.Tensor) -> torch.Tensor:
    return (sf.contiguous().view(torch.int32) >> 23).to(torch.uint8)


def get_tma_aligned_size(x: int, element_size: int) -> int:
    return align(x, 16 // element_size)


def get_mn_major_tma_aligned_tensor(sf: torch.Tensor) -> torch.Tensor:
    """FP32 scales [(G,) mn, k] with MN-major storage and an aligned column pitch."""
    if sf.dtype != torch.float32 or sf.dim() not in (2, 3):
        raise ValueError("sf must be a 2-D or 3-D float32 tensor")
    batched = sf.dim() == 3
    x = sf if batched else sf.unsqueeze(0)
    groups, mn, k = x.shape
    pitch = get_tma_aligned_size(mn, 4)
    out = torch.empty_strided((groups, mn, k), (pitch * k, 1, pitch), dtype=sf.dtype, device=sf.device)
    out.copy_(x)
    return out if batched else out[0]


def get_mn_major_tma_aligned_packed_ue8m0_tensor(
    sf: torch.Tensor, psum_layout: Optional[torch.Tensor] = None
) -> torch.Tensor:
    del psum_layout  # gap rows of a PSUM layout are never read back
    if sf.dtype != torch.float32 or sf.dim() not in (2, 3):
        raise ValueError("sf must be a 2-D or 3-D float32 tensor")
    return _pack_codes_mn_major(_ue8m0_codes(sf))


# Row (M) / K alignment of contiguous grouped layouts. The CDNA4 kernels use 256-row tiles.
_mk_alignment_for_contiguous_layout = 256


def set_mk_alignment_for_contiguous_layout(new_value: int) -> None:
    global _mk_alignment_for_contiguous_layout
    _mk_alignment_for_contiguous_layout = int(new_value)


def get_mk_alignment_for_contiguous_layout() -> int:
    return _mk_alignment_for_contiguous_layout


def get_theoretical_mk_alignment_for_contiguous_layout(expected_m: Optional[int] = None) -> int:
    del expected_m
    return 256


get_m_alignment_for_contiguous_layout = get_mk_alignment_for_contiguous_layout
get_k_alignment_for_contiguous_layout = get_mk_alignment_for_contiguous_layout


def _k_group_sizes(grouped_layout: torch.Tensor, ks_cpu: Optional[List[int]], k_alignment: int,
                   use_psum_layout: bool) -> List[int]:
    """Physical (k_alignment padded) K of each group of a K-grouped layout."""
    if ks_cpu:
        if len(ks_cpu) != grouped_layout.numel():
            raise ValueError("ks_cpu must have one entry per group")
        if not use_psum_layout and any(k % k_alignment for k in ks_cpu):
            raise ValueError("ks_cpu entries must be multiples of the K alignment")
        return [align(k, k_alignment) for k in ks_cpu]
    if not use_psum_layout:
        raise ValueError("ks_cpu is required without the PSUM layout")
    ends, sizes, prev = grouped_layout.tolist(), [], 0
    for end in ends:
        sizes.append(align(end - align(prev, k_alignment), k_alignment))
        prev = end
    return sizes


def get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(
    sf: torch.Tensor, grouped_layout: torch.Tensor, ks_cpu: Optional[List[int]], gran_k: int,
    k_alignment: int, use_psum_layout: bool = False,
) -> torch.Tensor:
    """Pack FP32 scales [sum of per-group K blocks, mn] into [sum of per-group words, mn].

    Each group's K blocks start a new word; unused slots of its last word are zero.
    """
    if sf.dtype != torch.float32 or sf.dim() != 2:
        raise ValueError("sf must be a 2-D float32 tensor")
    if gran_k not in (32, 128) or k_alignment % 128:
        raise ValueError("gran_k must be 32 or 128 and the K alignment a multiple of 128")
    mn = sf.size(1)
    rows = [k // gran_k for k in _k_group_sizes(grouped_layout, ks_cpu, k_alignment, use_psum_layout)]
    if sum(rows) != sf.size(0):
        raise ValueError(f"sf has {sf.size(0)} K blocks, the groups need {sum(rows)}")
    index, base = [], 0
    for r in rows:
        index += [base + i if i < r else sf.size(0) for i in range(align(r, 4))]
        base += r
    if not index:
        return torch.empty((0, mn), dtype=torch.int32, device=sf.device)
    codes = torch.cat([_ue8m0_codes(sf), torch.zeros((1, mn), dtype=torch.uint8, device=sf.device)])
    slots = codes.index_select(0, torch.tensor(index, dtype=torch.long, device=sf.device))
    return slots.view(-1, 4, mn).permute(0, 2, 1).contiguous().view(torch.int32).view(-1, mn)


def transform_sf_into_required_layout(
    sf: torch.Tensor,
    mn: int,
    k: int,
    recipe: Tuple,
    num_groups: Optional[int] = None,
    is_sfa: Optional[bool] = None,
    disable_ue8m0_cast: bool = False,
    psum_layout: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    del psum_layout  # gap rows of a PSUM layout are never read back
    if len(recipe) == 3:
        if is_sfa is None:
            raise ValueError("a 3-element recipe needs is_sfa")
        gran_mn, gran_k = (recipe[0] if is_sfa else recipe[1]), recipe[2]
    elif len(recipe) == 2:
        if is_sfa is not None:
            raise ValueError("a 2-element recipe cannot specify is_sfa")
        gran_mn, gran_k = recipe
    else:
        raise ValueError("recipe must contain two or three elements")
    if not isinstance(gran_mn, int) or gran_mn < 1:
        raise ValueError("MN scale granularity must be a positive integer")
    if gran_k not in (32, 128):
        raise NotImplementedError("CDNA4 FP8 GEMM supports K scale granularity 32 or 128")
    expected_dim = 2 if num_groups is None else 3
    if (sf.dim() != expected_dim or sf.size(-2) != ceil_div(mn, gran_mn) or
            (num_groups is not None and sf.size(0) != num_groups)):
        raise ValueError(f"invalid scale shape {tuple(sf.shape)} for mn={mn}, recipe={recipe}, "
                         f"num_groups={num_groups}")
    if sf.dtype == torch.float32:
        if sf.size(-1) != ceil_div(k, gran_k):
            raise ValueError(f"invalid scale shape {tuple(sf.shape)} for k={k}, recipe={recipe}")
        if disable_ue8m0_cast:
            raise NotImplementedError("CDNA4 FP8 GEMM requires UE8M0 scales")
        if gran_mn != 1:
            sf = sf.index_select(-2, torch.arange(mn, device=sf.device) // gran_mn)
        return _pack_codes_mn_major(_ue8m0_codes(sf))
    if sf.dtype == torch.int32:
        if gran_mn != 1:
            raise ValueError("packed scales require granularity one along MN")
        if sf.size(-1) != ceil_div(k, gran_k * 4):
            raise ValueError(f"invalid packed scale shape {tuple(sf.shape)} for k={k}, recipe={recipe}")
        pitch = sf.stride(-1) if sf.size(-1) > 1 else align(mn, 4)
        if not ((sf.stride(-2) == 1 or mn == 1) and pitch >= align(mn, 4) and pitch % 4 == 0):
            raise ValueError("packed scales must be MN-major with a column stride "
                             "aligned to four elements and at least align(mn, 4)")
        return sf
    raise TypeError(f"unsupported scale dtype {sf.dtype}")


def get_plan(m: int, n: int, k: int) -> Tuple[int, int]:
    return tuple(_backend().cdna4_fp8_gemm_plan(m, n, k))


def _check_fp8(x: torch.Tensor, name: str) -> None:
    if x.dtype != torch.float8_e4m3fn or x.dim() != 2:
        raise TypeError(f"{name} must be a 2-D torch.float8_e4m3fn tensor")
    if x.stride(1) != 1 and x.stride(0) != 1 and min(x.shape) > 1:
        raise ValueError(f"{name} must be K-major or MN-major")


def _operand_layout(x: torch.Tensor, name: str) -> Tuple[bool, int]:
    """Validate native storage; the GEMM reads this tensor without a copy."""
    mn, k = x.shape
    if mn == 1 and (x.stride(1) == 1 or k == 1):
        mn_major, ld = False, align(k, 4)
    elif k == 1 and x.stride(0) == 1:
        mn_major, ld = True, align(mn, 4)
    else:
        mn_major = mn == 1 or (k != 1 and x.stride(1) != 1)
        ld = x.stride(1) if mn_major else x.stride(0)
        row = mn if mn_major else k
        if ld % 4 or ld < row or ld >= 1 << 24:
            raise ValueError(f"{name}: physical row stride must cover the row, "
                             "be divisible by four and be below 1 << 24")
    row = mn if mn_major else k
    if row % 4 and x.data_ptr() % 4:
        raise ValueError(f"{name}: ragged rows require a four-byte aligned base")
    return mn_major, ld


def _transform_sf_pair(sfa, sfb, m, n, k, recipe, recipe_a, recipe_b, groups_a, groups_b,
                       disable_ue8m0_cast):
    """Pack both scale tensors; returns them with their K granularities."""
    if recipe is None and recipe_a is None:
        recipe = (1, 128, 128) if sfb.dtype == torch.float32 else (1, 1, 128)
    if (recipe_a is None) != (recipe_b is None) or (recipe_a is None) == (recipe is None):
        raise ValueError("pass either recipe or both recipe_a and recipe_b")
    if recipe is not None:
        sfa = transform_sf_into_required_layout(sfa, m, k, recipe, groups_a, True, disable_ue8m0_cast)
        sfb = transform_sf_into_required_layout(sfb, n, k, recipe, groups_b, False, disable_ue8m0_cast)
        return sfa, sfb, recipe[2], recipe[2]
    sfa = transform_sf_into_required_layout(sfa, m, k, recipe_a, groups_a, None, disable_ue8m0_cast)
    sfb = transform_sf_into_required_layout(sfb, n, k, recipe_b, groups_b, None, disable_ue8m0_cast)
    return sfa, sfb, recipe_a[1], recipe_b[1]


def fp8_gemm_nt(
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    recipe: Optional[Tuple] = None,
    recipe_a: Optional[Tuple] = None,
    recipe_b: Optional[Tuple] = None,
    compiled_dims: str = "nk",
    disable_ue8m0_cast: bool = False,
    alpha: Optional[float] = None,
    epilogue=None,
) -> None:
    del compiled_dims
    if epilogue is not None:
        raise NotImplementedError("custom epilogues are not supported by the CDNA4 backend")
    (a_fp8, sfa), (b_fp8, sfb) = a, b
    _check_fp8(a_fp8, "a")
    _check_fp8(b_fp8, "b")
    m, k = a_fp8.shape
    n, kb = b_fp8.shape
    if k != kb or d.dim() != 2 or tuple(d.shape) != (m, n):
        raise ValueError(f"shape mismatch: a={tuple(a_fp8.shape)}, b={tuple(b_fp8.shape)}, d={tuple(d.shape)}")
    if d.dtype not in (torch.bfloat16, torch.float32) or (d.stride(1) != 1 and n > 1):
        raise ValueError("d must be a row-major bf16 or fp32 tensor")
    if c is not None:
        if c.dtype != d.dtype or tuple(c.shape) != tuple(d.shape) or (c.stride(1) != 1 and n > 1):
            raise ValueError("c must match d in shape, dtype and layout")
        if c.data_ptr() == d.data_ptr() and c.stride() != d.stride():
            raise ValueError("c aliases d with different strides")
    tensors = (a_fp8, b_fp8, sfa, sfb, d) + ((c,) if c is not None else ())
    if not d.is_cuda or any(t.device != d.device for t in tensors):
        raise ValueError("all GEMM tensors must be on the same accelerator device")
    if m == 0 or n == 0:
        return
    if k == 0:
        if c is None:
            d.zero_()
        elif c.data_ptr() != d.data_ptr():
            d.copy_(c)
        return

    sfa, sfb, gran_k_a, gran_k_b = _transform_sf_pair(
        sfa, sfb, m, n, k, recipe, recipe_a, recipe_b, None, None, disable_ue8m0_cast)

    _operand_layout(a_fp8, "a")
    _operand_layout(b_fp8, "b")
    if c is not None and c.data_ptr() != d.data_ptr() and c.stride() != d.stride():
        d.copy_(c)
        c = d
    _backend().cdna4_fp8_gemm_nt(
        a_fp8, sfa, b_fp8, sfb, d, c, 1.0 if alpha is None else float(alpha), gran_k_a, gran_k_b
    )


def _transpose_operand(x):
    return x[0].transpose(0, 1), x[1].transpose(0, 1)


def fp8_gemm_nn(a, b, d, c=None, recipe=None, recipe_a=None, recipe_b=None, compiled_dims="nk",
                disable_ue8m0_cast=False, alpha=None, epilogue=None):
    return fp8_gemm_nt(a, _transpose_operand(b), d, c, recipe, recipe_a, recipe_b, compiled_dims,
                       disable_ue8m0_cast, alpha, epilogue)


def fp8_gemm_tn(a, b, d, c=None, recipe=None, recipe_a=None, recipe_b=None, compiled_dims="mn",
                disable_ue8m0_cast=False, alpha=None, epilogue=None):
    return fp8_gemm_nt(_transpose_operand(a), _transpose_operand(b), d, c, recipe, recipe_a, recipe_b,
                       compiled_dims, disable_ue8m0_cast, alpha, epilogue)


def fp8_gemm_tt(a, b, d, c=None, recipe=None, recipe_a=None, recipe_b=None, compiled_dims="mn",
                disable_ue8m0_cast=False, alpha=None, epilogue=None):
    return fp8_gemm_nt(_transpose_operand(a), b, d, c, recipe, recipe_a, recipe_b, compiled_dims,
                       disable_ue8m0_cast, alpha, epilogue)


fp8_fp4_gemm_nt = fp8_gemm_nt
fp8_fp4_gemm_nn = fp8_gemm_nn
fp8_fp4_gemm_tn = fp8_gemm_tn
fp8_fp4_gemm_tt = fp8_gemm_tt
fp4_gemm_nt = fp8_gemm_nt


def _check_fp8_3d(x: torch.Tensor, name: str) -> None:
    if x.dtype != torch.float8_e4m3fn or x.dim() != 3:
        raise TypeError(f"{name} must be a 3-D torch.float8_e4m3fn tensor")


def _check_same_device(d: torch.Tensor, *tensors: torch.Tensor) -> None:
    if not d.is_cuda or any(t.device != d.device for t in tensors):
        raise ValueError("all GEMM tensors must be on the same accelerator device")


def m_grouped_fp8_fp4_gemm_nt_contiguous(
    a, b, d, grouped_layout, recipe=None, recipe_a=None, recipe_b=None, compiled_dims="nk",
    disable_ue8m0_cast=False, use_psum_layout=False, ensure_zero_padding=True,
    expected_m_for_psum_layout=None, epilogue=None,
) -> None:
    """D[M, N] = A[M, K] @ B[group, N, K]^T for rows grouped by `grouped_layout`.

    Without the PSUM layout grouped_layout[m] is the group of row m (-1 marks padding rows);
    with it, grouped_layout[g] is the end row of group g, which starts at the M alignment
    of the previous end. Group starts must be multiples of the M alignment.
    """
    del compiled_dims
    if epilogue is not None:
        raise NotImplementedError("custom epilogues are not supported by the CDNA4 backend")
    if expected_m_for_psum_layout is not None and not use_psum_layout:
        raise ValueError("expected_m_for_psum_layout requires the PSUM layout")
    (a_fp8, sfa), (b_fp8, sfb) = a, b
    _check_fp8(a_fp8, "a")
    _check_fp8_3d(b_fp8, "b")
    m, k = a_fp8.shape
    num_groups, n, kb = b_fp8.shape
    if k != kb or d.dim() != 2 or tuple(d.shape) != (m, n):
        raise ValueError(f"shape mismatch: a={tuple(a_fp8.shape)}, b={tuple(b_fp8.shape)}, d={tuple(d.shape)}")
    if d.dtype != torch.bfloat16:
        raise ValueError("d must be bf16")
    _check_same_device(d, a_fp8, b_fp8, sfa, sfb, grouped_layout)
    if m == 0 or n == 0:
        return
    sfa, sfb, gran_k_a, gran_k_b = _transform_sf_pair(
        sfa, sfb, m, n, k, recipe, recipe_a, recipe_b, None, num_groups, disable_ue8m0_cast)
    _backend().cdna4_m_grouped_fp8_gemm_contiguous(
        a_fp8, sfa, b_fp8, sfb, d, grouped_layout, gran_k_a, gran_k_b, use_psum_layout,
        ensure_zero_padding, get_mk_alignment_for_contiguous_layout())


def m_grouped_fp8_fp4_gemm_nn_contiguous(
    a, b, d, grouped_layout, recipe=None, recipe_a=None, recipe_b=None, compiled_dims="nk",
    disable_ue8m0_cast=False, use_psum_layout=False, ensure_zero_padding=True, epilogue=None,
) -> None:
    return m_grouped_fp8_fp4_gemm_nt_contiguous(
        a, (b[0].transpose(1, 2), b[1].transpose(1, 2)), d, grouped_layout, recipe, recipe_a,
        recipe_b, compiled_dims, disable_ue8m0_cast, use_psum_layout, ensure_zero_padding, None,
        epilogue)


def m_grouped_fp8_fp4_gemm_nt_masked(
    a, b, d, masked_m, expected_m, recipe=None, recipe_a=None, recipe_b=None, compiled_dims="nk",
    disable_ue8m0_cast=False, epilogue=None,
) -> None:
    """D[g, :masked_m[g]] = A[g, :masked_m[g]] @ B[g]^T; rows past masked_m[g] are untouched."""
    del compiled_dims
    if epilogue is not None:
        raise NotImplementedError("custom epilogues are not supported by the CDNA4 backend")
    (a_fp8, sfa), (b_fp8, sfb) = a, b
    _check_fp8_3d(a_fp8, "a")
    _check_fp8_3d(b_fp8, "b")
    num_groups, max_m, k = a_fp8.shape
    _, n, _ = b_fp8.shape
    if (tuple(b_fp8.shape) != (num_groups, n, k) or d.dim() != 3 or
            tuple(d.shape) != (num_groups, max_m, n)):
        raise ValueError(f"shape mismatch: a={tuple(a_fp8.shape)}, b={tuple(b_fp8.shape)}, d={tuple(d.shape)}")
    if d.dtype != torch.bfloat16:
        raise ValueError("d must be bf16")
    _check_same_device(d, a_fp8, b_fp8, sfa, sfb, masked_m)
    sfa, sfb, gran_k_a, gran_k_b = _transform_sf_pair(
        sfa, sfb, max_m, n, k, recipe, recipe_a, recipe_b, num_groups, num_groups, disable_ue8m0_cast)
    _backend().cdna4_m_grouped_fp8_gemm_masked(
        a_fp8, sfa, b_fp8, sfb, d, masked_m, int(expected_m), gran_k_a, gran_k_b)


def _k_grouped_scales(sf, grouped_layout, ks_cpu, gran_k, k_alignment, use_psum_layout):
    if sf.dtype == torch.float32:
        return get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(
            sf, grouped_layout, ks_cpu, gran_k, k_alignment, use_psum_layout)
    if sf.dtype == torch.int32:
        if gran_k != 32:
            raise ValueError("pre-packed K-grouped scales require gran_k=32")
        return sf
    raise TypeError(f"unsupported scale dtype {sf.dtype}")


def _k_grouped_gemm(a, b, d, ks_cpu, grouped_layout, c, recipe, use_psum_layout, epilogue, k_major):
    if epilogue is not None:
        raise NotImplementedError("custom epilogues are not supported by the CDNA4 backend")
    if len(recipe) != 3 or recipe[0] != 1 or recipe[1] != 1 or recipe[2] not in (32, 128):
        raise ValueError("K-grouped GEMMs take recipe (1, 1, 32) or (1, 1, 128)")
    gran_k, k_alignment = recipe[2], get_mk_alignment_for_contiguous_layout()
    if k_alignment % 128:
        raise ValueError("the K alignment must be a multiple of 128")
    (a_fp8, sfa), (b_fp8, sfb) = a, b
    if d.dim() != 3 or grouped_layout.numel() != d.size(0):
        raise ValueError("d must be [G, M, N] with one grouped_layout entry per group")
    if ks_cpu and not use_psum_layout:
        if sum(ks_cpu) * d.size(1) != a_fp8.numel() or sum(ks_cpu) * d.size(2) != b_fp8.numel():
            raise ValueError("ks_cpu does not match the operand sizes")
    _check_same_device(d, a_fp8, b_fp8, sfa, sfb, grouped_layout)
    if not use_psum_layout and not ks_cpu:
        raise ValueError("ks_cpu is required without the PSUM layout")
    if k_major:
        # FP32 scales of K-major blocks come as [mn, sum of K blocks]
        sfa = sfa.T if sfa.dtype == torch.float32 else sfa
        sfb = sfb.T if sfb.dtype == torch.float32 else sfb
    if d.size(0) and d.size(1) and d.size(2) and a_fp8.numel():
        sfa = _k_grouped_scales(sfa, grouped_layout, ks_cpu, gran_k, k_alignment, use_psum_layout)
        sfb = _k_grouped_scales(sfb, grouped_layout, ks_cpu, gran_k, k_alignment, use_psum_layout)
    _backend().cdna4_k_grouped_fp8_gemm(
        a_fp8, sfa, b_fp8, sfb, d, c, grouped_layout, use_psum_layout, k_alignment, gran_k, k_major)


def k_grouped_fp8_gemm_tn_contiguous(
    a, b, d, ks_cpu, grouped_layout, c=None, recipe=(1, 1, 128), compiled_dims="mn",
    use_psum_layout=False, epilogue=None,
) -> None:
    """D[g] (+)= A[K range of g, :]^T @ B[K range of g, :] for A [sum_k, M], B [sum_k, N].

    Without the PSUM layout grouped_layout[g] is the padded K of group g; with it, the end K
    of group g, which starts at the K alignment of the previous end.
    """
    del compiled_dims
    _k_grouped_gemm(a, b, d, ks_cpu, grouped_layout, c, tuple(recipe), use_psum_layout, epilogue, False)


def k_grouped_fp8_gemm_nt_contiguous(
    a, b, d, ks_cpu, grouped_layout, c=None, recipe=(1, 1, 128), compiled_dims="mn",
    use_psum_layout=False, epilogue=None,
) -> None:
    """K-grouped GEMM over per-group K-major [M, k_g] / [N, k_g] blocks stored back to back."""
    del compiled_dims
    if use_psum_layout:
        raise ValueError("the K-major K-grouped GEMM does not take the PSUM layout")
    _k_grouped_gemm(a, b, d, ks_cpu, grouped_layout, c, tuple(recipe), False, epilogue, True)


m_grouped_fp4_gemm_nt_contiguous = m_grouped_fp8_fp4_gemm_nt_contiguous
m_grouped_fp8_gemm_nt_contiguous = m_grouped_fp8_fp4_gemm_nt_contiguous
m_grouped_fp8_gemm_nn_contiguous = m_grouped_fp8_fp4_gemm_nn_contiguous
m_grouped_fp4_gemm_nt_masked = m_grouped_fp8_fp4_gemm_nt_masked
m_grouped_fp8_gemm_nt_masked = m_grouped_fp8_fp4_gemm_nt_masked


def k_grouped_fp4_gemm_nt_contiguous(*args, **kwargs):
    raise NotImplementedError("FP4 GEMMs are not supported by the CDNA4 backend yet")
