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
    "bf16_gemm_nt", "bf16_gemm_nn", "bf16_gemm_tn", "bf16_gemm_tt",
    "m_grouped_bf16_gemm_nt_contiguous", "m_grouped_bf16_gemm_nn_contiguous",
    "m_grouped_bf16_gemm_nt_masked", "k_grouped_bf16_gemm_tn_contiguous",
    "shutdown", "set_num_sms", "get_num_sms", "set_tc_util", "get_tc_util", "set_ignore_compile_dims",
    "set_block_size_multiple_of", "set_pdl", "get_pdl", "use_deterministic_algorithms",
    "get_mqa_logits_metadata", "fp8_fp4_mqa_logits", "get_paged_mqa_logits_metadata", "fp8_fp4_paged_mqa_logits",
    "get_sparse_mqa_logits_metadata", "get_paged_sparse_mqa_logits_metadata", "fp8_fp4_sparse_mqa_logits",
    "fp8_fp4_paged_sparse_mqa_logits", "einsum", "fp8_einsum",
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


def _is_fp4(x: torch.Tensor) -> bool:
    """Packed FP4 (E2M1) operands are int8 tensors with two values per byte, the lower index in the
    low nibble."""
    return x.dtype in (torch.int8, torch.uint8)


def _check_fp8_fp4(x: torch.Tensor, name: str, dim: int = 2) -> None:
    if x.dim() != dim or (x.dtype != torch.float8_e4m3fn and not _is_fp4(x)):
        raise TypeError(f"{name} must be a {dim}-D torch.float8_e4m3fn or packed FP4 (torch.int8) tensor")
    if x.stride(-1) != 1 and x.stride(-2) != 1 and min(x.shape[-2:]) > 1:
        raise ValueError(f"{name} must be K-major or MN-major")


def _logical_mn_k(x: torch.Tensor) -> Tuple[int, int]:
    """(MN, K) of an FP8 or packed FP4 operand [..., MN, K]: FP4 packs two values per byte along its
    contiguous dimension (K, or MN when MN-major)"""
    mn, k = x.shape[-2:]
    if not _is_fp4(x):
        return mn, k
    return (2 * mn, k) if k > 1 and x.stride(-1) != 1 else (mn, 2 * k)


def _logical_k(x: torch.Tensor) -> int:
    return _logical_mn_k(x)[1]


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
    _check_fp8_fp4(a_fp8, "a")
    _check_fp8_fp4(b_fp8, "b")
    m, k = _logical_mn_k(a_fp8)
    n, kb = _logical_mn_k(b_fp8)
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
    _check_fp8_fp4(a_fp8, "a")
    _check_fp8_fp4(b_fp8, "b", 3)
    m, k = a_fp8.size(0), _logical_k(a_fp8)
    num_groups, (n, kb) = b_fp8.size(0), _logical_mn_k(b_fp8)
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
        ensure_zero_padding, get_mk_alignment_for_contiguous_layout(),
        int(expected_m_for_psum_layout or 0))


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
    _check_fp8_fp4(a_fp8, "a", 3)
    _check_fp8_fp4(b_fp8, "b", 3)
    num_groups, max_m, k = a_fp8.size(0), a_fp8.size(1), _logical_k(a_fp8)
    n = _logical_mn_k(b_fp8)[0]
    if (b_fp8.size(0) != num_groups or _logical_k(b_fp8) != k or d.dim() != 3 or
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
    fp4 = _is_fp4(a_fp8)
    if d.dim() != 3 or grouped_layout.numel() != d.size(0):
        raise ValueError("d must be [G, M, N] with one grouped_layout entry per group")
    if ks_cpu and not use_psum_layout:
        values_per_element = 2 if fp4 else 1
        if (sum(ks_cpu) * d.size(1) != a_fp8.numel() * values_per_element or
                sum(ks_cpu) * d.size(2) != b_fp8.numel() * values_per_element):
            raise ValueError("ks_cpu does not match the operand sizes")
    _check_same_device(d, a_fp8, b_fp8, sfa, sfb, grouped_layout)
    if not use_psum_layout and not ks_cpu:
        raise ValueError("ks_cpu is required without the PSUM layout")
    if k_major and not fp4:
        # FP32 scales of K-major FP8 blocks come as [mn, sum of K blocks] (FP4: [sum of K blocks, mn])
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


def _check_bf16(x: torch.Tensor, name: str, dim: int) -> None:
    if x.dtype != torch.bfloat16 or x.dim() != dim:
        raise TypeError(f"{name} must be a {dim}-D torch.bfloat16 tensor")


def bf16_gemm_nt(a, b, d, c=None, compiled_dims="nk", alpha=None, epilogue=None) -> None:
    """D = alpha * (A @ B.T) + C for BF16 A [M, K] and B [N, K] (K- or MN-major)."""
    del compiled_dims
    if epilogue is not None:
        raise NotImplementedError("custom epilogues are not supported by the CDNA4 backend")
    _check_bf16(a, "a", 2)
    _check_bf16(b, "b", 2)
    m, k = a.shape
    n, kb = b.shape
    if k != kb or d.dim() != 2 or tuple(d.shape) != (m, n):
        raise ValueError(f"shape mismatch: a={tuple(a.shape)}, b={tuple(b.shape)}, d={tuple(d.shape)}")
    if d.dtype not in (torch.bfloat16, torch.float32) or (d.stride(1) != 1 and n > 1):
        raise ValueError("d must be a row-major bf16 or fp32 tensor")
    if c is not None:
        if c.dtype != d.dtype or tuple(c.shape) != tuple(d.shape) or (c.stride(1) != 1 and n > 1):
            raise ValueError("c must match d in shape, dtype and layout")
        if c.data_ptr() == d.data_ptr() and c.stride() != d.stride():
            raise ValueError("c aliases d with different strides")
    _check_same_device(d, a, b, *((c,) if c is not None else ()))
    if m == 0 or n == 0:
        return
    if k == 0:
        if c is None:
            d.zero_()
        elif c.data_ptr() != d.data_ptr():
            d.copy_(c)
        return
    if c is not None and c.data_ptr() != d.data_ptr() and c.stride() != d.stride():
        d.copy_(c)
        c = d
    _backend().cdna4_fp8_gemm_nt(a, None, b, None, d, c, 1.0 if alpha is None else float(alpha), 0, 0)


def bf16_gemm_nn(a, b, d, c=None, compiled_dims="nk", alpha=None, epilogue=None):
    return bf16_gemm_nt(a, b.transpose(0, 1), d, c, compiled_dims, alpha, epilogue)


def bf16_gemm_tn(a, b, d, c=None, compiled_dims="mn", alpha=None, epilogue=None):
    return bf16_gemm_nt(a.transpose(0, 1), b.transpose(0, 1), d, c, compiled_dims, alpha, epilogue)


def bf16_gemm_tt(a, b, d, c=None, compiled_dims="mn", alpha=None, epilogue=None):
    return bf16_gemm_nt(a.transpose(0, 1), b, d, c, compiled_dims, alpha, epilogue)


def m_grouped_bf16_gemm_nt_contiguous(a, b, d, grouped_layout, compiled_dims="nk", use_psum_layout=False,
                                      ensure_zero_padding=True, expected_m_for_psum_layout=None,
                                      epilogue=None) -> None:
    """BF16 version of m_grouped_fp8_gemm_nt_contiguous (A [M, K], B [G, N, K], D [M, N] BF16)."""
    del compiled_dims
    if epilogue is not None:
        raise NotImplementedError("custom epilogues are not supported by the CDNA4 backend")
    if expected_m_for_psum_layout is not None and not use_psum_layout:
        raise ValueError("expected_m_for_psum_layout requires the PSUM layout")
    _check_bf16(a, "a", 2)
    _check_bf16(b, "b", 3)
    m, k = a.shape
    num_groups, n, kb = b.shape
    if k != kb or d.dim() != 2 or tuple(d.shape) != (m, n) or d.dtype != torch.bfloat16:
        raise ValueError(f"shape mismatch: a={tuple(a.shape)}, b={tuple(b.shape)}, d={tuple(d.shape)} (bf16)")
    _check_same_device(d, a, b, grouped_layout)
    if m == 0 or n == 0:
        return
    _backend().cdna4_m_grouped_fp8_gemm_contiguous(
        a, None, b, None, d, grouped_layout, 0, 0, use_psum_layout, ensure_zero_padding,
        get_mk_alignment_for_contiguous_layout(), int(expected_m_for_psum_layout or 0))


def m_grouped_bf16_gemm_nn_contiguous(a, b, d, grouped_layout, compiled_dims="nk", use_psum_layout=False,
                                      ensure_zero_padding=True, epilogue=None) -> None:
    return m_grouped_bf16_gemm_nt_contiguous(a, b.transpose(1, 2), d, grouped_layout, compiled_dims,
                                             use_psum_layout, ensure_zero_padding, None, epilogue)


def m_grouped_bf16_gemm_nt_masked(a, b, d, masked_m, expected_m, compiled_dims="nk", epilogue=None) -> None:
    """BF16 version of m_grouped_fp8_gemm_nt_masked."""
    del compiled_dims
    if epilogue is not None:
        raise NotImplementedError("custom epilogues are not supported by the CDNA4 backend")
    _check_bf16(a, "a", 3)
    _check_bf16(b, "b", 3)
    num_groups, max_m, k = a.shape
    _, n, _ = b.shape
    if (tuple(b.shape) != (num_groups, n, k) or d.dim() != 3 or tuple(d.shape) != (num_groups, max_m, n)
            or d.dtype != torch.bfloat16):
        raise ValueError(f"shape mismatch: a={tuple(a.shape)}, b={tuple(b.shape)}, d={tuple(d.shape)} (bf16)")
    _check_same_device(d, a, b, masked_m)
    _backend().cdna4_m_grouped_fp8_gemm_masked(a, None, b, None, d, masked_m, int(expected_m), 0, 0)


def k_grouped_bf16_gemm_tn_contiguous(a, b, d, ks_cpu, grouped_layout, c=None, compiled_dims="mn",
                                      use_psum_layout=False, epilogue=None) -> None:
    """D[g] (+)= A[K range of g, :]^T @ B[K range of g, :] for BF16 A [sum_k, M], B [sum_k, N]."""
    del compiled_dims
    if epilogue is not None:
        raise NotImplementedError("custom epilogues are not supported by the CDNA4 backend")
    _check_bf16(a, "a", 2)
    _check_bf16(b, "b", 2)
    k_alignment = get_mk_alignment_for_contiguous_layout()
    if k_alignment % 128:
        raise ValueError("the K alignment must be a multiple of 128")
    if d.dim() != 3 or grouped_layout.numel() != d.size(0):
        raise ValueError("d must be [G, M, N] with one grouped_layout entry per group")
    if not use_psum_layout and not ks_cpu:
        raise ValueError("ks_cpu is required without the PSUM layout")
    if ks_cpu and not use_psum_layout and (sum(ks_cpu) != a.size(0) or any(k % k_alignment for k in ks_cpu)):
        raise ValueError("ks_cpu must be K-aligned and add up to the rows of a")
    _check_same_device(d, a, b, grouped_layout, *((c,) if c is not None else ()))
    _backend().cdna4_k_grouped_fp8_gemm(a, None, b, None, d, c, grouped_layout, use_psum_layout, k_alignment, 0,
                                        False)


m_grouped_fp4_gemm_nt_contiguous = m_grouped_fp8_fp4_gemm_nt_contiguous
m_grouped_fp8_gemm_nt_contiguous = m_grouped_fp8_fp4_gemm_nt_contiguous
m_grouped_fp8_gemm_nn_contiguous = m_grouped_fp8_fp4_gemm_nn_contiguous
m_grouped_fp4_gemm_nt_masked = m_grouped_fp8_fp4_gemm_nt_masked
m_grouped_fp8_gemm_nt_masked = m_grouped_fp8_fp4_gemm_nt_masked


def k_grouped_fp4_gemm_nt_contiguous(
    a, b, d, ks_cpu, grouped_layout, c=None, recipe=(1, 1, 32), compiled_dims="mn",
    use_psum_layout=False, epilogue=None,
) -> None:
    """K-grouped FP4 GEMM over packed K-major A [M, sum_k / 2], B [N, sum_k / 2]: group g takes the
    K columns of its range; without the PSUM layout grouped_layout[g] is its padded K, with it its
    end K."""
    del compiled_dims
    if tuple(recipe) != (1, 1, 32):
        raise ValueError("FP4 K-grouped GEMMs take recipe (1, 1, 32)")
    if not (_is_fp4(a[0]) and _is_fp4(b[0]) and a[0].dim() == 2 and b[0].dim() == 2):
        raise TypeError("a and b must be 2-D packed FP4 (torch.int8) tensors")
    _k_grouped_gemm(a, b, d, ks_cpu, grouped_layout, c, tuple(recipe), use_psum_layout, epilogue, True)


# ---- Einsum: the expressions of the CUDA backend as strided batched GEMMs (no copies)
_full_rows_cache = {}


def _full_rows(groups: int, rows: int, device) -> torch.Tensor:
    """masked_m of a batched GEMM whose groups are full: cached, so that calls launch one kernel
    (create it before graph capture)."""
    key = (str(device), groups, rows)
    if key not in _full_rows_cache:
        _full_rows_cache[key] = torch.full((groups, ), rows, dtype=torch.int32, device=device)
    return _full_rows_cache[key]


def _batched_gemm(a, sfa, b, sfb, d, c, gran_k_a=0, gran_k_b=0):
    """d[g] (+)= a[g] @ b[g]^T over [G, M, K] / [G, N, K] / [G, M, N] views."""
    groups, rows = d.size(0), d.size(1)
    _backend().cdna4_m_grouped_fp8_gemm_masked(a, sfa, b, sfb, d, _full_rows(groups, rows, d.device), max(rows, 1),
                                               gran_k_a, gran_k_b, c=c, alpha=1.0)


def _einsum_views(expr, a, b, d, c):
    """(A [G, M, K], B [G, N, K], D [G, M, N], C) views of a batched expression"""
    perm = lambda t, *dims: None if t is None else t.permute(*dims)
    if expr == "bhr,hdr->bhd":
        return a.permute(1, 0, 2), b, d.permute(1, 0, 2), perm(c, 1, 0, 2)
    if expr == "bhd,hdr->bhr":
        return a.permute(1, 0, 2), b.permute(0, 2, 1), d.permute(1, 0, 2), perm(c, 1, 0, 2)
    if expr == "bhd,bhr->hdr":
        return a.permute(1, 2, 0), b.permute(1, 2, 0), d, c
    raise NotImplementedError(f"unsupported einsum expression: {expr}")


def einsum(expr, a, b, d, c=None, epilogue=None) -> None:
    """The BF16 einsum expressions of the CUDA backend: "bmk,bnk->mn" (D (+)= sum over b),
    "bhr,hdr->bhd", "bhd,hdr->bhr" and "bhd,bhr->hdr"."""
    if epilogue is not None:
        raise NotImplementedError("custom epilogues are not supported by the CDNA4 backend")
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16 or d.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("einsum takes BF16 a and b, and a BF16 or FP32 d")
    if c is not None and c.dtype != d.dtype:
        raise TypeError("c must have the dtype of d")
    _check_same_device(d, a, b, *((c,) if c is not None else ()))
    if expr == "bmk,bnk->mn":
        _backend().cdna4_k_batched_gemm(a, b, d, c)
        return
    av, bv, dv, cv = _einsum_views(expr, a, b, d, c)
    _batched_gemm(av, None, bv, None, dv, cv)


def fp8_einsum(expr, a, b, d, c=None, recipe=(1, 128, 128), epilogue=None) -> None:
    """The FP8 einsum expressions of the CUDA backend ("bhr,hdr->bhd", "bhd,hdr->bhr" and
    "bhd,bhr->hdr") with BF16 or FP32 outputs; a and b are (data, scales) pairs."""
    if epilogue is not None or isinstance(d, (tuple, list)):
        raise NotImplementedError("FP8 outputs and custom epilogues are not supported by the CDNA4 backend")
    (a_fp8, sfa), (b_fp8, sfb) = a, b
    _check_same_device(d, a_fp8, b_fp8, sfa, sfb, *((c,) if c is not None else ()))
    av, bv, dv, cv = _einsum_views(expr, a_fp8, b_fp8, d, c)
    # the scale tensors follow the permutation of their data
    sfa_v, sfb_v, _, _ = _einsum_views(expr, sfa, sfb, d, None)
    groups, m, k = av.size(0), av.size(1), av.size(2)
    sfa_p, sfb_p, gran_k_a, gran_k_b = _transform_sf_pair(sfa_v, sfb_v, m, bv.size(1), k, tuple(recipe), None, None,
                                                          groups, groups, False)
    _batched_gemm(av, sfa_p, bv, sfb_p, dv, cv, gran_k_a, gran_k_b)


# ---- Runtime configuration (CUDA API parity). The CDNA4 kernels size their grids from the compute
# units, or from set_num_sms; the remaining switches have no CDNA4 counterpart and are recorded only.
_config = {"num_sms": None, "tc_util": 100, "pdl": False, "deterministic": False}


def shutdown() -> None:
    pass


def set_num_sms(num_sms: int) -> None:
    if num_sms <= 0:
        raise ValueError("the number of compute units must be positive")
    _config["num_sms"] = int(num_sms)


def get_num_sms() -> int:
    if _config["num_sms"] is None:
        return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    return _config["num_sms"]


def set_tc_util(tc_util: int) -> None:
    _config["tc_util"] = int(tc_util)


def get_tc_util() -> int:
    return _config["tc_util"]


def set_ignore_compile_dims(ignore: bool) -> None:
    del ignore


def set_block_size_multiple_of(multiple: int) -> None:
    del multiple


def set_pdl(enable: bool) -> None:
    _config["pdl"] = bool(enable)


def get_pdl() -> bool:
    return _config["pdl"]


def use_deterministic_algorithms(enable: bool = True) -> None:
    _config["deterministic"] = bool(enable)


# ---- MQA logits (the V3.2 indexer). FP8 Q (no scales) takes E4M3 KV with one float scale per
# token, float weights and gives float logits; MXFP8 / packed MXFP4 Q and KV carry UE8M0 scales of
# 32 values packed into one int32 per head / token, with BF16 weights and logits.
_MQA_CONTIGUOUS, _MQA_PAGED, _MQA_SPARSE, _MQA_PAGED_SPARSE = 0, 1, 2, 3


def _mqa_q(q, num_dims):
    q_fp, q_sf = q
    if q_fp.dim() != num_dims or not q_fp.is_contiguous():
        raise ValueError(f"q must be a contiguous {num_dims}-D tensor")
    if q_sf is None and q_fp.dtype != torch.float8_e4m3fn:
        raise TypeError("Q without scales must be torch.float8_e4m3fn")
    if q_sf is not None and (q_sf.dtype != torch.int32 or tuple(q_sf.shape) != tuple(q_fp.shape[:-1])):
        raise ValueError("q scales must be packed int32, one per token and head")
    head_dim = q_fp.size(-1) * (2 if _is_fp4(q_fp) else 1)
    return q_fp, q_sf, head_dim


def _mqa_logits_buffer(rows, columns, q_sf, device):
    dtype = torch.float32 if q_sf is None else torch.bfloat16
    storage = torch.empty((rows, align(max(columns, 1), 256)), device=device, dtype=dtype)
    return storage[:, :columns]


def get_mqa_logits_metadata(cu_seq_len_k_start, cu_seq_len_k_end, num_kv_tokens, num_heads):
    """KV ranges of the groups of tokens the contiguous MQA logits kernel gives each wave."""
    del num_kv_tokens
    tokens = _backend().cdna4_mqa_logits_tokens_per_wave(int(num_heads))
    num_q = cu_seq_len_k_start.numel()
    groups = ceil_div(num_q, tokens)
    pad = groups * tokens - num_q
    ks = torch.nn.functional.pad(cu_seq_len_k_start, (0, pad), value=2**31 - 1).view(groups, tokens)
    ke = torch.nn.functional.pad(cu_seq_len_k_end, (0, pad), value=-2**31).view(groups, tokens)
    return torch.stack((ks.amin(1), ke.amax(1)), dim=1).to(torch.int32).contiguous()


def fp8_fp4_mqa_logits(q, kv, weights, cu_seq_len_k_start, cu_seq_len_k_end, max_seqlen_k, schedule_meta=None):
    """logits[i, j] = sum_h weights[i, h] relu(q[i, h] . kv[ks[i] + j]) for j < ke[i] - ks[i]
    (times the KV scale with FP8 Q); entries past a row's span are not written."""
    q_fp, q_sf, _ = _mqa_q(q, 3)
    kv_fp, kv_sf = kv
    if max_seqlen_k <= 0:
        raise ValueError("max_seqlen_k must be positive")
    if kv_fp.dim() != 2 or not kv_fp.is_contiguous() or kv_fp.dtype != q_fp.dtype or kv_fp.size(1) != q_fp.size(2):
        raise ValueError("kv must be a contiguous [seq_len_kv, D] tensor of the dtype of q")
    if kv_sf.dtype != (torch.float32 if q_sf is None else torch.int32) or kv_sf.numel() != kv_fp.size(0):
        raise ValueError("kv scales must hold one float (FP8) or packed int32 (MX) per token")
    logits = _mqa_logits_buffer(q_fp.size(0), int(max_seqlen_k), q_sf, q_fp.device)
    _backend().cdna4_mqa_logits(q_fp, q_sf, kv_fp, kv_sf.contiguous(), weights, logits, _MQA_CONTIGUOUS,
                                ks=cu_seq_len_k_start, ke=cu_seq_len_k_end, group_ranges=schedule_meta)
    return logits


def get_paged_mqa_logits_metadata(context_lens, block_kv, num_sms, indices=None):
    """With `indices` (the request of every token, one token per row): the first token of each
    token's request. Without, requests are the rows of context_lens and need no metadata."""
    del block_kv, num_sms
    if indices is None:
        return torch.zeros((1, 2), device=context_lens.device, dtype=torch.int32)
    num_q = indices.numel()
    first = torch.ones(num_q, device=indices.device, dtype=torch.bool)
    first[1:] = indices[1:] != indices[:-1]
    positions = torch.where(first, torch.arange(num_q, device=indices.device), 0)
    return torch.cummax(positions, dim=0).values.to(torch.int32)


def fp8_fp4_paged_mqa_logits(q, kv_cache, weights, context_lens, block_table, schedule_meta, max_context_len,
                             indices=None):
    """Paged MQA logits: q [batch, next_n, H, D], kv_cache [blocks, block_kv, 1, D + 4] (a block holds
    block_kv rows, then their scales), token (i, j) takes positions [0, context_lens[i, j]) of
    block_table row i (with `indices`: rows are tokens, next_n = 1)."""
    q_fp, q_sf, head_dim = _mqa_q(q, 4)
    batch, next_n, num_heads = q_fp.shape[:3]
    if kv_cache.dim() != 4 or kv_cache.dtype != torch.uint8 or kv_cache.size(2) != 1:
        raise ValueError("kv_cache must be a [blocks, block_kv, 1, D + 4] uint8 tensor")
    row_bytes = head_dim // 2 if _is_fp4(q_fp) else head_dim
    if kv_cache.size(3) != row_bytes + 4 or kv_cache.stride(3) != 1:
        raise ValueError("kv_cache rows must hold D values and a 4-byte scale")
    if indices is not None and next_n != 1:
        raise ValueError("varlen requests take one token per row")
    if tuple(context_lens.shape) != (batch, next_n) or block_table.size(0) != batch or block_table.stride(1) != 1:
        raise ValueError("context_lens must be [batch, next_n] and block_table [batch, max_blocks]")
    logits = _mqa_logits_buffer(batch * next_n, int(max_context_len), q_sf, q_fp.device)
    _backend().cdna4_mqa_logits(q_fp.view(batch * next_n, num_heads, -1), q_sf, kv_cache, None, weights, logits,
                                _MQA_PAGED, context_lens=context_lens.contiguous(), block_table=block_table,
                                run_starts=schedule_meta if indices is not None else None, next_n=next_n,
                                page_kv=kv_cache.size(1))
    return logits


def get_sparse_mqa_logits_metadata(cu_seq_len_k_start, cu_seq_len_k_end, num_kv_tokens, sparse_kv_block_indices,
                                   qk_dtype, sparse_block_kv, use_unaligned_ks=False):
    """The tokens' KV ranges and the first KV row of each selected block (block i of a token starts
    at i * sparse_block_kv, plus ks % sparse_block_kv with unaligned ks)."""
    del num_kv_tokens, qk_dtype
    starts = sparse_kv_block_indices * sparse_block_kv
    if use_unaligned_ks:
        starts = starts + (cu_seq_len_k_start % sparse_block_kv)[:, None]
    return torch.cat((cu_seq_len_k_start.int(), cu_seq_len_k_end.int(), starts.int().flatten()))


def get_paged_sparse_mqa_logits_metadata(context_lens, block_table, indices, page_kv, sparse_kv_block_indices,
                                         qk_dtype, sparse_block_kv):
    """The tokens' context lengths and the physical row (page * page_kv + row) of each selected block."""
    del indices, qk_dtype
    positions = sparse_kv_block_indices.long() * sparse_block_kv
    pages = block_table.long().gather(1, (positions // page_kv).clamp(max=block_table.size(1) - 1))
    lens = context_lens.reshape(-1).int()
    return torch.cat((lens, lens, (pages * page_kv + positions % page_kv).int().flatten()))


def _sparse_logits(q_fp, q_sf, kv, kv_sf, weights, metadata, num_max_sparse_blocks, sparse_block_kv, source):
    num_q = q_fp.size(0)
    if metadata.numel() != num_q * (2 + num_max_sparse_blocks):
        raise ValueError("the sparse metadata does not match q and num_max_sparse_blocks")
    logits = _mqa_logits_buffer(num_q, num_max_sparse_blocks * sparse_block_kv, q_sf, q_fp.device)
    paged = source == _MQA_PAGED_SPARSE
    _backend().cdna4_mqa_logits(q_fp, q_sf, kv, kv_sf, weights, logits, source,
                                ks=None if paged else metadata[:num_q], ke=None if paged else metadata[num_q:2 * num_q],
                                context_lens=metadata[:num_q] if paged else None,
                                page_kv=kv.size(1) if paged else 0, sparse_starts=metadata[2 * num_q:],
                                num_max_sparse_blocks=num_max_sparse_blocks, sparse_block_kv=sparse_block_kv)
    return logits


def fp8_fp4_sparse_mqa_logits(q, kv, weights, metadata, num_max_sparse_blocks, sparse_block_kv,
                              use_unaligned_ks=False):
    """logits[i, b * sparse_block_kv + t]: token t of the b-th selected block of token i."""
    del use_unaligned_ks  # folded into the metadata
    q_fp, q_sf, _ = _mqa_q(q, 3)
    kv_fp, kv_sf = kv
    return _sparse_logits(q_fp, q_sf, kv_fp, kv_sf.contiguous(), weights, metadata, num_max_sparse_blocks,
                          sparse_block_kv, _MQA_SPARSE)


def fp8_fp4_paged_sparse_mqa_logits(q, kv_cache, weights, metadata, num_max_sparse_blocks, sparse_block_kv):
    q_fp, q_sf, _ = _mqa_q(q, 4)
    return _sparse_logits(q_fp.view(q_fp.size(0) * q_fp.size(1), q_fp.size(2), -1), q_sf.view(-1, q_sf.size(-1))
                          if q_sf is not None else None, kv_cache, None, weights, metadata, num_max_sparse_blocks,
                          sparse_block_kv, _MQA_PAGED_SPARSE)
