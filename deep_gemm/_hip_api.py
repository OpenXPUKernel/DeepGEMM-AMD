"""CDNA4 FP8 GEMM backend.

This module provides the same dense FP8 GEMM entry points as DeepGEMM's CUDA
backend.  The HIP kernel accepts OCP FP8 E4M3 operands and UE8M0 scales with
32- or 128-element K granularity and native K/MN-major operands.
The HIP implementation is compiled into
``deep_gemm._C`` by the normal DeepGEMM build flow.

KnowingNothing implementation.
"""
from __future__ import annotations

from typing import Optional, Tuple
from pathlib import Path

import torch

__all__ = [
    "ceil_div", "align", "ceil_to_ue8m0", "pack_ue8m0_to_int",
    "per_token_cast_to_fp8", "per_block_cast_to_fp8",
    "get_mn_major_tma_aligned_packed_ue8m0_tensor",
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
    rows, blocks = codes.shape
    padded = torch.zeros((align(rows, 4), align(blocks, 4)), dtype=torch.uint8, device=codes.device)
    padded[:rows, :blocks] = codes
    packed = padded.view(torch.int32)
    out = torch.empty_strided(
        (rows, packed.size(1)), (1, align(rows, 4)), dtype=torch.int32, device=codes.device
    )
    return out.copy_(packed[:rows])


def get_mn_major_tma_aligned_packed_ue8m0_tensor(sf: torch.Tensor) -> torch.Tensor:
    if sf.dtype != torch.float32 or sf.dim() != 2:
        raise ValueError("sf must be a 2-D float32 tensor")
    return _pack_codes_mn_major((sf.contiguous().view(torch.int32) >> 23).to(torch.uint8))


def transform_sf_into_required_layout(
    sf: torch.Tensor,
    mn: int,
    k: int,
    recipe: Tuple,
    num_groups: Optional[int] = None,
    is_sfa: Optional[bool] = None,
    disable_ue8m0_cast: bool = False,
) -> torch.Tensor:
    if num_groups is not None:
        raise NotImplementedError("grouped GEMMs are not supported by the CDNA4 backend")
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
    if sf.dim() != 2 or sf.size(0) != ceil_div(mn, gran_mn):
        raise ValueError(f"invalid scale shape {tuple(sf.shape)} for mn={mn}, recipe={recipe}")
    if sf.dtype == torch.float32:
        if sf.size(1) != ceil_div(k, gran_k):
            raise ValueError(f"invalid scale shape {tuple(sf.shape)} for k={k}, recipe={recipe}")
        if disable_ue8m0_cast:
            raise NotImplementedError("CDNA4 FP8 GEMM requires UE8M0 scales")
        if gran_mn != 1:
            sf = sf.index_select(0, torch.arange(mn, device=sf.device) // gran_mn)
        codes = (sf.contiguous().view(torch.int32) >> 23).to(torch.uint8)
        return _pack_codes_mn_major(codes)
    if sf.dtype == torch.int32:
        if gran_mn != 1:
            raise ValueError("packed scales require granularity one along MN")
        if sf.size(1) != ceil_div(k, gran_k * 4):
            raise ValueError(f"invalid packed scale shape {tuple(sf.shape)} for k={k}, recipe={recipe}")
        if not ((sf.stride(0) == 1 or mn == 1) and
                sf.stride(1) >= align(mn, 4) and sf.stride(1) % 4 == 0):
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

    if recipe is None and recipe_a is None:
        recipe = (1, 128, 128) if sfb.dtype == torch.float32 else (1, 1, 128)
    if (recipe_a is None) != (recipe_b is None) or (recipe_a is None) == (recipe is None):
        raise ValueError("pass either recipe or both recipe_a and recipe_b")
    if recipe is not None:
        sfa = transform_sf_into_required_layout(sfa, m, k, recipe, is_sfa=True, disable_ue8m0_cast=disable_ue8m0_cast)
        sfb = transform_sf_into_required_layout(sfb, n, k, recipe, is_sfa=False, disable_ue8m0_cast=disable_ue8m0_cast)
        gran_k_a = gran_k_b = recipe[2]
    else:
        sfa = transform_sf_into_required_layout(sfa, m, k, recipe_a, disable_ue8m0_cast=disable_ue8m0_cast)
        sfb = transform_sf_into_required_layout(sfb, n, k, recipe_b, disable_ue8m0_cast=disable_ue8m0_cast)
        gran_k_a, gran_k_b = recipe_a[1], recipe_b[1]

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


def _grouped_unavailable(*args, **kwargs):
    raise NotImplementedError("grouped GEMMs are not supported by the CDNA4 backend")


m_grouped_fp8_fp4_gemm_nt_contiguous = _grouped_unavailable
m_grouped_fp8_fp4_gemm_nn_contiguous = _grouped_unavailable
m_grouped_fp8_fp4_gemm_nt_masked = _grouped_unavailable
m_grouped_fp4_gemm_nt_contiguous = _grouped_unavailable
m_grouped_fp8_gemm_nt_contiguous = _grouped_unavailable
m_grouped_fp8_gemm_nn_contiguous = _grouped_unavailable
m_grouped_fp4_gemm_nt_masked = _grouped_unavailable
m_grouped_fp8_gemm_nt_masked = _grouped_unavailable
k_grouped_fp4_gemm_nt_contiguous = _grouped_unavailable
k_grouped_fp8_gemm_nt_contiguous = _grouped_unavailable
k_grouped_fp8_gemm_tn_contiguous = _grouped_unavailable
