"""Mega MoE on CDNA4 with the CUDA backend's interface: expert-parallel MoE (dispatch, the two
expert GEMMs with SwiGLU, combine, and shared experts) across the GPUs of a node, on uni_symm
symmetric buffers. See mega_moe.cuh for the kernels."""
from __future__ import annotations

import math
import warnings
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import uni_symm

from . import _C

__all__ = [
    "SymmBuffer", "get_symm_buffer_for_mega_moe", "get_token_alignment_for_mega_moe", "get_block_m_for_mega_moe",
    "transform_weights_for_mega_moe", "fp8_fp4_mega_moe", "bf16_mega_moe",
]

# Expert rows start at multiples of the grouped GEMMs' 256-row tiles
_EXPERT_ALIGNMENT = 256


def get_token_alignment_for_mega_moe() -> int:
    return 4


def get_block_m_for_mega_moe(num_ranks: int, num_experts: int, num_max_tokens_per_rank: int, num_tokens: int,
                             num_topk: int, mma_type: str = "fp8xfp4") -> int:
    del num_ranks, num_experts, num_max_tokens_per_rank, num_tokens, num_topk, mma_type
    return _EXPERT_ALIGNMENT


class SymmBuffer:
    """The symmetric buffer of a Mega MoE configuration on every rank of `group`, with the views the
    caller fills before each call (x, x_sf, topk_idx, topk_weights) and the kernels' activations."""

    def __init__(self, group: dist.ProcessGroup, num_experts: int, num_max_tokens_per_rank: int, num_topk: int,
                 hidden: int, intermediate_hidden: int, num_shared_experts: int = 0, mma_type: str = "fp8xfp4",
                 activation: str = "swiglu", base: Optional["SymmBuffer"] = None):
        if activation != "swiglu":
            raise ValueError(f"only the swiglu activation is supported, got {activation}")
        if mma_type not in ("fp8xfp4", "fp8xfp8", "bf16xbf16"):
            raise ValueError(f"unsupported mma_type {mma_type}")
        num_max_tokens_per_rank = -(-num_max_tokens_per_rank // get_token_alignment_for_mega_moe()) * \
            get_token_alignment_for_mega_moe()
        self.group = group
        self.num_experts, self.num_max_tokens_per_rank, self.num_topk = num_experts, num_max_tokens_per_rank, num_topk
        self.hidden, self.intermediate_hidden = hidden, intermediate_hidden
        self.num_shared_experts, self.mma_type, self.activation = num_shared_experts, mma_type, activation
        with_sf = mma_type != "bf16xbf16"
        layout = _C.cdna4_mega_moe_layout(group.size(), num_experts, num_max_tokens_per_rank, num_topk, hidden,
                                          intermediate_hidden, intermediate_hidden * num_shared_experts, with_sf)
        self.layout = layout
        if base is None:
            self.buffer = uni_symm.empty(layout["bytes"], dtype=torch.int8, device="cuda")
            self.handle = uni_symm.rendezvous(self.buffer, group=group)
            self.group.barrier()
            torch.cuda.synchronize()
        else:
            if base.buffer is None or base.group is not group:
                raise ValueError("cannot reuse an invalid symmetric buffer")
            if layout["bytes"] > base.buffer.nbytes:
                raise ValueError(f"the configuration needs {layout['bytes']} bytes, the buffer has {base.buffer.nbytes}")
            self.buffer, self.handle = base.buffer, base.handle

        act = torch.float8_e4m3fn if with_sf else torch.bfloat16
        t, rows, shared_inter = num_max_tokens_per_rank, layout["max_rows"], intermediate_hidden * num_shared_experts
        self.x = self._view(layout["x"], (t, hidden), act)
        self.x_sf = self._view(layout["x_sf"], (t, hidden // 128), torch.int32) if with_sf else None
        self.topk_idx = self._view(layout["topk_idx"], (t, num_topk), torch.int64)
        self.topk_weights = self._view(layout["topk_weights"], (t, num_topk), torch.float32)
        self.shared_l1_acts = self.x
        mn_major = lambda key, n, k, pitch: self._view(layout[key], (n, k // 128), torch.int32, (1, pitch))
        self.shared_l1_acts_sf = mn_major("shared_l1_sf", t, hidden, layout["shared_sf_pitch"]) \
            if with_sf and num_shared_experts else None
        self.shared_l2_acts = self._view(layout["shared_l2_acts"], (t, shared_inter), act) if num_shared_experts else None
        self.shared_l2_acts_sf = mn_major("shared_l2_sf", t, shared_inter, layout["shared_sf_pitch"]) \
            if with_sf and num_shared_experts else None
        self.l1_acts = self._view(layout["l1_acts"], (rows, hidden), act)
        self.l1_acts_sf = mn_major("l1_sf", rows, hidden, layout["sf_pitch"]) if with_sf else None
        self.l2_acts = self._view(layout["l2_acts"], (rows, intermediate_hidden), act)
        self.l2_acts_sf = mn_major("l2_sf", rows, intermediate_hidden, layout["sf_pitch"]) if with_sf else None

    def _view(self, offset: int, size, dtype: torch.dtype, stride=None) -> torch.Tensor:
        itemsize = torch.empty((), dtype=dtype).element_size()
        if stride is None:
            stride = tuple(math.prod(size[i + 1:]) for i in range(len(size)))
        nbytes = (1 + sum((s - 1) * st for s, st in zip(size, stride))) * itemsize
        raw = self.buffer[offset:offset + nbytes].view(dtype)
        return raw.as_strided(size, stride)

    def destroy(self):
        self.handle = self.buffer = self.group = None
        self.x = self.x_sf = None


def get_symm_buffer_for_mega_moe(group: dist.ProcessGroup, num_experts: int, num_max_tokens_per_rank: int,
                                 num_topk: int, hidden: int, intermediate_hidden: int, num_shared_experts: int = 0,
                                 use_fp8_dispatch: Optional[bool] = None, mma_type: str = "fp8xfp4",
                                 activation: str = "swiglu") -> SymmBuffer:
    if use_fp8_dispatch is not None:
        assert use_fp8_dispatch == (mma_type.split("x")[0] == "fp8")
        warnings.warn("`use_fp8_dispatch` will be deprecated in the future, please use `mma_type`",
                      DeprecationWarning, stacklevel=2)
    return SymmBuffer(group, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden,
                      num_shared_experts, mma_type=mma_type, activation=activation)


def _interleave_gate_up(t: torch.Tensor, gran: int = 8) -> torch.Tensor:
    # rows [gate 0..7, up 0..7, gate 8..15, up 8..15, ...] instead of [gate | up], strides kept
    squeeze = t.dim() == 2
    t = t.unsqueeze(0) if squeeze else t
    g, n, *rest = t.shape
    gate = t[:, :n // 2].reshape(g, n // 2 // gran, gran, *rest)
    up = t[:, n // 2:].reshape(g, n // 2 // gran, gran, *rest)
    result = torch.empty_like(t).copy_(torch.stack([gate, up], dim=2).reshape(g, n, *rest))
    return result.squeeze(0) if squeeze else result


def transform_weights_for_mega_moe(l1_weights, l2_weights, activation: str = "swiglu"):
    """L1 weights (and their scales) with gate and up rows interleaved by 8, so that the SwiGLU
    reads both halves from one 64-column span of the L1 output; L2 is unchanged. The scales stay
    in the layout of `transform_sf_into_required_layout`."""
    if activation != "swiglu":
        raise ValueError(f"only the swiglu activation is supported, got {activation}")
    if isinstance(l1_weights, tuple):
        return (_interleave_gate_up(l1_weights[0]), _interleave_gate_up(l1_weights[1])), l2_weights
    return _interleave_gate_up(l1_weights), l2_weights


def _split(weights) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    return weights if isinstance(weights, tuple) else (weights, None)


def _mega_moe(y, l1_weights, l2_weights, sym_buffer: SymmBuffer, shared_l1_weights, shared_l2_weights,
              cumulative_local_expert_recv_stats, activation, activation_clamp, fast_math, with_sf):
    if activation != "swiglu":
        raise ValueError(f"only the swiglu activation is supported, got {activation}")
    if (shared_l1_weights is None) != (shared_l2_weights is None):
        raise ValueError("pass both shared expert weights or neither")
    if (sym_buffer.mma_type != "bf16xbf16") != with_sf:
        raise ValueError(f"the buffer was made for {sym_buffer.mma_type}")
    (l1, l1_sf), (l2, l2_sf) = _split(l1_weights), _split(l2_weights)
    (s1, s1_sf), (s2, s2_sf) = (_split(shared_l1_weights), _split(shared_l2_weights)) if shared_l1_weights is not None \
        else ((None, None), (None, None))
    num_tokens, hidden = y.shape
    if y.dtype != torch.bfloat16 or not y.is_contiguous() or hidden != sym_buffer.hidden:
        raise ValueError("y must be a contiguous BF16 [tokens, hidden] tensor")
    if num_tokens > sym_buffer.num_max_tokens_per_rank:
        raise ValueError("more tokens than the buffer holds")
    if l1.dim() != 3 or l2.dim() != 3 or l1.size(0) != l2.size(0) or \
            l1.size(0) * sym_buffer.group.size() != sym_buffer.num_experts or \
            l1.size(1) != 2 * sym_buffer.intermediate_hidden:
        raise ValueError("expert weights must be L1 [experts per rank, 2 intermediate, hidden], "
                         "L2 [experts per rank, hidden, intermediate]")
    if (s1 is not None) and s2.size(1) != sym_buffer.intermediate_hidden * sym_buffer.num_shared_experts:
        raise ValueError("the shared expert weights do not match the buffer's shared experts")
    if (s1 is None) != (sym_buffer.num_shared_experts == 0):
        raise ValueError("the buffer was made for a different number of shared experts")
    stats = cumulative_local_expert_recv_stats
    if stats is not None and (stats.dtype != torch.int32 or stats.numel() != l1.size(0) or not stats.is_contiguous()):
        raise ValueError("cumulative_local_expert_recv_stats must be a contiguous int32 [experts per rank] tensor")
    clamp = float("inf") if activation_clamp is None else float(activation_clamp)
    if clamp < 0:
        raise ValueError("activation_clamp must be non-negative")
    _C.cdna4_mega_moe(y, l1, l1_sf, l2, l2_sf, s1, s1_sf, s2, s2_sf, stats, sym_buffer.buffer,
                      sym_buffer.handle.buffer_ptrs, sym_buffer.handle.rank, sym_buffer.num_max_tokens_per_rank,
                      sym_buffer.num_experts, sym_buffer.num_topk, clamp, bool(fast_math))


def fp8_fp4_mega_moe(y: torch.Tensor, l1_weights: Tuple[torch.Tensor, torch.Tensor],
                     l2_weights: Tuple[torch.Tensor, torch.Tensor], sym_buffer: SymmBuffer,
                     shared_l1_weights: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                     shared_l2_weights: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                     cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                     recipe: Tuple[int, int, int] = (1, 1, 32), activation: str = "swiglu",
                     activation_clamp: Optional[float] = None, fast_math: bool = True) -> None:
    """y = the MoE of the tokens in sym_buffer (x, x_sf, topk_idx, topk_weights): FP8 activations and
    FP8 or packed FP4 expert weights with UE8M0 scales per 32 values, SwiGLU with the top-k weight
    and the clamp, and the shared experts (FP8)."""
    if tuple(recipe) != (1, 1, 32):
        raise ValueError("Mega MoE takes the (1, 1, 32) recipe")
    _mega_moe(y, l1_weights, l2_weights, sym_buffer, shared_l1_weights, shared_l2_weights,
              cumulative_local_expert_recv_stats, activation, activation_clamp, fast_math, True)


def bf16_mega_moe(y: torch.Tensor, l1_weights: torch.Tensor, l2_weights: torch.Tensor, sym_buffer: SymmBuffer,
                  shared_l1_weights: Optional[torch.Tensor] = None, shared_l2_weights: Optional[torch.Tensor] = None,
                  cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None, activation: str = "swiglu",
                  activation_clamp: Optional[float] = None, fast_math: bool = True) -> None:
    """As fp8_fp4_mega_moe with BF16 activations and weights."""
    _mega_moe(y, l1_weights, l2_weights, sym_buffer, shared_l1_weights, shared_l2_weights,
              cumulative_local_expert_recv_stats, activation, activation_clamp, fast_math, False)
