"""Independent general-GEMM reference for AMD Mega MoE tests."""

from .. import _C
from .._hip_mega_moe import ResidentMoEPanelWeights


def reference_mega_moe(
    y,
    l1,
    l2,
    buffer,
    *,
    shared_l1_weights=None,
    shared_l2_weights=None,
    cumulative_local_expert_recv_stats=None,
    activation_clamp=None,
    fast_math=True,
):
    def ordinary(pair):
        if isinstance(pair, ResidentMoEPanelWeights):
            w, sf = pair
            e, n, kb = w.shape
            w = (
                w.view(e, n // 256, kb // 64, 256, 64)
                .permute(0, 1, 3, 2, 4)
                .contiguous()
                .view_as(w)
            )
            return w, sf
        return pair if isinstance(pair, tuple) else (pair, None)

    w1, sf1 = ordinary(l1)
    w2, sf2 = ordinary(l2)
    s1, ssf1 = (
        ordinary(shared_l1_weights) if shared_l1_weights is not None else (None, None)
    )
    s2, ssf2 = (
        ordinary(shared_l2_weights) if shared_l2_weights is not None else (None, None)
    )
    _C.cdna4_mega_moe(
        y,
        w1,
        sf1,
        w2,
        sf2,
        s1,
        ssf1,
        s2,
        ssf2,
        cumulative_local_expert_recv_stats,
        buffer.buffer,
        buffer.handle.buffer_ptrs,
        buffer.handle.rank,
        buffer.num_max_tokens_per_rank,
        buffer.num_experts,
        buffer.num_topk,
        float("inf") if activation_clamp is None else activation_clamp,
        fast_math,
        reference=True,
    )
