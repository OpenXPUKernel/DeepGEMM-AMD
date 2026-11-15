import torch

if getattr(torch.version, "hip", None) is not None:
    from .._hip_api import (
        get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor,
        get_mk_alignment_for_contiguous_layout,
        get_mn_major_tma_aligned_packed_ue8m0_tensor,
        get_mn_major_tma_aligned_tensor,
        get_theoretical_mk_alignment_for_contiguous_layout,
        get_tma_aligned_size,
        set_mk_alignment_for_contiguous_layout,
    )
else:
    from .._C import (
        get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor as get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor,
    )
    from .._C import (
        get_mk_alignment_for_contiguous_layout,
    )
    from .._C import (
        get_mn_major_tma_aligned_packed_ue8m0_tensor as get_mn_major_tma_aligned_packed_ue8m0_tensor,
    )
    from .._C import (
        get_mn_major_tma_aligned_tensor as get_mn_major_tma_aligned_tensor,
    )
    from .._C import (
        get_theoretical_mk_alignment_for_contiguous_layout as get_theoretical_mk_alignment_for_contiguous_layout,
    )
    from .._C import (
        get_tma_aligned_size as get_tma_aligned_size,
    )
    from .._C import (
        set_mk_alignment_for_contiguous_layout as set_mk_alignment_for_contiguous_layout,
    )

# Some alias
get_m_alignment_for_contiguous_layout = get_mk_alignment_for_contiguous_layout
get_k_alignment_for_contiguous_layout = get_mk_alignment_for_contiguous_layout
