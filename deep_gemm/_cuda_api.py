import atexit
import os

import torch as torch

# Set some default environment provided at setup
try:
    # noinspection PyUnresolvedReferences
    from .envs import persistent_envs

    for key, value in persistent_envs.items():
        if key not in os.environ:
            os.environ[key] = value
except ImportError:
    pass

# Configs
# Locality domain helpers
# Some utils
from . import _C
from . import locality_domain as locality_domain
from . import testing as testing
from . import utils as utils

# cuBLASLt Kernels
# DeepGEMM Kernels
# Epilogue classes
from ._C import (
    batched_symm as batched_symm,
)
from ._C import (
    batched_syrk as batched_syrk,
)
from ._C import (
    bf16_gemm_nn as bf16_gemm_nn,
)
from ._C import (
    # BF16 GEMMs
    bf16_gemm_nt as bf16_gemm_nt,
)
from ._C import (
    bf16_gemm_tn as bf16_gemm_tn,
)
from ._C import (
    bf16_gemm_tt as bf16_gemm_tt,
)
from ._C import (
    # MegaGate kernels
    bf16_mega_gate as bf16_mega_gate,
)
from ._C import (
    cublaslt_gemm_nn as cublaslt_gemm_nn,
)
from ._C import (
    cublaslt_gemm_nt as cublaslt_gemm_nt,
)
from ._C import (
    cublaslt_gemm_tn as cublaslt_gemm_tn,
)
from ._C import (
    cublaslt_gemm_tt as cublaslt_gemm_tt,
)
from ._C import (
    cublaslt_nvfp4_gemm_nt as cublaslt_nvfp4_gemm_nt,
)
from ._C import (
    # Einsum kernels
    einsum as einsum,
)
from ._C import (
    epilogue as epilogue,
)
from ._C import (
    # FP8/FP4 GEMMs (alias)
    fp4_gemm_nt as fp4_gemm_nt,
)
from ._C import (
    fp8_einsum as fp8_einsum,
)
from ._C import (
    fp8_fp4_gemm_nn as fp8_fp4_gemm_nn,
)
from ._C import (
    # FP8-FP4 GEMMs (accept any FP8/FP4 operand combination)
    fp8_fp4_gemm_nt as fp8_fp4_gemm_nt,
)
from ._C import (
    fp8_fp4_gemm_tn as fp8_fp4_gemm_tn,
)
from ._C import (
    fp8_fp4_gemm_tt as fp8_fp4_gemm_tt,
)
from ._C import (
    fp8_fp4_mqa_logits as fp8_fp4_mqa_logits,
)
from ._C import (
    fp8_fp4_paged_mqa_logits as fp8_fp4_paged_mqa_logits,
)
from ._C import (
    fp8_fp4_paged_sparse_mqa_logits as fp8_fp4_paged_sparse_mqa_logits,
)
from ._C import (
    fp8_fp4_sparse_mqa_logits as fp8_fp4_sparse_mqa_logits,
)
from ._C import (
    fp8_gemm_nn as fp8_gemm_nn,
)
from ._C import (
    fp8_gemm_nt as fp8_gemm_nt,
)
from ._C import (
    fp8_gemm_tn as fp8_gemm_tn,
)
from ._C import (
    fp8_gemm_tt as fp8_gemm_tt,
)
from ._C import (
    # MegaMoE
    get_block_m_for_mega_moe as get_block_m_for_mega_moe,
)
from ._C import (
    # Attention kernels
    get_mqa_logits_metadata as get_mqa_logits_metadata,
)
from ._C import (
    get_num_sms as get_num_sms,
)
from ._C import (
    get_paged_mqa_logits_metadata as get_paged_mqa_logits_metadata,
)
from ._C import (
    get_paged_sparse_mqa_logits_metadata as get_paged_sparse_mqa_logits_metadata,
)
from ._C import (
    get_pdl as get_pdl,
)
from ._C import (
    get_sparse_mqa_logits_metadata as get_sparse_mqa_logits_metadata,
)
from ._C import (
    get_tc_util as get_tc_util,
)
from ._C import (
    k_grouped_bf16_gemm_tn_contiguous as k_grouped_bf16_gemm_tn_contiguous,
)
from ._C import (
    k_grouped_fp4_gemm_nt_contiguous as k_grouped_fp4_gemm_nt_contiguous,
)
from ._C import (
    k_grouped_fp8_gemm_nt_contiguous as k_grouped_fp8_gemm_nt_contiguous,
)
from ._C import (
    k_grouped_fp8_gemm_tn_contiguous as k_grouped_fp8_gemm_tn_contiguous,
)
from ._C import (
    m_grouped_bf16_gemm_nn_contiguous as m_grouped_bf16_gemm_nn_contiguous,
)
from ._C import (
    m_grouped_bf16_gemm_nt_contiguous as m_grouped_bf16_gemm_nt_contiguous,
)
from ._C import (
    m_grouped_bf16_gemm_nt_masked as m_grouped_bf16_gemm_nt_masked,
)
from ._C import (
    m_grouped_fp4_gemm_nt_contiguous as m_grouped_fp4_gemm_nt_contiguous,
)
from ._C import (
    m_grouped_fp4_gemm_nt_masked as m_grouped_fp4_gemm_nt_masked,
)
from ._C import (
    m_grouped_fp8_fp4_gemm_nn_contiguous as m_grouped_fp8_fp4_gemm_nn_contiguous,
)
from ._C import (
    m_grouped_fp8_fp4_gemm_nt_contiguous as m_grouped_fp8_fp4_gemm_nt_contiguous,
)
from ._C import (
    m_grouped_fp8_fp4_gemm_nt_masked as m_grouped_fp8_fp4_gemm_nt_masked,
)
from ._C import (
    m_grouped_fp8_gemm_nn_contiguous as m_grouped_fp8_gemm_nn_contiguous,
)
from ._C import (
    m_grouped_fp8_gemm_nt_contiguous as m_grouped_fp8_gemm_nt_contiguous,
)
from ._C import (
    m_grouped_fp8_gemm_nt_masked as m_grouped_fp8_gemm_nt_masked,
)
from ._C import (
    # Hyperconnection kernels
    mega_mhc as mega_mhc,
)
from ._C import (
    set_block_size_multiple_of as set_block_size_multiple_of,
)
from ._C import (
    set_ignore_compile_dims as set_ignore_compile_dims,
)
from ._C import (
    set_num_sms as set_num_sms,
)
from ._C import (
    set_pdl as set_pdl,
)
from ._C import (
    set_tc_util as set_tc_util,
)
from ._C import (
    shutdown,
)
from ._C import (
    tf32_hc_prenorm_gemm as tf32_hc_prenorm_gemm,
)
from ._C import (
    # Layout kernels
    transform_sf_into_required_layout as transform_sf_into_required_layout,
)
from ._C import (
    use_deterministic_algorithms as use_deterministic_algorithms,
)
from .locality_domain import (
    destroy_localizer as destroy_localizer,
)
from .locality_domain import (
    get_num_locality_domains as get_num_locality_domains,
)
from .locality_domain import (
    is_localization_available as is_localization_available,
)
from .locality_domain import (
    is_localized as is_localized,
)
from .locality_domain import (
    localize as localize,
)

# Mega kernels
from .mega import (
    SymmBuffer as SymmBuffer,
)
from .mega import (
    bf16_mega_moe as bf16_mega_moe,
)
from .mega import (
    fp8_fp4_mega_moe as fp8_fp4_mega_moe,
)
from .mega import (
    get_symm_buffer_for_mega_moe as get_symm_buffer_for_mega_moe,
)
from .mega import (
    transform_weights_for_mega_moe as transform_weights_for_mega_moe,
)
from .utils import *

# Legacy Triton kernels for A100
try:
    from . import legacy as legacy
except Exception as e:
    print(f"Failed to load legacy DeepGEMM A100 Triton kernels: {e}")

# Initialize CPP modules
_C.init(os.path.dirname(os.path.abspath(__file__)))
# Native cuBLASLt exit callbacks can free handle storage before C++ static destructors.
# Release handles and workspaces while Python, PyTorch and CUDA are still available.
atexit.register(shutdown)

__version__ = "2.8.1"
