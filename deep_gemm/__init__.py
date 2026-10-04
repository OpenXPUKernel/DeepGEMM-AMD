"""DeepGEMM public API with CUDA and CDNA4 HIP backends."""
import torch

__version__ = '2.8.1'

# KnowingNothing's CDNA4 backend uses the same dense FP8 entry points on ROCm.
# The CUDA module remains the default for NVIDIA builds.
if getattr(torch.version, "hip", None) is not None:
    from . import _C
    from ._hip_api import *
else:
    from ._cuda_api import *
