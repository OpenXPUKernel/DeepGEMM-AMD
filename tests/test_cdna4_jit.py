"""Consumer integration checks for the shared HIP JIT and concurrent first use."""
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

if not torch.version.hip or not torch.cuda.is_available():
    pytest.skip("ROCm device required", allow_module_level=True)

import deep_gemm


def test_runtime_is_device_local():
    with torch.cuda.device(0):
        first = deep_gemm._C.get_jit()
        assert deep_gemm._C.get_jit() is first
    if torch.cuda.device_count() > 1:
        with torch.cuda.device(1):
            second = deep_gemm._C.get_jit()
            assert second is not first
            assert deep_gemm._C.get_jit() is second
    with torch.cuda.device(0):
        assert deep_gemm._C.get_jit() is first


def test_concurrent_first_gemm_use():
    # Run this file in a fresh process/cache to exercise concurrent compilation.
    # Acquiring the JIT mutex while holding Python's GIL would deadlock against
    # another compiling thread that needs to reacquire the GIL before returning.
    def worker(value):
        torch.cuda.set_device(0)
        a = torch.full((17, 256), value, device="cuda", dtype=torch.bfloat16)
        b = torch.ones((65, 256), device="cuda", dtype=torch.bfloat16)
        out = torch.empty((17, 65), device="cuda", dtype=torch.bfloat16)
        deep_gemm.bf16_gemm_nt(a, b, out)
        torch.cuda.synchronize()
        torch.testing.assert_close(out, torch.full_like(out, value * 256), rtol=0, atol=0)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(worker, [1.0, 2.0, 3.0, 4.0]))
