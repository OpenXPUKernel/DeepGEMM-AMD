# AMD backend and shared HIP JIT validation

The AMD backend now uses the HIP backend of
[OpenXPUKernel/DeepJIT](https://github.com/OpenXPUKernel/DeepJIT). The submodule
pins the exact revision. Kernel implementations, tensor validation, shape
planning and MoE scheduling stay in DeepGEMM. HIPRTC compilation, source/header
hashing, persistent caching, module ownership and native launches live in DeepJIT.

The current Stage1 queue optimization is included, opt-in through
`DG_MEGA_MOE_STAGE1_QUEUE=1`. Default scheduling is unchanged. The Stage1 path
requires FP8×FP4 and no shared expert; see [the MegaMoE guide](cdna4-mega-moe.md)
for constraints. The full resident pipeline remains an experimental option.

## Validation on 2026-10-20

Environment: 8 MI355X GPUs, ROCm 7.2.4, ROCm PyTorch 2.8.0, GCC 12.2. The
GCC 12 build uses DeepJIT's fmt fallback; the standard formatting implementation
is still used on hosts that provide `std::format`.

| Check | Result |
|---|---|
| JIT device identity and concurrent first use | 2 passed in a fresh cache |
| FP8 dense GEMM/storage contracts | 147 passed, 7 skipped |
| BF16, FP4, grouped GEMM, einsum, HC, layout, MegaGate, MegaMHC, MQA | 678 passed, 9 skipped |
| Build selection/packaging and runtime test selection | 9 passed; 3 CUDA-only shutdown tests skipped |
| Stage1 queue, 8 ranks, H7168/I512 | Five routing cases pass, including empty, masked and skewed routes; changed-input and asynchronous graph checks pass |
| Default MoE, 8 ranks | Five cases each for FP8×FP4, FP8×FP8 and BF16 pass |
| Full resident pipeline, 2 ranks | Five cases and changed-input/asynchronous graphs pass |
| DeepJIT HIP integration | Arithmetic, cache reuse across processes, include invalidation, flags, current stream, graph, multiple entry points, cooperative launch, 128 KiB LDS, error cleanup and device isolation pass |
| DeepJIT exception/backtrace | Pass on the GCC 12 formatting fallback |

Optional large-memory and unsupported cases retain their existing skip conditions.
The cuBLASLt shutdown tests are explicitly CUDA-only. The packaging fixture now
includes the required vendored uni_symm headers, without requiring CUTLASS on HIP.
CUDA and Ascend hardware regression suites were not run on this AMD machine.

## End-to-end MegaMoE

Shape: EP8, 8192 tokens/rank, H7168, I3072, 384 routed experts, top-6, **no
shared expert**, FP8 activation / FP4 weights, BF16 return. Inputs are already
quantized. Weight conversion and initialization are outside timing; dispatch,
both GEMMs, activation, communication, synchronization and final combine are
inside the captured graph. Outputs match the serial path bitwise.

- Uninstrumented graph: **4.482 ms**, **28.98% MFU**, slowest rank; 100 warmup
  replays and 200 timed replays.
- This is consistent with the preceding approximately 4.48 ms / 29.0% result;
  this refactor does not claim an additional kernel performance improvement.
- Separate diagnostic graph: Stage1 approximately 2.319 ms / 37.33% MFU;
  Stage2 approximately 2.233 ms / 19.39% MFU. Stage maxima can come from different
  ranks and must not be added as though they were a single rank's latency.
- Useful FLOPs are `6 * H * I * sum(received_rows)`. Dense peak convention:
  5000 TFLOPS per GPU. The 50% MFU objective remains unmet.

Machine-readable results: [validation/amd-hip-jit.json](validation/amd-hip-jit.json).

## Reproduce

Use a ROCm PyTorch environment with the pinned submodule initialized:

```bash
git submodule update --init --recursive
DG_USE_LOCAL_VERSION=0 python setup.py build_ext --inplace
export PYTHONPATH="$PWD"
export DG_JIT_CACHE_DIR=/path/to/writable/jit-cache

python -m pytest -q tests/test_cdna4_fp8_gemm.py
python -m pytest -q tests/test_cdna4_bf16_gemm.py tests/test_cdna4_fp4_gemm.py \
  tests/test_cdna4_grouped_fp8_gemm.py tests/test_cdna4_einsum.py \
  tests/test_cdna4_hyperconnection.py tests/test_cdna4_layout.py \
  tests/test_cdna4_mega_gate.py tests/test_cdna4_mega_mhc.py tests/test_cdna4_mqa_logits.py
python -m pytest -q tests/test_build_install.py tests/test_lazy_init.py
timeout 120 python -m pytest -q tests/test_cdna4_jit.py

DG_MEGA_MOE_TEST_HIDDEN=7168 python tests/test_cdna4_mega_moe_stage1.py --ranks 8
DG_MEGA_MOE_TEST_SKEW=1 python tests/test_cdna4_mega_moe.py --ranks 8
python tests/test_cdna4_mega_moe_pipeline.py --ranks 2

DG_MEGA_MOE_STAGE1_QUEUE=1 python tests/bench_cdna4_mega_moe.py \
  --ranks 8 --mma-type fp8xfp4 --shared-experts 0 \
  --num-tokens 8192 --hidden 7168 --inter 3072 --experts 384 --topk 6 \
  --graph --iterations 200 --compare-serial --phase-mfu --json-output result.json
```

Run GPU workloads sequentially. The JIT concurrency test is most useful in a
fresh process with an empty cache: it checks device-local runtime identity and
simultaneous first-use compilation without a Python GIL/cache-mutex deadlock.
