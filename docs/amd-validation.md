# AMD backend and shared HIP JIT validation

The AMD backend now uses the HIP backend of
[OpenXPUKernel/DeepJIT](https://github.com/OpenXPUKernel/DeepJIT). The submodule
pins the exact revision. Kernel implementations, tensor validation, shape
planning and MoE scheduling stay in DeepGEMM. HIPRTC compilation, source/header
hashing, persistent caching, module ownership and native launches live in DeepJIT.

This integration pins HIP's DeepJIT separately under `third-party/deep_jit_hip`.
The original `third-party/deep_jit` gitlink and CUDA include path remain at the
pre-AMD revision: a HIP bring-up must not implicitly opt CUDA into the newer
stable-PyTorch-ABI migration. Neither backend searches the other include root.

## Current Mega MoE path

Aligned FP8 activations and packed FP4 weights use the complete cooperative
kernel described in [the Mega MoE guide](cdna4-mega-moe.md). The implementation
owns routing, gather, both GEMMs, shared experts, peer return and ordered output
reduction. Initialization and weight preparation precede graph capture.

The previous stage-only and alternate scheduling experiments have been removed.

## Validation on AMD GPU

| Check | Result |
| --- | --- |
| Dense FP8/BF16 and grouped GEMM contracts | 388 passed, 7 optional cases skipped |
| Mega MoE, H1280/I768 | Five cases on eight ranks: serial equality, FP32 oracle, receive statistics, graph replay and eight asynchronous input updates pass |
| Mega MoE, H2048/I1024 | The same five-case checks pass |
| CPU scheduling and prepared-weight layout | 11 tests pass |

MoE cases include empty ranks, concentrated and fully masked routes, tail rows,
and changed activations, input scales, routing and expert/shared weight bytes.
The asynchronous mutation scheme retains the previously qualified contract.
Performance artifacts and experimental diagnostics remain outside the library
source tree. CUDA and Ascend hardware were not validated in this AMD run.

## Reproduce

Use a ROCm PyTorch environment with the pinned submodules initialized:

```sh
git submodule update --init --recursive
DG_USE_LOCAL_VERSION=0 python setup.py build_ext --inplace
export PYTHONPATH="$PWD"
export DG_JIT_CACHE_DIR=/path/to/writable/jit-cache
python -m pytest -q tests/test_cdna4_fp8_gemm.py tests/test_cdna4_bf16_gemm.py \
  tests/test_cdna4_grouped_fp8_gemm.py
python tests/test_cdna4_mega_moe.py --ranks 8
python tests/test_cdna4_mega_moe_schedule.py
python tests/test_moe_weight_panels.py
python tests/bench_cdna4_mega_moe.py --ranks 8
```

Run GPU workloads sequentially. For JIT integration checks, use a fresh process
and writable cache with `tests/test_cdna4_jit.py`.

### Ordered expert returns

The native reducer and the independent serial oracle can produce different BF16 words in the optional shared-expert stress case with hidden size 5120 and intermediate size 2304. Comparing the original resident reduction and the wave-slot reduction on identical inputs produced bitwise identical native outputs. The routed return words, routed activations and scales, and the shared intermediate activations and scales matched the independent reference. Zeroing the shared down-projection weights removed the final discrepancy. The test retains its strict bitwise assertion for this case.

### Balanced synthetic routes

Run `python tests/bench_cdna4_mega_moe.py --ranks 8 --routing balanced` to distribute the same number of routed rows to every rank. Experts receive equal counts when the total number of slots is divisible by the expert count. Otherwise the counts differ by one. Each token still selects distinct experts. Successive values of `--route-phase` rotate which experts receive the extra rows. The benchmark checks native receive counters against the global assignment counts before timing. It reports the per-expert range and per-rank totals and checks the complete output against the reference before and after timing.

For resident FP4 expert tiles with at most 64 rows, `DG_MEGA_MOE_SHORT_COLUMN_WAVES` distributes the columns of an unpaired tile across all four waves. Use `1` for the up projection, `2` for the down projection or `3` for both. Select the corresponding projections with `DG_MEGA_MOE_ACTIVE_SLABS`, for example `3,3` for both. The paired up-projection schedule has its own switch, `DG_MEGA_MOE_HALF_ROW_PAIR=1`. Compare the full model on the intended routing distribution when selecting these options.
