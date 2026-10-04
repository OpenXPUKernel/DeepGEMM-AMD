# CDNA4 FP8 GEMM NT

This backend adds a HIP implementation of DeepGEMM's dense FP8 GEMM interface for CDNA4 architecture GPUs. The implementation is marked as a KnowingNothing implementation and keeps the CUDA call signature for `fp8_gemm_nt`, `fp8_gemm_nn`, `fp8_gemm_tn`, and `fp8_gemm_tt`.

The supported operation is

```text
D = alpha * (A @ B.T) + C
```

`A` and `B` contain OCP FP8 E4M3 values. Their scale tensors use UE8M0 values with 32 elements per K block. `recipe_a=(1, 32)` and `recipe_b=(1, 32)` are the direct MXFP8 form; a 128-element recipe is expanded internally. Scales can be passed as float32 tensors or as the packed int32 layout returned by `transform_sf_into_required_layout`. `D` and `C` can be BF16 or FP32. Dense GEMMs and the `nn`/`tn`/`tt` aliases are supported; grouped GEMMs and custom epilogues are not part of this backend.

Use a Python environment with ROCm-enabled PyTorch and the ROCm development
toolkit (HIP headers and HIPRTC). Install from the repository root:

```bash
bash install.sh
```

The build selects HIP automatically from `torch.version.hip` and reports
`Building DeepGEMM for ROCm/HIP`. It builds the native C++ pybind host extension
as `deep_gemm._C`, links the HIP runtime and HIPRTC, and packages the device
templates. No additional build script or backend flag is needed. CUDA and
CUTLASS dependencies are only used by the CUDA build. `ROCM_HOME` can point to
a non-default toolkit installation.

`install.sh` builds and installs with the same Python interpreter and stops if
either step fails. For development within a source checkout, use `bash develop.sh`;
`bash build.sh` produces the wheel without installing it.

On first use, the extension asks the cost model for a template specialization,
compiles that wrapper with HIPRTC, and caches the resulting code object in the
process. See [the CDNA4 kernel layout guide](cdna4-kernel-layout.md) for the source
organization.

Run correctness tests on a CDNA4 architecture GPU:

```bash
pytest -q tests/test_cdna4_fp8_gemm.py
```

The tests compare the operator with a PyTorch dequantize-and-matmul reference, check accumulation and `alpha`, exercise packed scales and layout aliases, cover empty M/K cases, and verify repeatability.

Build selection and installer failure handling can also be checked without a GPU:

```bash
python tests/test_build_install.py
```

Validation on 2026-10-04: `bash install.sh` and `bash develop.sh` succeeded in a
ROCm PyTorch environment without CUDA submodules. The installed wheel contained
one `_C` extension and its JIT templates. All 7 GPU test cases and 9 build/installer
regression tests passed. Additional exact-input checks exercised both MFMA
families, persistent execution, split-K, and repeated JIT launches.

Run the complete benchmark:

```bash
python tests/bench_cdna4_fp8.py --csv results/cdna4-fp8.csv
```

The default catalog contains **117 configurations across 93 distinct `(M, N, K)`
shapes**, including every performance configuration from the reference tutorial:

| Suite | Configurations | Coverage |
| --- | ---: | --- |
| `forward` | 21 | Seven N/K pairs with M = 1, 128, 4096; BF16 output |
| `dgrad` | 14 | Seven backward shapes, K-major and native MN-major B; BF16 output |
| `wgrad` | 14 | Seven weight-gradient shapes, K-major or native MN-major A/B; FP32 in-place accumulation |
| `extended` | 42 | The same seven N/K pairs with M = 8, 16, 32, 64, 256, 1024 |
| `square` | 8 | Square and large rectangular GEMMs |
| `edge` | 18 | Nine non-aligned/small shapes, BF16 and FP32 accumulation |

`--suite tutorial` selects the original 49 measured configurations. To inspect
all shapes or narrow a run:

```bash
python tests/bench_cdna4_fp8.py --list-shapes
python tests/bench_cdna4_fp8.py --suite tutorial
python tests/bench_cdna4_fp8.py --suite wgrad
python tests/bench_cdna4_fp8.py --shape 128,7168,4096 --shape 4096,4096,4096 --dtype both
python tests/bench_cdna4_fp8.py --shape 512,1024,4096 --dtype fp32 --layout MM --accumulate
```

Catalog coverage, metric units and CLI/CSV behavior have CPU-only regression tests:

```bash
python tests/test_cdna4_benchmark.py
```

Each successful row reports the shape, output type, accumulation mode, input
layout, DeepGEMM/PyTorch latency in microseconds, effective TFLOPS for both,
and `speedup = pytorch_us / deepgemm_us`. MFMA compute, split-K reduction and any
layout copies inside the public interface are included. Throughput is
`2*M*N*K / elapsed_seconds / 1e12`; it is an API-level measurement, not a
measurement of an isolated instruction or the device's peak throughput.

Timing uses the median of 30 GPU-event samples after at least five calls and
200 ms of warmup for each API. JIT compilation, FP8 quantization, scale packing,
and PyTorch operand preparation happen before timing. Both APIs use the same
cache policy: a fixed scratch-buffer flush before each sample (outside the event
interval), or warm caches with `--no-flush-cache`. GPU-event intervals include
launch gaps. `--rounds`, `--warmup` and `--warmup-ms` control sampling.

The PyTorch baseline consumes the same dequantized values and physical layouts.
BF16 output rows use BF16 operands; FP32 output rows use FP32 operands with TF32
disabled. Non-accumulating rows call `torch.matmul`; accumulating rows call
in-place `torch.addmm`, matching `C=D` on the FP8 path. Input layouts are shown as
two letters: `K` means K-major and `M` means MN-major for A and B respectively.
The FP8 wrapper's conversion of MN-major operands is included in its timing.

Before timing, up to 32 distributed rows and 32 columns are compared against an
FP32 reference. The reported `relative_l2` is the sampled relative L2 error; a
value above 0.01 fails the case. This aggregate check is not an elementwise
allclose guarantee. It runs before repeated accumulation changes the outputs.
OOM skips and failures remain visible in CSV, and any incomplete run exits with
a nonzero status. Geometric mean speedups are calculated from successful rows
only, with their counts shown.

For reports containing only relative performance, use:

```bash
python tests/bench_cdna4_fp8.py --relative-only --csv results/cdna4-relative.csv
```

This removes absolute timing and throughput columns from both the terminal table
and CSV. Neither output mode prints a GPU model or hardware configuration.

Recorded run on 2026-10-04 used the default full catalog, warmup, event sampling
and cache-flush settings on a CDNA4 architecture GPU. All 117 cases passed the
sampled accuracy check; no cases were skipped. The
[complete relative results](benchmarks/cdna4-fp8-gemm-relative.csv) retain every
row, including speedups below 1×. Group summaries are kept separate because
FP32 accumulation uses a different PyTorch baseline from BF16 GEMM.

| Suite | Successful configurations | Geometric mean speedup vs PyTorch |
| --- | ---: | ---: |
| forward | 21 | 0.466× |
| dgrad | 14 | 1.052× |
| wgrad | 14 | 5.160× |
| extended | 42 | 0.366× |
| square | 8 | 0.828× |
| edge | 18 | 0.541× |

Values below 1× mean DeepGEMM was slower under this measurement. These results
measure the current implementation and include public-interface overhead; they
are not a claim of peak kernel throughput. To record absolute metrics as well,
run the default command without `--relative-only`.
