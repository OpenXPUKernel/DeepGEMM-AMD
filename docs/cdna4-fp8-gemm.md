# CDNA4 FP8 GEMM NT

This backend adds a HIP implementation of DeepGEMM's dense FP8 GEMM interface for CDNA4 architecture GPUs. The implementation is marked as a KnowingNothing implementation and keeps the CUDA call signature for `fp8_gemm_nt`, `fp8_gemm_nn`, `fp8_gemm_tn`, and `fp8_gemm_tt`.

The supported operation is

```text
D = alpha * (A @ B.T) + C
```

`A` and `B` contain OCP FP8 E4M3 values. Their scale tensors use UE8M0 values with 32 or 128 elements per K block. `recipe_a=(1, 32)` and `recipe_b=(1, 32)` are the direct MXFP8 form; 128-element and mixed A/B recipes are read directly by the kernel. Scales can be passed as float32 tensors or as the packed int32 layout returned by `transform_sf_into_required_layout`. `D` and `C` can be BF16 or FP32. Dense GEMMs and the `nn`/`tn`/`tt` aliases are supported; grouped GEMMs and custom epilogues are not part of this backend.

The production kernels were refreshed from the current tutorial implementation
on 2026-10-06. Inputs remain in their native K-major or MN-major storage. Physical
row strides must cover a row, be divisible by four bytes, and be below `1 << 24`.
For a physical row whose length is not divisible by four, the base address must
also be four-byte aligned. Invalid storage raises an error rather than launching
an implicit copy. Pad storage in the preceding quantization/preparation stage
when needed; the logical M/N/K dimensions may still be ragged.

Packed scales must be int32, MN-major, with column pitch at least `align(rows, 4)`
and divisible by four. `transform_sf_into_required_layout` returns valid packed
scales unchanged. FP32 scale preparation broadcasts MN blocks and packs UE8M0;
it does not expand a 128-wide block into four 32-wide blocks. The shape of packed
scales is `[rows, ceil(K / (4 * gran_k))]`. Use the recipe matching the packed
scale granularity.

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

Run the complete benchmark:

```bash
python tests/bench_cdna4_fp8.py --csv results/cdna4-fp8.csv
```

The default catalog contains **129 configurations across 93 distinct `(M, N, K)`
shapes**, including every performance configuration from the updated tutorial:

| Suite | Configurations | Coverage |
| --- | ---: | --- |
| `forward` | 21 | Seven N/K pairs with M = 1, 128, 4096; BF16 output |
| `dgrad` | 14 | Seven backward shapes, K-major and native MN-major B; BF16 output |
| `wgrad` | 14 | Seven weight-gradient shapes, K-major or native MN-major A/B; FP32 in-place accumulation |
| `extended` | 42 | The same seven N/K pairs with M = 8, 16, 32, 64, 256, 1024 |
| `square` | 8 | Square and large rectangular GEMMs |
| `edge` | 18 | Nine non-aligned/small shapes, BF16 and FP32 accumulation |
| `recipe` | 12 | All four native operand layouts with 128/128, 32/128 and 128/32 scale blocks |

`--suite tutorial` selects 52 configurations: the 49 forward/backward cases plus the new 128-wide scale case and two square shapes. To inspect
all shapes or narrow a run:

```bash
python tests/bench_cdna4_fp8.py --list-shapes
python tests/bench_cdna4_fp8.py --suite tutorial
python tests/bench_cdna4_fp8.py --suite wgrad
python tests/bench_cdna4_fp8.py --suite recipe
python tests/bench_cdna4_fp8.py --shape 128,7168,4096 --shape 4096,4096,4096 --dtype both
python tests/bench_cdna4_fp8.py --shape 512,1024,4096 --dtype fp32 --layout MM --accumulate --gran-k-b 128
```

Catalog coverage, metric units and CLI/CSV behavior have CPU-only regression tests:

```bash
python tests/test_cdna4_benchmark.py
```

Each successful row reports the shape, output type, accumulation mode, input
layout, DeepGEMM/PyTorch latency in microseconds, effective TFLOPS for both,
and `speedup = pytorch_us / deepgemm_us`. MFMA compute and split-K reduction are included; native operand reads need no
layout conversion kernel. Throughput is
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
Both paths receive prepared native layouts; operand padding/preparation is outside timing.

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

Validation on 2026-10-06 used the refreshed production kernels through the normal
ROCm wheel installation. **153 GPU cases passed**, including native layouts,
mixed scale granularities, ragged/offset storage, invalid arguments, and six
opt-in tests accessing K-/MN-major operands, output or scales beyond 4 GiB. The packed-input
profiler check found no input transpose, alignment-copy or scale-expansion work.
The default benchmark completed **129/129** sampled accuracy checks and timings
with no skipped cases. Build/installer regressions (9) and benchmark regressions
(10) also passed.

To enable the large-address tests explicitly:

```bash
DG_TEST_LARGE=1 pytest -q tests/test_cdna4_fp8_gemm.py
```

The [complete relative results](benchmarks/cdna4-fp8-gemm-relative.csv) retain
every configuration, including values below 1×. The run used the default warmup,
event sampling and cache-flush settings on a CDNA4 architecture GPU. BF16 and
FP32 accumulation suites use their separately documented PyTorch baselines.

| Suite | Successful configurations | Geometric mean speedup vs PyTorch |
| --- | ---: | ---: |
| forward | 21 | 1.008× |
| dgrad | 14 | 1.508× |
| wgrad | 14 | 7.803× |
| extended | 42 | 0.826× |
| square | 8 | 0.887× |
| recipe | 12 | 1.634× |
| edge | 18 | 0.671× |

Values below 1× indicate a slower result under this measurement. These are
public-interface timings, not isolated instruction throughput. The benchmark's
default terminal/CSV output also includes absolute latency and effective TFLOPS;
use `--relative-only` for reports containing only speedups.
