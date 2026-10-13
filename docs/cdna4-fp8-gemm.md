# CDNA4 FP8 GEMM NT

This backend adds a HIP implementation of DeepGEMM's dense FP8 GEMM interface for CDNA4 architecture GPUs. The implementation is marked as a KnowingNothing implementation and keeps the CUDA call signature for `fp8_gemm_nt`, `fp8_gemm_nn`, `fp8_gemm_tn`, and `fp8_gemm_tt`.

The supported operation is

```text
D = alpha * (A @ B.T) + C
```

`A` and `B` contain OCP FP8 E4M3 values. Their scale tensors use UE8M0 values with 32 or 128 elements per K block. `recipe_a=(1, 32)` and `recipe_b=(1, 32)` are the direct MXFP8 form; 128-element and mixed A/B recipes are read directly by the kernel. Scales can be passed as float32 tensors or as the packed int32 layout returned by `transform_sf_into_required_layout`. `D` and `C` can be BF16 or FP32. Dense GEMMs, the `nn`/`tn`/`tt` aliases and the grouped GEMMs below are supported; custom epilogues are not part of this backend.

## Grouped GEMMs

The grouped entry points keep the CUDA signatures and layouts:

| API | Operands | Groups |
| --- | --- | --- |
| `m_grouped_fp8_gemm_nt_contiguous` | A `[M, K]`, B `[G, N, K]` (K- or MN-major), D `[M, N]` BF16 | `grouped_layout[m]` is the group of row m (-1 for padding), or with `use_psum_layout=True` `grouped_layout[g]` is the end row of group g |
| `m_grouped_fp8_gemm_nn_contiguous` | B passed as `[G, K, N]` | as above |
| `m_grouped_fp8_gemm_nt_masked` | A `[G, max_m, K]`, B `[G, N, K]`, D `[G, max_m, N]` BF16 | rows below `masked_m[g]`; other rows of D are not written |
| `k_grouped_fp8_gemm_tn_contiguous` | A `[sum_k, M]`, B `[sum_k, N]`, D `[G, M, N]` BF16/FP32, optional C | `grouped_layout[g]` is the padded K of group g, or with the PSUM layout its end K |
| `k_grouped_fp8_gemm_nt_contiguous` | per-group K-major `[M, k_g]` / `[N, k_g]` blocks stored back to back | padded K per group |

Group starts of contiguous layouts are multiples of `get_mk_alignment_for_contiguous_layout()`.
The default alignment, 256 (also `get_theoretical_mk_alignment_for_contiguous_layout()`), matches
the 256-row tiles; other multiples of 16 (for example the CUDA default 128) run the small-M
kernel's 16- to 64-row tiles. K-grouped layouts take any multiple of 128. With the PSUM layout
and `ensure_zero_padding=True` (the default) the rows from a group's end to the next aligned row
are stored as well: the 256-row tiles compute them from the zero A padding rows, the small-M
kernel stores zeros without reading them. K-grouped groups with K = 0 write C (or zero), and a
PSUM group whose K is not a multiple of 128 stops reading at its end.

Scales follow the CUDA SM100 forms: FP32 UE8M0 values (B of m-grouped GEMMs typically as
128x128 blocks) are packed per group; packed int32 scales are `[G, rows, words]` MN-major per
group for m-grouped operands, and `[sum of per-group words, rows]` for K-grouped operands, where
each group starts a new word.

Grouped GEMMs run the 256x256 MFMA pipeline of the dense GEMM, or the small-M kernel below for
few rows per group (`expected_m` of a masked layout, `expected_m_for_psum_layout` of a PSUM
layout, at most 64). m-grouped GEMMs use the persistent kernel when K is a multiple of 128 and
at least 256; the PSUM and masked layouts count and locate their tiles on the device, so no host
synchronization is needed. K-grouped GEMMs run one data-parallel tile per workgroup with the K
range of its group.

## Small M

GEMMs with up to 64 rows (32 for weights above 64 Mi elements) and K-major operands run
`fp8_gemm_mfma_16x16x128_small_m.cuh`: streaming the weights bounds their time, so every wave
loads its MFMA operands from global memory straight into a ring of registers, with no LDS and no
barrier in the K loop. The four waves of a workgroup split K and add their tiles in LDS in a
fixed order, which keeps results deterministic. Tiles are 16, 32 or 64 rows by 16 to 64 columns,
chosen from the rows and the number of workgroups the columns provide.

```bash
pytest -q tests/test_cdna4_grouped_fp8_gemm.py                  # reduced shapes
DG_TEST_FULL=1 pytest -q tests/test_cdna4_grouped_fp8_gemm.py   # shapes of tests/test_fp8_fp4.py
python tests/bench_cdna4_grouped_gemm.py --csv results/cdna4-grouped.csv
```

The benchmark times kernels with the profiler on the shapes of `tests/test_fp8_fp4.py` and, for
contiguous layouts, the dense FP8 GEMM on the same total problem. On the contiguous shapes the
grouped kernel reaches 0.93-1.02x of the dense kernel's speed.

## BF16 GEMMs

`bf16_gemm_{nt,nn,tn,tt}`, `m_grouped_bf16_gemm_{nt,nn}_contiguous` (per-row and PSUM layouts),
`m_grouped_bf16_gemm_nt_masked` and `k_grouped_bf16_gemm_tn_contiguous` keep the CUDA signatures
and run the same two kernels without scales. A K step still covers 128 bytes of every row
(64 BF16 values): each FP8 MFMA of the schedule becomes two 16x16x32 BF16 MFMAs on the two
halves of the same fragment, so tiles, the LDS layout and the pipeline are unchanged. MN-major
BF16 operands are read with 16-bit transposing LDS reads. Outputs, `alpha`, accumulation and
group layouts follow the FP8 GEMMs.

```bash
pytest -q tests/test_cdna4_bf16_gemm.py
python tests/bench_cdna4_grouped_gemm.py --dtype bf16
```

Against the PyTorch (hipBLASLt) BF16 GEMM, the dense kernel is 1.0-1.5x as fast for a single
row on six of the seven `forward` shapes of the FP8 benchmark below, 0.36-0.98x with 128 rows,
and 0.92-0.97x on large square shapes. Contiguous grouped GEMMs reach 0.97-0.99x of the dense
kernel on the same total problem, and masked GEMMs 0.92-2.4x of a per-group hipBLASLt loop.

## Einsum

`einsum` covers the BF16 expressions of the CUDA backend and `fp8_einsum` the FP8 ones, with BF16
or FP32 (accumulating) outputs:

| Expression | GEMM |
| --- | --- |
| `bmk,bnk->mn` | one GEMM whose K runs through the batches (`GemmType::KBatched`), split over the compute units for small outputs |
| `bhr,hdr->bhd` | a GEMM per head over strided views: A `[h, b, r]`, D `[h, b, d]` |
| `bhd,hdr->bhr` | as above, with an MN-major B `[h, r, d]` |
| `bhd,bhr->hdr` | a GEMM per head with K = b: MN-major A `[h, d, b]` and B `[h, r, b]` |

The per-head GEMMs run the masked GEMM kernels with full groups, so no operand is copied or
transposed; FP8 scales are permuted with their data and packed per head. FP8 outputs (the
`(d, sfd)` pair) and custom epilogues are not supported.

```bash
pytest -q tests/test_cdna4_einsum.py
```

The tests check the FP32 results against an FP32 reference and that BF16 outputs are those
results rounded once. Against `torch.einsum` (hipBLASLt) the expressions run at 0.4-1.45x; the
256x256 tiles make outputs with 128 columns or rows, and small K, the slow cases.

## Hyper-connection prenorm GEMM

`tf32_hc_prenorm_gemm(a, b, d, sqr_sum, num_splits=None)` computes `D = A @ B^T` and the rows' sums
of squares of BF16 A for an FP32 B with few rows (the mix outputs), whole or as `num_splits` K
partials. CDNA4 has no TF32 MFMA: `hc_prenorm_gemm.cuh` splits B into the BF16 terms
`hi = bf16(b)` and `lo = bf16(b - hi)` in registers and runs both against the exact BF16 A, which
keeps 16 mantissa bits of B (TF32 keeps 10). A workgroup streams 64 rows of A for every column;
its four waves split K and add their tiles in LDS in a fixed order, and when the rows alone leave
compute units idle, internal K splits are added in split order by a second kernel.

```bash
pytest -q tests/test_cdna4_hyperconnection.py
```

The tests compare against an FP64 reference with the CUDA test's bound. On the shapes of
`tests/test_hyperconnection.py` the kernel is 2.8-5.7x as fast as a PyTorch BF16 GEMM plus a sum of
squares for 4096 or more rows, and 1.0-2.5x for 13 and 137 rows.

## FP4 GEMMs

`fp8_fp4_gemm_{nt,nn,tn,tt}`, `m_grouped_fp8_fp4_gemm_{nt,nn}_contiguous`,
`m_grouped_fp8_fp4_gemm_nt_masked` (and their `fp4_*` aliases) take packed FP4 (E2M1) operands as
int8 tensors, two values per byte with the lower index in the low nibble, wherever they take FP8
operands: FP8 x FP4, FP4 x FP8 and FP4 x FP4, with 32-wide UE8M0 scale blocks for FP4 operands.
FP4 operands are K-major (`[MN, K / 2]`) or MN-major, packed along MN (`[MN / 2, K]` viewed
through its transpose, as `transpose_packed_fp4` produces). `k_grouped_fp4_gemm_nt_contiguous`
takes K-major `[M, sum_k / 2]` / `[N, sum_k / 2]` operands, recipe `(1, 1, 32)`, and a K alignment
that is a multiple of 128.

The scaled MFMA mixes the two formats directly, so FP4 operands keep their packed layout up to the
instruction: a K step still covers 128 K values, 64 bytes of an FP4 row (one 16-byte read per lane
and fragment instead of two), and MN-major FP4 tiles use `ds_read_b64_tr_b4`. FP8 x FP4 runs at
the FP8 MFMA rate with half the B traffic; FP4 x FP4 runs at the FP4 rate.

```bash
pytest -q tests/test_cdna4_fp4_gemm.py
python tests/bench_cdna4_fp4_gemm.py
```

On the shapes of the benchmark, with the weights read from memory: for a single row, FP8 x FP4 is
1.2-1.8x and FP4 x FP4 1.2-1.8x as fast as the FP8 GEMM; with 4096 rows, FP4 x FP4 is 1.15-1.4x as
fast as the FP8 GEMM and 1.0-2.0x as fast as the PyTorch MXFP4 GEMM, while the FP8 GEMM is 1.4-2.2x
as fast as the PyTorch MXFP8 GEMM.

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
