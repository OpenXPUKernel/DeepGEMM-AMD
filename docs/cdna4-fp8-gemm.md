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

Run the relative performance script:

```bash
python tests/bench_cdna4_fp8.py
```

The script prints only the acceleration ratio against the PyTorch matmul reference, for example:

```text
shape=(128,7168,4096) speedup_vs_pytorch=<ratio>x
shape=(4096,7168,4096) speedup_vs_pytorch=<ratio>x
shape=(4096,2048,7168) speedup_vs_pytorch=<ratio>x
```

The ratio values are produced on the target CDNA4 system when the script is run; no absolute throughput or device configuration is required for reporting.
