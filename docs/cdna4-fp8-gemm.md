# CDNA4 FP8 GEMM NT

This backend adds a HIP implementation of DeepGEMM's dense FP8 GEMM interface for CDNA4 architecture GPUs. The implementation is marked as a KnowingNothing implementation and keeps the CUDA call signature for `fp8_gemm_nt`, `fp8_gemm_nn`, `fp8_gemm_tn`, and `fp8_gemm_tt`.

The supported operation is

```text
D = alpha * (A @ B.T) + C
```

`A` and `B` contain OCP FP8 E4M3 values. Their scale tensors use UE8M0 values with 32 elements per K block. `recipe_a=(1, 32)` and `recipe_b=(1, 32)` are the direct MXFP8 form; a 128-element recipe is expanded internally. Scales can be passed as float32 tensors or as the packed int32 layout returned by `transform_sf_into_required_layout`. `D` and `C` can be BF16 or FP32. Dense GEMMs and the `nn`/`tn`/`tt` aliases are supported; grouped GEMMs and custom epilogues are not part of this backend.

Build the HIP shared library from the repository root:

```bash
./develop.sh
```

The normal DeepGEMM build flow compiles the pybind host extension into `deep_gemm._C`. On first use, the extension asks the cost model for a template specialization, compiles that wrapper with HIPRTC, and caches the resulting code object in the process. The kernel organization is documented in [the CDNA4 kernel layout guide](cdna4-kernel-layout.md). A ROCm wheel includes the same extension automatically. After the library is built, importing `deep_gemm` on a ROCm installation selects the HIP backend automatically. CUDA installations continue to use the existing CUDA backend.

Run correctness tests on a CDNA4 architecture GPU:

```bash
pytest -q tests/test_cdna4_fp8_gemm.py
```

The tests compare the operator with a PyTorch dequantize-and-matmul reference, check accumulation and `alpha`, exercise packed scales and layout aliases, cover empty M/K cases, and verify repeatability.

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
