# CDNA4 kernel layout

The HIP backend has one pybind extension and a small runtime JIT layer:

- `csrc/hip_python_api.cpp` validates tensors, asks the cost model for a configuration, and launches the cached HIPRTC module.
- `csrc/cdna4/cost_model.hpp` maps a GEMM shape to a tile, pipeline, and split-K specialization.
- `csrc/cdna4/jit_runtime.hpp` generates the wrapper source, compiles it with HIPRTC on first use, caches the code object, and launches it through the current PyTorch stream.
- `deep_gemm/include/deep_gemm/impls/cdna4/` contains the packaged device templates: the production FP8 pipeline, shared device helpers, and the plain `GemmArgs` ABI.
- `deep_gemm/_hip_api.py` keeps quantization, scale packing, layout aliases, and the public Python interface aligned with the CUDA backend.

The packaged headers are organized by responsibility:

| Header | Responsibility |
| --- | --- |
| `fp8_gemm_mfma_16x16x128.cuh` | 16×16×128 scaled MFMA, direct global-to-LDS DMA, and interleaved instruction scheduling; data-parallel, persistent, and split-K execution |
| `epilogue.cuh` | Output stores, accumulation, and boundary handling |
| `split_k_reduce.cuh` | Deterministic reduction of split-K partials |
| `tile_scheduler.cuh` | Output-tile ordering for the production execution modes |
| `device_common.h` | Vector types, numeric conversions, and memory primitives |
| `gemm_args.h` | Argument layout shared by the host and generated device code |

The MFMA dimensions describe one instruction, not the workgroup tile. The
current production implementation uses 256×256 output tiles and three execution
modes: data-parallel, persistent, and split-K. The planner accounts for reduction
overhead and available parallelism; split counts are normalized so every
workgroup gets a nonempty K range. Persistent mode additionally checks output
alignment, row stride, dtype and accumulation.

The generated wrapper specializes the execution mode, output dtype, both operand
layouts, and both scale granularities. All these choices appear in the in-process
JIT cache key. Kernels directly consume K-major or MN-major operands and packed
32-/128-wide scale blocks. Each DMA constructs a bounded descriptor with a
64-bit base, so tensors can span more than 4 GiB. MN-major fragments use
transposing LDS reads; no input transpose or alignment copy is launched.

For one output tile, the device code follows this data path:

```text
A/B and scales in global memory
        │ direct DMA with per-step 64-bit buffer bases
        ▼
      LDS stage 0/1
        │ LDS reads
        ▼
   MFMA register fragments
        │ scaled MFMA
        ▼
    BF16/FP32 epilogue
```

FP32 scales are still broadcast and packed by the Python preparation helper.
Valid packed scales pass through unchanged, including their original K block
width. C is copied into D only when their row strides differ. Split-K workspace
contains exactly the required FP32 partials; it is not zero-filled.

The normal `develop.sh`/`setup.py` build compiles the pybind extension. No separate HIP shared library is required.
