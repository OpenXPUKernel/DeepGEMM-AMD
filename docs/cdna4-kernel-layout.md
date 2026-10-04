# CDNA4 kernel layout

The HIP backend has one pybind extension and a small runtime JIT layer:

- `csrc/hip_python_api.cpp` validates tensors, asks the cost model for a configuration, and launches the cached HIPRTC module.
- `csrc/cdna4/cost_model.hpp` maps a GEMM shape to a tile, pipeline, and split-K specialization.
- `csrc/cdna4/jit_runtime.hpp` generates the wrapper source, compiles it with HIPRTC on first use, caches the code object, and launches it through the current PyTorch stream.
- `deep_gemm/include/deep_gemm/impls/cdna4/` contains the packaged device templates: the active FP8 pipeline, tiled kernels, shared device helpers, and the plain `GemmArgs` ABI.
- `deep_gemm/cdna4_fp8_gemm.py` keeps quantization, scale packing, layout aliases, and the public Python interface aligned with the CUDA backend.

The packaged headers are organized by responsibility:

| Header | Responsibility |
| --- | --- |
| `fp8_gemm.cuh` | Pipeline using 16×16×128 scaled MFMA; data-parallel, persistent, and split-K execution |
| `gemm_tiled.h` | The two tiled configurations selected by the cost model |
| `epilogue.cuh` | Output stores, accumulation, and boundary handling |
| `split_k_reduce.cuh` | Deterministic reduction of split-K partials |
| `tile_scheduler.cuh` | Output-tile ordering shared by both GEMM families |
| `device_common.h` | Vector types, numeric conversions, and memory primitives |
| `gemm_args.h` | Argument layout shared by the host and generated device code |

The JIT includes the selected GEMM header and adds the reduction header for split-K. Public layout aliases normalize inputs to K-major before launching these kernels.

For one output tile, the device code follows this data path:

```text
A/B and scales in global memory
        │ DMA
        ▼
      LDS stage 0/1
        │ LDS reads
        ▼
   MFMA register fragments
        │ scaled MFMA
        ▼
    BF16/FP32 epilogue
```

The cost model selects among 128/256 tiled kernels, the pipelined data-parallel kernel, persistent execution, and split-K. The selected template wrapper is compiled once per configuration and output type, then reused for later shapes that select the same specialization.

The normal `develop.sh`/`setup.py` build compiles the pybind extension. No separate HIP shared library is required.
