# CDNA4 kernel layout

The HIP backend has one pybind extension and a small runtime JIT layer:

- `csrc/hip_python_api.cpp` validates tensors, asks the cost model for a configuration, and launches the cached HIPRTC module.
- `csrc/cdna4/cost_model.hpp` maps a GEMM shape to a tile, pipeline, and split-K specialization.
- `csrc/cdna4/jit_runtime.hpp` generates the wrapper source, compiles it with HIPRTC on first use, caches the code object, and launches it through the current PyTorch stream.
- `csrc/cdna4/mqa_logits.hpp` plans the MQA logits kernels (tokens per wave, chunk of KV rows) and launches them.
- `deep_gemm/include/deep_gemm/impls/cdna4/` contains the packaged device templates: the production FP8 pipeline, shared device helpers, and the plain `GemmArgs` ABI.
- `deep_gemm/_hip_api.py` keeps quantization, scale packing, layout aliases, and the public Python interface aligned with the CUDA backend.

The packaged headers are organized by responsibility:

| Header | Responsibility |
| --- | --- |
| `fp8_gemm_mfma_16x16x128.cuh` | 16×16×128 scaled MFMA on FP8 / FP4 operands (or two 16×16×32 BF16 MFMAs per 128-byte K step), direct global-to-LDS DMA, and interleaved instruction scheduling; data-parallel, persistent, and split-K execution |
| `fp8_gemm_mfma_16x16x128_small_m.cuh` | Small-M FP8 / FP4 and BF16 GEMMs: operands streamed from global memory into registers, K split across the waves of a workgroup; dense and m-grouped |
| `hc_prenorm_gemm.cuh` | Hyper-connection prenorm GEMM: BF16 A against FP32 B as two BF16 terms, with the rows' sums of squares |
| `mqa_logits.cuh` | MQA logits (contiguous, paged and sparse KV): scaled-MFMA dot products with the weighted ReLU sum over the heads in the lanes |
| `mqa_logits_args.h` | Argument layout, formats and KV sources of the MQA logits kernels |
| `mega_mhc.cuh` | Mega mHC: the residual update with the mix GEMM from registers, the mixes (sigmoids, Sinkhorn), and the RMSNorm with the FP8 cast |
| `mega_gate.cuh`, `mega_gate_args.h` | MoE gate routing from FP32 logits: top-k by wave reductions, weights, shared experts, physical replicas and the routing controls |
| `epilogue.cuh` | Output stores, accumulation, and boundary handling |
| `split_k_reduce.cuh` | Deterministic reduction of split-K partials |
| `tile_scheduler.cuh` | Output-tile ordering for the execution modes, and the mapping of tiles onto the groups (and K ranges) of grouped GEMMs |
| `device_common.h` | Vector types, numeric conversions, and memory primitives |
| `gemm_args.h` | Argument layout and grouped GEMM types shared by the host and generated device code |

The MFMA dimensions describe one instruction, not the workgroup tile. The
current production implementation uses 256×256 output tiles and three execution
modes: data-parallel, persistent, and split-K. The planner accounts for reduction
overhead and available parallelism; split counts are normalized so every
workgroup gets a nonempty K range. Persistent mode additionally checks output
alignment, row stride, dtype and accumulation.

The generated wrapper specializes the execution mode, grouped GEMM type, both operand types (FP8,
FP4 or BF16), output dtype, both operand layouts, and both scale granularities. All these choices appear in the in-process
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
