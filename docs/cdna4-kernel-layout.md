# AMD GPU kernel layout

The HIP backend has one pybind extension, a kernel-specific planner, and the shared DeepJIT HIP runtime:

- `csrc/hip_python_api.cpp` validates tensors, asks the cost model for a configuration, and launches the cached HIPRTC module.
- `csrc/cdna4/cost_model.hpp` maps a GEMM shape to a tile, pipeline, and split-K specialization.
- `csrc/cdna4/jit_runtime.hpp` generates GEMM wrappers, plans launch dimensions, and maintains a per-device specialization lookup. It delegates compilation, code-object caching, loading, and native launches to `deep_jit::Runtime<deep_jit::HIP>`.
- `third-party/deep_jit_hip/include/deep_jit/backend/hip/` owns the HIPRTC compiler, module lifetime, exported entry points, and regular/cooperative launches. DeepJIT tracks include contents and compiler options for its cross-process disk cache. The original `third-party/deep_jit` pin remains independent for CUDA compatibility.
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
| `mega_moe.cuh`, `mega_moe_args.h` | Mega MoE across the GPUs of a node (uni_symm buffers): routing, push dispatch, SwiGLU, combine and reduction around the grouped GEMMs |
| `moe_epilogue.cuh` | Optional L1 GEMM epilogue consuming BF16 gate/up LDS tiles for SwiGLU and quantization; L2 direct peer stores are in the GEMM's persistent epilogue |
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

## Shared HIP JIT

The HIP backend uses the same DeepJIT `Runtime`, `Config`, parser and disk cache
as the other backends. `DG_JIT_CACHE_DIR` selects a writable cache (or a colon-
separated writable root and read-only lookup roots); `DJ_JIT_CACHE_DIR` is the
fallback. Cache entries contain HIP source, HSACO, exported names and metadata.
The default location is `$HOME/.dj`. A process-local runtime is created lazily
for each current device. `deep_gemm._C.get_jit()` exposes that runtime.

The GEMM front cache includes the specialization, exported names, compiler
options and device. The persistent cache additionally hashes actual source,
tracked `deep_gemm/` and `uni_symm/` includes, and the HIPRTC/runtime version.
Device includes use canonical angle-bracket paths so changes invalidate the
cache correctly. Configuration and include digests are process snapshots;
restart the process after editing device headers.

The consumer retains launch planning, occupancy checks, split-K workspace and
cooperative residency decisions. Runtime access is serialized, releasing the
Python GIL before acquiring the cache mutex. All compilation must be warmed up
before graph capture; loaded modules stay alive for the process lifetime.
