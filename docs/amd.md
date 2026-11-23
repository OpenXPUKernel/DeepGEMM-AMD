# AMD GPU backend integration

The default ROCm backend now uses the native CDNA4 implementation from
`feat/cdna4-all-kernels` (integration input `39504df`). It builds a HIP host
extension and JIT-compiles device code with HIPRTC. See
[the kernel layout](cdna4-kernel-layout.md), [GEMM](cdna4-fp8-gemm.md),
[MQA logits](cdna4-mqa-logits.md) and [MegaMoE](cdna4-mega-moe.md).

The earlier Triton-only NT baseline is superseded, not selected as a silent
fallback. No external vendor runtime dependency is required. CUDA continues to
load the original initialization and kernels from `_cuda_api.py`.

Additional regression coverage inherited from the initial NT baseline:

- `bf16_gemm_nt`, with BF16/FP32 output and optional C accumulation;
- `fp8_fp4_gemm_nt` and its FP8/FP4 aliases, including mixed operand formats;
- per-32/per-128 UE8M0 scaling and packed int32 scale tensors;
- FP32 scale packing and row broadcasting, including grouped scale tensors;
- strided operands/scales/output, alpha, C accumulation and graph capture.

The default scaling recipe remains the upstream legacy per-128 recipe.
Specify per-32 recipes explicitly; packed tensor shapes alone cannot tell
the two formats apart. FP4 checkpoint weights use packed int8 storage.

Grouped GEMM, MegaMoE, attention index logits, symmetric buffers,
hyperconnections and non-NT entry points are supplied by the CDNA4
implementation. Full DeepInfer integration and model-level correctness remain
separate validation gates.

Development matrix: AMD GPU, ROCm 7.2, PyTorch 2.11.
Build/test in a ROCm PyTorch environment:

```bash
python setup.py bdist_wheel
python -m pytest tests/amd/test_gemm.py -q
```

Native matrix accumulation is not bitwise equal to an FP64 reference. Tests
bound relative L2 error against dequantized FP64 operands to 5e-5 and also
check maximum absolute error relative to the output magnitude. These are
kernel-level gates, not proof of unchanged model-level accuracy. Full-model
logits, generation and evaluation must still be compared before deployment.

Optional fixed-contract kernels live under `csrc/amd/selected_*.hip`.
They expose only the selected projection, mHC and Q-up configurations through
a small C ABI. General GEMM and MegaMoE dispatch remain separate. The Q-up
specialization's extended arguments and mHC's cache-domain implementation are
isolated from the general runtime headers; unsupported requests fail rather
than selecting an experimental variant. DeepInfer's
`scripts/build_amd_artifacts.py` builds these libraries and the integrity
manifest consumed by its guarded model adapters.

### Explicit small-row GEMM plans

Normal AMD GPU GEMMs with at most 512 rows and MN-major mode zero can opt into the existing small-row kernel with DG_CDNA4_GEMM_TUNE=FM,NF,DEPTH,WAVES. FM is 1/2/4/8, NF is 1/2/4, pipeline depth is 1..4 and waves is 1/2/4; configurations exceeding the kernel LDS limit are rejected. Set the variable to off to force normal planning.

DG_CDNA4_GEMM_TUNE_MAP selects exact contracts: m,n,planner_k,a_type,b_type,out_dtype=FM,NF,DEPTH,WAVES, separated by semicolons. Operand types are FP8=0, FP4=1, BF16=2; output types are BF16=0, FP32=1. planner_k is twice logical K for BF16 A, otherwise logical K. Entries apply only without an additive C operand. For example, 128,128,1024,2,2,0=2,2,2,4 targets a BF16 [128,512] by [128,512] NT product. Unmatched shapes retain normal planning. The global variable, including off, overrides the map; unset it when selecting per-shape plans. The table is limited to 32768 bytes.

Set plans before graph capture. They change scheduling and may change reduction order, so compare numerical outputs and measure the complete workload on matched random and real inputs before deployment. No new plan is enabled by default, and no performance guarantee or cross-backend hardware qualification is implied.

For speculative decode, `DG_CDNA4_DECODE_BATCH` opts the compact projection and resident routing schedules into the exact target and draft row counts of that per-rank batch. Its default specialized batch is 100. The normal planner handles other row counts. The route bitmap is checked against the available shared memory and the fused gate must cover every pair of rows in its launch grid.

The resident grid's epoch and diagnostic state occupy a dedicated region of the symmetric buffer. All configurations sharing a base allocation use the same offset, available as `layout["resident_control"]`. Rebuild the host extension with the matching device headers when changing revisions so both use the same layout.
