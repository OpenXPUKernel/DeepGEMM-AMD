# AMD GPU backend integration

The default ROCm backend now uses the native AMD GPU implementation from
`feat/cdna4-all-kernels` (integration input `39504df`). It builds a HIP host
extension and JIT-compiles device code with HIPRTC. See
[the kernel layout](cdna4-kernel-layout.md), [GEMM](cdna4-fp8-gemm.md),
[MQA logits](cdna4-mqa-logits.md) and [MegaMoE](cdna4-mega-moe.md).

The earlier Triton-only NT baseline is superseded, not selected as a silent
fallback. There is no runtime dependency on AITER or FlyDSL. CUDA continues to
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
hyperconnections and non-NT entry points are supplied by the AMD GPU
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
