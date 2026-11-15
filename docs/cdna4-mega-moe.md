# AMD Mega MoE

The public APIs match the CUDA backend: expert-parallel routing, two expert GEMMs
with SwiGLU, shared experts, and ordered FP32 accumulation on uni_symm buffers.
Aligned FP8 activations and packed FP4 routed weights use one cooperative kernel
for the complete call. BF16, FP8 weights and other supported layouts use the
general GEMM lowering. Routing is stable in source-token/slot order.

```python
buffer = deep_gemm.get_symm_buffer_for_mega_moe(
    group, num_experts, max_tokens, topk, hidden, intermediate,
    num_shared_experts=1, mma_type="fp8xfp4")
l1, l2 = deep_gemm.transform_weights_for_mega_moe(l1_weights, l2_weights)
l1, l2 = deep_gemm.prepare_weights_for_resident_mega_moe(l1, l2)
s1, s2 = deep_gemm.transform_weights_for_mega_moe(shared_l1, shared_l2)
buffer.x[:n].copy_(x_fp8)
buffer.x_sf[:n].copy_(x_sf)
buffer.topk_idx[:n].copy_(topk_idx)
buffer.topk_weights[:n].copy_(topk_weights)
deep_gemm.fp8_fp4_mega_moe(
    y, l1, l2, buffer, shared_l1_weights=s1, shared_l2_weights=s2,
    activation_clamp=10.0)
```

Interleave and prepare weights once at model loading, before graph capture.
The prepared-weight type explicitly owns its panel layout; scales stay unchanged.
Shared experts retain their ordinary FP8 layout. Ordinary packed-FP4 rows are also
accepted. Weight preparation never runs inside the MoE call.

G1 rounds to BF16 before clipping and SwiGLU. Routing weights precede intermediate
quantization. G2 rounds to BF16 before returning results; the final FP32 reduction
retains the original slot order. Gather readiness follows completion of every
payload writer, and all return writers join before the peer barrier and reduction.

| Layer | Source |
| --- | --- |
| Host API and dtype dispatch | `csrc/apis/mega_moe_hip.hpp` |
| Launch configuration | `csrc/jit_kernels/heuristics/cdna4_mega_moe.hpp` |
| JIT compilation and launch | `csrc/jit_kernels/impls/cdna4_fp8_fp4_mega_moe.hpp` |
| Cooperative kernel | `deep_gemm/include/deep_gemm/impls/cdna4_fp8_fp4_mega_moe.cuh` |
| GEMM and epilogue helpers | `deep_gemm/include/deep_gemm/impls/cdna4_mega_moe_*.cuh` |
| Symmetric layout and launch arguments | `deep_gemm/include/deep_gemm/layout/cdna4_mega_moe.cuh` |
| Readiness, epochs and gather | `deep_gemm/include/deep_gemm/scheduler/cdna4_mega_moe.cuh` |

The launch verifies that the complete grid can remain resident together. Every
replay initializes graph-private queues in the same kernel.

### Work distribution

The resident implementation accepts two bounded scheduling controls before
graph capture:

| Variable | Default | Range | Effect |
| --- | ---: | ---: | --- |
| `DG_MEGA_MOE_COLUMN_GROUPS` | 6 | 1–16 | Workgroups cooperating on each row tile's projections |
| `DG_MEGA_MOE_GATHER_WORKGROUPS` | 16 | 1–64 | Initial input-gather actors, which rejoin compute after draining gather work |

The complete-grid residency check remains enabled. These controls preserve
matrix operations, quantization boundaries, return ordering and synchronization.
Changing either setting requires recapturing an existing graph.

Tune with the same retained input tensors, weights, scales and routes. Compare
balanced random controls with actual model inputs: a plan that improves a
concentrated route distribution can be slower on uniformly distributed routes.
Validate the complete shared-expert call and the full serving cycle before
selecting a deployment-specific plan. The generic defaults remain unchanged.

Validation on AMD GPU:

```sh
python tests/test_cdna4_mega_moe.py --ranks 8
python tests/test_cdna4_mega_moe_schedule.py
python tests/test_moe_weight_panels.py
python tests/bench_cdna4_mega_moe.py --ranks 8
```

The graph test checks an FP32 oracle and independent general-GEMM reference,
empty ranks, concentrated and fully masked routing, tails, and eight asynchronous
replays with changed inputs, scales, routing and expert/shared weights. The
benchmark times the complete call and reports the median of rank maxima.

### Serial execution streams

The default native launch is cooperative so independent concurrent callers retain
the residency guarantee required by the software grid barriers. A serving engine
that owns the device and serializes all GPU work onto one execution stream can
explicitly set `DG_MEGA_MOE_SERIAL_STREAM=1` to use a normal HIP launch. The same
occupancy and grid-size checks still apply. This is a caller execution contract,
not a switch for overlapping independent streams or multiple GPU processes.

`DG_MEGA_MOE_PHASE_TIMESTAMPS=1` enables diagnostic device phase stamps. Leave it
unset for performance qualification. Phase stamps are not cross-device clocks.
