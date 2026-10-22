# AMD GPU Mega MoE

`fp8_fp4_mega_moe` and `bf16_mega_moe` run an expert-parallel MoE layer across the GPUs of a node
with the CUDA backend's interface: the tokens of every rank are dispatched to the ranks of their
top-k experts, the experts' two GEMMs run with SwiGLU in between, the results are combined back on
the tokens' ranks, and shared experts are added. `get_symm_buffer_for_mega_moe` allocates the
symmetric buffer through [uni_symm](../third-party/uni_symm/README.md) and exposes the same views
(`x`, `x_sf`, `topk_idx`, `topk_weights`, ...); the caller fills the inputs before each call.

```python
buffer = deep_gemm.get_symm_buffer_for_mega_moe(group, num_experts, num_max_tokens_per_rank, num_topk,
                                                hidden, intermediate_hidden, num_shared_experts=1,
                                                mma_type="fp8xfp4")
l1, l2 = deep_gemm.transform_weights_for_mega_moe(l1_weights, l2_weights)
buffer.x[:n].copy_(x_fp8); buffer.x_sf[:n].copy_(x_sf)
buffer.topk_idx[:n].copy_(topk_idx); buffer.topk_weights[:n].copy_(topk_weights)
deep_gemm.fp8_fp4_mega_moe(y, l1, l2, buffer, shared_l1_weights=s1, shared_l2_weights=s2,
                           cumulative_local_expert_recv_stats=stats, activation_clamp=10.0)
```

Weights are `[experts per rank, 2 intermediate, hidden]` (L1) and `[experts per rank, hidden,
intermediate]` (L2): FP8 or packed FP4 with UE8M0 scales of 32 values in the layout of
`transform_sf_into_required_layout`, or BF16. `transform_weights_for_mega_moe` interleaves the gate
and up rows of L1 by 8, as on CUDA, and keeps the scales in the AMD GPU layout (the CUDA transform's
scale reordering is specific to its tensor-core loads). The SwiGLU is CUDA's: the BF16 gate and up
values are clamped (`gate <= clamp`, `|up| <= clamp`), `silu(gate) * up * topk_weight` is cast to FP8
with UE8M0 scales (CUDA's exponent rounding) or rounded to BF16, and fast math uses the hardware
exponential and reciprocal.

## Kernels

`DG_MEGA_MOE_DETERMINISTIC_ROUTE=1` optionally assigns each source's expert rows
in ascending token/slot order with integer workgroup prefix counts. The default
remains atomic ticket assignment. This option does not change gate selections,
weights or quantization, and does not by itself guarantee identical floating
outputs across different GEMM shapes. It currently supports the standard path,
not the opt-in queue pipelines; conflicting settings raise an error. Set it
before graph capture. `tests/test_cdna4_stable_route.py` checks row positions,
masked/empty inputs, wave/workgroup boundaries and dynamic graph replay.

`mega_moe.cuh` runs a call in stream order on every rank:

| Kernel | Work |
| --- | --- |
| route | every (token, slot) takes a position among its expert's rows from this rank |
| barrier (publish) | this rank's counts go to every rank; from all counts each rank knows its rows in every expert's block (experts start at multiples of 256 rows, sources follow in rank order) and its own experts' end rows |
| push | one wave per token writes its row, and a record of its scales, top-k weight and source, into the rows of its slots on the experts' ranks |
| barrier | every row arrived |
| finish | the records' scales become the MN-major layout of the grouped GEMM |
| grouped GEMMs | L1 over the experts' rows (the PSUM layout of their end rows), SwiGLU, L2 |
| combine | FP8/mixed L2 GEMMs with full-width tiles write directly into each token's slot on its source rank; other shapes use the standalone combine |
| barrier, reduce | every result arrived; `y` is the sum of the token's slots and the shared experts in FP32 |

The direct-combine path builds a per-row output-address table in `finish`. The L2 epilogue stages
BF16 through LDS as before, but coalesced stores target the source GPU instead of a local scratch
followed by a separate copy. A system-scope release at the end of every producer workgroup and
the final cross-rank barrier precede the reduction's system-scope acquire. Partial M tiles keep
their bounds checks. Hidden sizes not divisible by 256 and short K retain the standalone path.
Generic grouped GEMMs are unchanged; Mega MoE uses groups of four M tiles for scheduling.

The shared experts' GEMMs and SwiGLU run on a side stream joined before the reduction. Readers of
memory written by peers acquire at system scope first: every XCD's L2 is invalidated by its own
acquire. A row's position within its expert does not change its result (the grouped GEMMs are
row-independent), so results are bitwise repeatable although the route takes positions with
atomics. The barriers are device kernels, so a call can be captured in a CUDA graph.

Runtime correction (2026-10-16): the optional expert-window stream masks must not be interpreted
as verified CU isolation, especially inside an outer graph. See the
[measured CU sets and graph limitation](cdna4-kernel-layout.md).

Two properties of the fabric shape the dispatch. Remote reads reach about half the all-to-all rate
of remote writes, so tokens are pushed rather than pulled. And the push and combine are bound by
the fabric: they reach the same rate with an eighth of the compute units as with all of them, and
they visit the peers in rotation (the combine iterates rows source-rank-major) so every link is
busy.

## Tests and performance

```bash
python tests/test_cdna4_mega_moe.py              # every GPU of the node; --ranks N
python tests/bench_cdna4_mega_moe.py --relative-only
```

The test compares `y` of every rank with an FP32 reference of its own tokens (each rank builds all
experts from one seed) that rounds where the kernels do, checks the received-row statistics
against all ranks' top-k, and checks bitwise repeatability, for fp8xfp4, fp8xfp8 and bf16xbf16 with
masked slots, a shared expert, and a rank without tokens. The host exchanges go through gloo.

The benchmark reports complete-call latency and useful, unpadded routed/shared expert GEMM FLOPs
summed across ranks. It does not assume or publish a GPU model or hardware peak specification.
Peak-normalized metrics are omitted unless the caller explicitly supplies `--peak-tflops` for
their own private measurements. `--graph` includes every layer stage in graph replay.
`--json-output` records shapes, timing, effective FLOPs, and overrides. If an explicit
denominator is supplied, `--target-mfu` controls the optional target-latency calculation.
`--compare-serial` checks the full benchmark workload bitwise against the standalone path before timing.

The default 8-rank shape has 8192 tokens/rank, hidden 7168, intermediate 3072, top-6 of 384 experts,
and one shared expert. Hardware-peak-derived targets are not part of the public record.
The current no-shared-expert workload has a different FLOPs numerator and is
reported separately in [AMD validation](amd-validation.md). The cooperative
pipeline and other scheduling experiments remain opt-in; they do not change
the default implementation.

## Experimental paths and regression controls

All ranks must use identical options. Defaults select one expert window, the original token-major
dispatch, standalone SwiGLU, and direct combine for eligible FP8/mixed shapes. BF16 keeps its
standalone combine unless requested. Options are read before enqueueing kernels, including capture.

| Environment variable | Behavior |
| --- | --- |
| `DG_MEGA_MOE_DIRECT_COMBINE=0/1` | Disable/enable direct peer stores from L2; incompatible shapes fall back |
| `DG_MEGA_MOE_TILE_GROUP_M=1/2/4/8/16` | M-tile scheduling group for MoE; default 4 |
| `DG_MEGA_MOE_CHUNKS=N` | Expert windows on dispatch/compute/combine streams; default 1 |
| `DG_MEGA_MOE_CUMASK=1` | Experimental AMD GPU stream partition for supported topologies |
| `DG_MEGA_MOE_COMM_CUS=N` | Reserve 8–128 CUs in multiples of eight; default 32 |
| `DG_MEGA_MOE_ORDERED_PUSH=1` | Destination/expert-major send lists instead of scanning tokens per window |
| `DG_MEGA_MOE_FUSE_SWIGLU=1` | Fuse BF16 rounding, clamp/SwiGLU and FP8/BF16 cast into L1's LDS epilogue |

The last three algorithmic experiments are opt-in; their composition was not faster than the
selected default on the measured shape. Separate dispatch/combine barrier phases are used for
multi-stream launches. The tests compare experimental and standalone paths bitwise, and cover
graph replay, empty experts, entirely masked routes and routing concentrated on one rank:

```bash
DG_MEGA_MOE_TEST_GRAPH=1 DG_MEGA_MOE_TEST_SKEW=1 DG_MEGA_MOE_COMPARE_SERIAL=1 \
  python tests/test_cdna4_mega_moe.py
```

`--gemm-ceiling` measures routed GEMMs separately as a diagnostic and labels them explicitly;
those measurements are never substituted into the end-to-end MFU. Intermediate storage still
uses worst-case sizing. A bounded-memory persistent pipeline remains unfinished.

An NV-style resident block pipeline is now available through
`DG_MEGA_MOE_NV_PIPELINE=1`, default off. It overlaps dispatch, routed expert
compute and output writeback, but the controlled full workload regresses from
5.42 ms to 6.76 ms. See the [pipeline correctness regression](../tests/test_cdna4_mega_moe_pipeline.py).

For the matched **no-shared-expert** EP8 workload (8192 tokens/rank, H7168,
I3072, 384 experts, top-6, FP8×FP4, BF16 return), the newer
`DG_MEGA_MOE_STAGE1_QUEUE=1` path lowers end-to-end latency by 10.36–10.86%
against aiter's matched BF16 generic Stage2 configuration across three seeds
and a main-tree check. See [current validation and reproduction](amd-validation.md).
This is a separate workload from the earlier shared-expert benchmark.
