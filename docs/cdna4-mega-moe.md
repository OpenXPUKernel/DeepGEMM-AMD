# CDNA4 Mega MoE

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
and up rows of L1 by 8, as on CUDA, and keeps the scales in the CDNA4 layout (the CUDA transform's
scale reordering is specific to its tensor-core loads). The SwiGLU is CUDA's: the BF16 gate and up
values are clamped (`gate <= clamp`, `|up| <= clamp`), `silu(gate) * up * topk_weight` is cast to FP8
with UE8M0 scales (CUDA's exponent rounding) or rounded to BF16, and fast math uses the hardware
exponential and reciprocal.

## Kernels

`mega_moe.cuh` runs a call in stream order on every rank:

| Kernel | Work |
| --- | --- |
| route | every (token, slot) takes a position among its expert's rows from this rank |
| barrier (publish) | this rank's counts go to every rank; from all counts each rank knows its rows in every expert's block (experts start at multiples of 256 rows, sources follow in rank order) and its own experts' end rows |
| push | one wave per token writes its row, and a record of its scales, top-k weight and source, into the rows of its slots on the experts' ranks |
| barrier | every row arrived |
| finish | the records' scales become the MN-major layout of the grouped GEMM |
| grouped GEMMs | L1 over the experts' rows (the PSUM layout of their end rows), SwiGLU, L2 |
| combine | each row's result is written into its slot of its token on the source rank |
| barrier, reduce | every result arrived; `y` is the sum of the token's slots and the shared experts in FP32 |

The shared experts' GEMMs and SwiGLU run on a side stream joined before the reduction. Readers of
memory written by peers acquire at system scope first: every XCD's L2 is invalidated by its own
acquire. A row's position within its expert does not change its result (the grouped GEMMs are
row-independent), so results are bitwise repeatable although the route takes positions with
atomics. The barriers are device kernels, so a call can be captured in a CUDA graph.

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

On the default shapes of `tests/test_mega_moe.py` (8192 tokens per rank, top-6 of 384 experts and a
shared expert, 8 ranks), the push and combine move rows at about 0.8 of the node's all-to-all
remote-write rate, measured with the same row size. In this version the stages run one after
the other: the routed expert GEMMs and the push plus combine take about 40% of a call each. Next are
overlapping them (a compute-unit partition between a communication and a compute stream,
pipelined over groups of experts) and a persistent kernel that fuses the stages; the activation
buffers are sized for the worst case (every token on every rank), which the pipelined version
will turn into a ring.
