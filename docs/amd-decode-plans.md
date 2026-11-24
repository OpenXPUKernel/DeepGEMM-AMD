# Optional AMD decode plans

These opt-in paths preserve the existing tensor types, clipping, quantization
boundaries and reduction order. They are shape-specific implementations, not
performance guarantees for arbitrary workloads. CUDA dispatch is unchanged.

| Control | Behavior |
| --- | --- |
| `DG_CDNA4_SPECIALIZE_GEMM=1` | Compile exact shape/stride constants for normal and masked grouped GEMMs with at most 512 rows. Pointer addresses and weights remain runtime inputs. The constants are part of the JIT cache key. |
| `DG_MHC_MIX_THREADS=128` | Select the mix workgroup size; supported values remain 64, 128 and 256. |
| `DG_MHC_FUSED_FINISH=1` | With 128-thread mixes, put shifted mHC's independent new-mix and current-normalization work in one launch. Normal mHC keeps its dependency ordering. |
| `DG_GATE_FUSED_REDUCTION=1` | For `(M,N,K)=(500,384,5120)` or `(400,128,5120)`, consume the original BF16 GEMM split partials directly in gate selection. Other shapes retain the existing implementation. |
| `DG_GATE_PARTIAL_TILE=8,4,2` | Optional `FM,NF,DEPTH` plan inside the original gate split ranges. A single wave keeps each partial's K accumulation order; the gate then adds partials in the original split order. `off` retains the original partial GEMM. |

`FM` accepts 1, 2, 4 or 8; `NF` accepts 1, 2 or 4; `DEPTH` accepts 1 or 2.
Selecting a different partial tile does not change the split boundaries.
Validate logits, route IDs and normalized routing weights, not just selected
expert IDs. Gates with masks, image bias, fixed routing and random routing retain
those controls.

The resident MegaMoE scheduler now joins route writers at the metadata producer,
publishes the complete task table once, and lets the final return writer publish
completion. All original release/acquire and peer-completion requirements remain.
`DG_MEGA_MOE_SERIAL_STREAM=1` still requires the caller's exclusive serial
execution contract; it is not a switch for arbitrary concurrent callers.

The optional native Q-up artifact now bounds incomplete M tiles. Its C entrypoint
exports `qu_row_tails_supported()` so framework adapters can reject older
artifacts before graph capture. Full supported N tiles, operand alignment and
32-bit buffer-extent limits remain required. Rebuild this artifact together with
an adapter that enables its row-tail path.

Run CPU backend/build contracts separately from GPU tests. On a supported AMD
GPU, `python -m pytest -q tests/test_amd_decode_fastpaths.py` checks exact gate
partials, routing controls, mHC outputs, strided accumulation and graph replay
with changed inputs. Use the existing MegaMoE multi-process tests for communication
and shared-expert contracts. Timing comparisons should use the same actual
inputs, real expert distributions, shapes and graph boundaries; an isolated
kernel result does not establish a complete-model speedup.

### Routed expert return reduction

`DG_MEGA_MOE_WAVE_SLOT_REDUCE=3` enables the ordered wave reduction on supported bounded return buffers. A wave collects the signed expert IDs for one token. A missing slot uses an out-of-bounds offset in the bounded buffer descriptor, producing zero. The kernel adds BF16 slot values in the original top-k order in FP32, adds the shared expert last, and rounds the result to BF16. Leave the setting unset to use the existing reduction.

`DG_MEGA_MOE_ACTIVE_SLABS=3,1` specializes the routed up projection when at most 64 rows are live. The other matrix workers still participate in the original data publication and joins. These settings affect the AMD native resident implementation.


For sparse expert loads, `DG_MEGA_MOE_ACTIVE_SLABS=4,3` with
`DG_MEGA_MOE_SHORT_MFMA16=5` selects a 16- or 32-row matrix tile for either
routed projection when the live rows fit. Larger tiles retain the existing
matrix schedule. Short down projections consume the completed up projection.
Use random inputs and representative routed tensors to check both the final
outputs and receive counts before enabling a plan.

Explicit compact WV tiles support short row counts on their packed output
path. The general small-row planner remains available for ordinary BF16 output.
