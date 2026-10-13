# CDNA4 MQA logits

The HIP backend implements the V3.2 indexer kernels with the CUDA signatures:
`fp8_fp4_mqa_logits`, `fp8_fp4_paged_mqa_logits`, `fp8_fp4_sparse_mqa_logits`,
`fp8_fp4_paged_sparse_mqa_logits` and their metadata functions `get_mqa_logits_metadata`,
`get_paged_mqa_logits_metadata`, `get_sparse_mqa_logits_metadata` and
`get_paged_sparse_mqa_logits_metadata`. For a query token `i` and a KV row `n`,

```text
logits[i, n] = sum_h weights[i, h] * relu(q[i, h] . kv[n])
```

Both input formats of the CUDA backend are accepted, chosen by whether Q carries scales:

| Format | Q / KV | Scales | Weights, logits |
| --- | --- | --- | --- |
| FP8 (as on SM90) | E4M3 | none for Q, one float per KV token (it multiplies the logit) | float |
| MXFP8 (as on SM100) | E4M3 | UE8M0 per 32 values, packed into one int32 per head / token | BF16 |
| MXFP4 (as on SM100) | packed E2M1 | as MXFP8 | BF16 |

Head dims of 32, 64 or 128 values (64 or 128 with FP4) and any number of heads are supported.
The paged functions take the fused cache of the CUDA backend (a page holds its rows, then their
scales), 32 or more rows per page, both `[batch, next_n]` requests and varlen requests given by
`indices` (one token per row, block-table rows per token). Rows past a token's KV span are not
written, as in the CUDA backend.

The metadata tensors are this backend's own: KV ranges of groups of tokens (contiguous), the
first token of each token's request (paged, varlen), and for the sparse functions the tokens' KV
spans plus the first KV row of every selected block (paged: page and row, resolved through the
block table), which the sparse kernels read instead of the index and block tables. They are built
with tensor operations on the device, without synchronization, and may be captured in graphs.

## Kernel

`mqa_logits.cuh` computes the dot products with 16x16x128 scaled MFMAs. The (token, head) pairs
of a group of tokens are the rows of up to 16 MFMA tiles; a KV row is a column. The rows are
ordered so that the heads of a token land in the result registers of one lane, or of two or four
lane groups when a group has fewer than four tokens (decoding): the weighted ReLU sum then runs
inside the lanes, with packed FMAs, plus a butterfly over the lane groups. Every source adds a
token's heads in the same order, so sparse logits equal the full logits bit for bit.

A workgroup of four waves stages the Q tiles once in LDS, in operand order, and its waves take
quarters of a chunk of KV rows. Each wave loads its KV rows from memory straight into registers,
one step of 64 rows ahead; paged and sparse rows are found through the chunk's row table, which
the wave loads one chunk ahead and keeps in LDS. Short KV ranges get shorter chunks, so that
small batches still occupy every compute unit.

```bash
pytest -q tests/test_cdna4_mqa_logits.py                  # reduced shapes
DG_TEST_FULL=1 pytest -q tests/test_cdna4_mqa_logits.py   # shapes of tests/test_attention.py
python tests/bench_cdna4_mqa_logits.py --relative-only
```

The tests use the references and tolerances of `tests/test_attention.py` for all three formats,
compare the scheduled and unscheduled paths and repeated calls bit for bit, and check that sparse
logits (contiguous with aligned and unaligned starts, and paged) equal the full logits at every
selected position.

On the prefill shapes of `tests/test_attention.py` (causal, 64 heads of 128 values), the FP8
kernel is 1.5-2.0x as fast as aiter's Triton FP8 MQA logits kernel; MXFP8 runs at the speed of
FP8, MXFP4 1.25-1.3x faster. Decoding with paged KV (one token per request) reads the KV cache at
0.65-0.95x the memory traffic rate of a large device copy with FP8 and MXFP8 caches, and
0.45-0.95x with MXFP4 caches (half the bytes per row).
