"""Correctness tests for the CDNA4 MQA logits (contiguous, paged and sparse KV) in the FP8 format
(E4M3 Q, per-token float KV scales) and the MXFP8 / MXFP4 formats, with the references, input
generators and tolerances of tests/test_attention.py.
"""
import os
import random
import sys

import pytest

try:
    import torch
except Exception as exc:  # pragma: no cover - collection-only environments
    pytest.skip(f"PyTorch is required: {exc}", allow_module_level=True)

if torch.version.hip is None or not torch.cuda.is_available():
    pytest.skip("CDNA4 tests require a ROCm device", allow_module_level=True)

sys.path.insert(0, os.path.dirname(__file__))

import deep_gemm
from deep_gemm.testing import calc_diff
from deep_gemm.utils import ceil_div, per_custom_dims_cast_to_fp8, per_token_cast_to_fp4, per_token_cast_to_fp8
import test_attention as ref

FULL = os.environ.get("DG_TEST_FULL") == "1"
FORMATS = [("fp8", torch.float), ("mxfp8", torch.bfloat16), ("mxfp4", torch.bfloat16)]


@pytest.fixture(autouse=True)
def _seed():
    random.seed(0)
    torch.manual_seed(0)


def _check(logits, ref_logits, simulated_logits, valid, fmt):
    logits = logits.float().masked_fill(~valid, 0)
    diff_tol, simulated_tol = ref.mqa_logits_tolerances(fmt)
    assert calc_diff(logits, ref_logits.masked_fill(~valid, 0)) < diff_tol
    assert calc_diff(logits, simulated_logits.masked_fill(~valid, 0)) < simulated_tol


CONTIGUOUS = ([(s, kv, h, d) for s in (2048, 8192) for kv in (8192, 65536) for h in (8, 12, 16, 20, 32, 64)
               for d in (32, 64, 128)] if FULL else
              [(512, 2048, 64, 128), (256, 4096, 32, 64), (128, 1024, 12, 128), (300, 1024, 20, 32), (64, 512, 8, 64)])


@pytest.mark.parametrize("seq_len,seq_len_kv,num_heads,head_dim", CONTIGUOUS)
@pytest.mark.parametrize("fmt,dtype", FORMATS, ids=[f for f, _ in FORMATS])
@pytest.mark.parametrize("disable_cp", [False, True], ids=["cp", "causal"])
def test_mqa_logits(seq_len, seq_len_kv, num_heads, head_dim, fmt, dtype, disable_cp):
    if fmt == "mxfp4" and head_dim == 32:
        pytest.skip("FP4 heads hold 64 or 128 values")
    if not disable_cp and (seq_len_kv % seq_len or seq_len % 2):
        pytest.skip("context parallelism splits the KV evenly")
    q = torch.randn(seq_len, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(seq_len_kv, head_dim, device="cuda", dtype=torch.bfloat16)
    weights = ref.to_mqa_weights(torch.randn(seq_len, num_heads, device="cuda", dtype=dtype))
    if disable_cp:
        ks = torch.zeros(seq_len, dtype=torch.int, device="cuda")
        ke = torch.arange(seq_len, dtype=torch.int, device="cuda") + (seq_len_kv - seq_len)
    else:
        chunk, cp_size = seq_len // 2, seq_len_kv // seq_len
        cp_id = cp_size // 3
        ks = torch.zeros(seq_len, dtype=torch.int, device="cuda")
        ke = torch.zeros(seq_len, dtype=torch.int, device="cuda")
        for i in range(chunk):
            ke[i] = cp_id * chunk + i
            ke[i + chunk] = (cp_size * 2 - 1 - cp_id) * chunk + i
    ref_logits, _ = ref.ref_mqa_logits(q, kv, weights.float(), ks, ke)
    q_in, q_simulated = ref.quantize_mqa_q(q, fmt)
    if fmt == "fp8":
        kv_in = per_custom_dims_cast_to_fp8(kv, (0, ), False)
        kv_simulated = (kv_in[0].float() * kv_in[1].unsqueeze(1)).to(torch.bfloat16)
    else:
        kv_in, kv_simulated = ref.quantize_mqa_q(kv, fmt)
    simulated_logits, _ = ref.ref_mqa_logits(q_simulated, kv_simulated, weights.float(), ks, ke)

    max_seqlen_k = (ke - ks).max().item()
    kwargs = dict(q=q_in, kv=kv_in, weights=weights, cu_seq_len_k_start=ks, cu_seq_len_k_end=ke,
                  max_seqlen_k=max_seqlen_k)
    logits = deep_gemm.fp8_fp4_mqa_logits(**kwargs)
    assert logits.dtype == dtype and tuple(logits.shape) == (seq_len, max_seqlen_k)
    own = torch.arange(max_seqlen_k, device="cuda")[None, :] < (ke - ks)[:, None]
    masked = logits.masked_fill(~own, 0)
    for _ in range(3):
        assert torch.equal(deep_gemm.fp8_fp4_mqa_logits(**kwargs).masked_fill(~own, 0), masked)
    meta = deep_gemm.get_mqa_logits_metadata(ks, ke, seq_len_kv, num_heads)
    assert torch.equal(deep_gemm.fp8_fp4_mqa_logits(**kwargs, schedule_meta=meta).masked_fill(~own, 0), masked)

    # expand the compressed rows
    full = torch.zeros((seq_len, seq_len_kv), device="cuda")
    for i in range(seq_len):
        full[i, ks[i]:ke[i]] = logits[i, :ke[i] - ks[i]].float()
    valid = ref_logits != float("-inf")
    _check(full, ref_logits, simulated_logits, valid, fmt)


PAGED = ([(fmt, bkv, seqs, tpr, h, d, kv) for fmt, _ in FORMATS for bkv in (32, 64, 128) for seqs in (64, 256)
          for tpr in (1, 2, 6) for h in (8, 32, 64) for d in (64, 128) for kv in (8192, 65536)] if FULL else
         [("fp8", 64, 64, 1, 64, 128, 4096), ("fp8", 64, 32, 2, 32, 128, 8192), ("mxfp8", 32, 32, 6, 64, 128, 4096),
          ("mxfp4", 128, 48, 3, 12, 64, 2048), ("mxfp4", 64, 16, 8, 64, 128, 8192), ("mxfp8", 64, 24, 1, 20, 64, 3000)])


@pytest.mark.parametrize("fmt,block_kv,num_sequences,tokens_per_seq,num_heads,head_dim,avg_kv", PAGED)
@pytest.mark.parametrize("varlen", [False, True], ids=["batched", "varlen"])
def test_paged_mqa_logits(fmt, block_kv, num_sequences, tokens_per_seq, num_heads, head_dim, avg_kv, varlen):
    dtype = torch.float if fmt == "fp8" else torch.bfloat16
    if varlen:
        counts = torch.randint(1, tokens_per_seq + 1, (num_sequences, ), device="cuda", dtype=torch.int)
        indices = torch.arange(num_sequences, device="cuda", dtype=torch.int).repeat_interleave(counts)
        batch_size, next_n = counts.sum().item(), 1
    else:
        indices, batch_size, next_n = None, num_sequences, tokens_per_seq
    num_q = batch_size * next_n
    q = torch.randn((batch_size, next_n, num_heads, head_dim), device="cuda", dtype=torch.bfloat16)
    weights = ref.to_mqa_weights(torch.randn((num_q, num_heads), device="cuda", dtype=dtype))
    context_lens = torch.randint(int(0.7 * avg_kv), int(1.3 * avg_kv), (num_sequences, ), device="cuda",
                                 dtype=torch.int)
    if varlen:
        offsets = torch.cat([torch.arange(n.item(), device="cuda", dtype=torch.int) for n in counts])
        lens = (context_lens.repeat_interleave(counts) + offsets).view(-1, 1)
        max_lens = context_lens + (counts - 1)
    else:
        lens = ((context_lens.unsqueeze(1) + 1) * torch.rand(batch_size, next_n, device="cuda")).int()
        lens[:, -1] = context_lens
        max_lens = context_lens
    blocks_per_seq = ceil_div(max_lens, block_kv)
    max_model_len = blocks_per_seq.max().item() * block_kv
    num_blocks = blocks_per_seq.sum().item()
    kv_cache = torch.randn((num_blocks, block_kv, 1, head_dim), device="cuda", dtype=torch.bfloat16)
    block_table = torch.zeros((num_sequences, blocks_per_seq.max().item()), device="cuda", dtype=torch.int)
    pool = torch.randperm(num_blocks, device="cuda", dtype=torch.int)
    offset = 0
    for i, n in enumerate(blocks_per_seq.tolist()):
        block_table[i, :n] = pool[offset:offset + n]
        offset += n
    if varlen:
        block_table = block_table.repeat_interleave(counts, dim=0)

    ref_logits = ref.ref_paged_mqa_logits(q, kv_cache, weights.float(), lens, block_table, max_model_len)
    cast = {"fp8": ref.kv_cache_cast_to_fp8, "mxfp8": ref.kv_cache_cast_to_mxfp8, "mxfp4": ref.kv_cache_cast_to_mxfp4}[fmt]
    q_in, q_simulated = ref.quantize_mqa_q(q, fmt)
    kv_in, kv_simulated = cast(kv_cache)
    simulated_logits = ref.ref_paged_mqa_logits(q_simulated, kv_simulated, weights.float(), lens, block_table,
                                                max_model_len)
    meta = deep_gemm.get_paged_mqa_logits_metadata(lens, block_kv, deep_gemm.get_num_sms(), indices)
    kwargs = dict(q=q_in, kv_cache=kv_in, weights=weights, context_lens=lens, block_table=block_table,
                  schedule_meta=meta, max_context_len=max_model_len, indices=indices)
    logits = deep_gemm.fp8_fp4_paged_mqa_logits(**kwargs)
    assert logits.dtype == dtype
    valid = torch.arange(max_model_len, device="cuda")[None, :] < lens.view(-1, 1)
    masked = logits.masked_fill(~valid, 0)
    for _ in range(3):
        assert torch.equal(deep_gemm.fp8_fp4_paged_mqa_logits(**kwargs).masked_fill(~valid, 0), masked)
    _check(logits, ref_logits, simulated_logits, valid, fmt)


SPARSE = ([(fmt, paged, q, kv, sb, blocks, ua) for fmt in ("mxfp4", "mxfp8") for paged in (False, True)
           for q, kv in ((512, 8192), (64, 65536)) for sb in (8, 16) for blocks in (512, 2048) for ua in (False, True)
           if not (paged and ua)] if FULL else
          [("mxfp4", False, 96, 4096, 8, 256, False), ("mxfp8", False, 64, 3000, 16, 128, True),
           ("mxfp4", True, 48, 4096, 16, 128, False), ("mxfp8", True, 9, 640, 8, 64, False),
           ("fp8", False, 40, 2048, 16, 64, True)])


@pytest.mark.parametrize("fmt,paged,num_q_tokens,avg_kv_len,sparse_block_kv,num_max_sparse_blocks,unaligned", SPARSE)
@pytest.mark.parametrize("num_heads", [12, 32, 64])
def test_sparse_mqa_logits(fmt, paged, num_q_tokens, avg_kv_len, sparse_block_kv, num_max_sparse_blocks, unaligned,
                           num_heads):
    """Sparse logits equal the full logits bit for bit at every selected position."""
    head_dim, page_kv = 128, 64
    is_fp4 = fmt == "mxfp4"
    if paged and fmt == "fp8":
        pytest.skip("the paged sparse cache stores MX scales")
    rng = random.Random(num_q_tokens * 1000 + avg_kv_len)
    sizes, remaining = [], num_q_tokens
    while remaining > 0:
        sizes.append(min(rng.randint(2, 6), remaining))
        remaining -= sizes[-1]
    request_indices = [r for r, n in enumerate(sizes) for _ in range(n)]
    if paged:
        starts = [0] * num_q_tokens
        request_lens = [rng.randint(int(0.7 * avg_kv_len) // sparse_block_kv, int(1.3 * avg_kv_len) // sparse_block_kv)
                        * sparse_block_kv for _ in sizes]
        ends = [length + i for length, n in zip(request_lens, sizes) for i in range(n)]
    else:
        starts, ends, kv_end = [], [], 0
        for n in sizes:
            start = ceil_div(kv_end, sparse_block_kv) * sparse_block_kv + (rng.randrange(1, sparse_block_kv) if unaligned else 0)
            starts += [start] * n
            ends += [start + avg_kv_len + i for i in range(n)]
            kv_end = ends[-1]
        num_kv_tokens = max(1, kv_end)
    q_shape = (num_q_tokens, 1, num_heads) if paged else (num_q_tokens, num_heads)
    q_bf16 = torch.randn((*q_shape, head_dim), device="cuda", dtype=torch.bfloat16)
    if fmt == "fp8":
        q = (q_bf16.to(torch.float8_e4m3fn), None)
        weights = ref.to_mqa_weights(torch.randn((num_q_tokens, num_heads), device="cuda", dtype=torch.float))
    else:
        cast_fwd = per_token_cast_to_fp4 if is_fp4 else per_token_cast_to_fp8
        q_fp, q_sf = cast_fwd(q_bf16.view(-1, head_dim), use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
        q = q_fp.view(*q_shape, -1), q_sf.view(*q_shape)
        weights = ref.to_mqa_weights(torch.randn((num_q_tokens, num_heads), device="cuda", dtype=torch.bfloat16))
    sparse_indices, num_blocks = ref.make_sparse_kv_block_indices(ends, request_indices, sparse_block_kv,
                                                                  num_max_sparse_blocks, avg_kv_len, starts)

    if paged:
        max_lens = [length + n - 1 for length, n in zip(request_lens, sizes)]
        pages_per_request = [ceil_div(length, page_kv) for length in max_lens]
        num_pages = sum(pages_per_request)
        kv_cache, _ = (ref.kv_cache_cast_to_mxfp4 if is_fp4 else ref.kv_cache_cast_to_mxfp8)(
            torch.randn((num_pages, page_kv, 1, head_dim), device="cuda", dtype=torch.bfloat16))
        indices = torch.tensor(request_indices, device="cuda", dtype=torch.int32)
        lens = torch.tensor(ends, device="cuda", dtype=torch.int32)
        pool = torch.randperm(num_pages, device="cuda", dtype=torch.int32)
        table = torch.zeros((len(sizes), max(pages_per_request)), device="cuda", dtype=torch.int32)
        begin = 0
        for r, n in enumerate(pages_per_request):
            table[r, :n] = pool[begin:begin + n]
            begin += n
        block_table = table[indices.long()].contiguous()
        meta = deep_gemm.get_paged_sparse_mqa_logits_metadata(lens, block_table, indices, page_kv, sparse_indices,
                                                              q[0].dtype, sparse_block_kv)
        run_sparse = lambda: deep_gemm.fp8_fp4_paged_sparse_mqa_logits(q, kv_cache, weights, meta,
                                                                       num_max_sparse_blocks, sparse_block_kv)
        full = deep_gemm.fp8_fp4_paged_mqa_logits(
            q, kv_cache, weights, lens.view(-1, 1), block_table,
            deep_gemm.get_paged_mqa_logits_metadata(lens.view(-1, 1), page_kv, deep_gemm.get_num_sms(), indices),
            max(ends), indices)
    else:
        kv_bf16 = torch.randn((num_kv_tokens, head_dim), device="cuda", dtype=torch.bfloat16)
        if fmt == "fp8":
            kv = per_custom_dims_cast_to_fp8(kv_bf16, (0, ), False)
        else:
            kv_fp, kv_sf = cast_fwd(kv_bf16, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
            kv = kv_fp, kv_sf.view(num_kv_tokens)
        ks = torch.tensor(starts, device="cuda", dtype=torch.int32)
        ke = torch.tensor(ends, device="cuda", dtype=torch.int32)
        meta = deep_gemm.get_sparse_mqa_logits_metadata(ks, ke, num_kv_tokens, sparse_indices, q[0].dtype,
                                                        sparse_block_kv, use_unaligned_ks=unaligned)
        run_sparse = lambda: deep_gemm.fp8_fp4_sparse_mqa_logits(q, kv, weights, meta, num_max_sparse_blocks,
                                                                 sparse_block_kv, use_unaligned_ks=unaligned)
        full = deep_gemm.fp8_fp4_mqa_logits(q, kv, weights, ks, ke, num_kv_tokens)

    sparse = run_sparse()
    offsets = torch.arange(sparse_block_kv, device="cuda", dtype=torch.int32)
    starts_t = torch.tensor(starts, device="cuda", dtype=torch.int32)
    tokens = (sparse_indices.unsqueeze(-1) * sparse_block_kv + (starts_t % sparse_block_kv)[:, None, None]
              + offsets).flatten(1).long()
    num_tokens = torch.tensor(num_blocks, device="cuda") * sparse_block_kv
    valid = torch.arange(tokens.size(1), device="cuda")[None, :] < num_tokens[:, None]
    valid &= (tokens >= starts_t[:, None]) & (tokens < torch.tensor(ends, device="cuda")[:, None])
    expected = full.gather(1, (tokens - starts_t[:, None]).clamp(0, full.size(1) - 1))[valid]
    assert torch.equal(sparse[valid], expected)
    for _ in range(3):
        assert torch.equal(run_sparse()[valid], sparse[valid])
