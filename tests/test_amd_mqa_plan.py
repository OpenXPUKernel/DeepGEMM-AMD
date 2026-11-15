"""Paged launch plans preserve score bits under graph/stream/length changes."""

import pytest
import torch

if not torch.version.hip or not torch.cuda.is_available():
    pytest.skip("requires ROCm", allow_module_level=True)

import deep_gemm


@torch.inference_mode()
def test_paged_query_group_plan_is_bitwise(monkeypatch):
    torch.manual_seed(1017)
    rows, heads, dimension, page = 12, 32, 128, 64
    q = torch.randint(
        -128, 128, (rows, 1, heads, dimension // 2), dtype=torch.int8, device="cuda"
    )
    sf = torch.full((rows, 1, heads), 0x7F7F7F7F, dtype=torch.int32, device="cuda")
    cache = torch.randint(
        0, 256, (160, page, 1, dimension // 2 + 4), dtype=torch.uint8, device="cuda"
    )
    cache.view(160, -1)[:, page * dimension // 2 :].fill_(127)
    weights = torch.randn((rows, heads), dtype=torch.bfloat16, device="cuda")
    lengths = torch.tensor(
        [0, 1, 127, 128, 129, 4097] * 2, dtype=torch.int32, device="cuda"
    ).reshape(rows, 1)
    indices = torch.arange(2, dtype=torch.int32, device="cuda").repeat_interleave(6)
    blocks = (
        torch.stack(
            (torch.arange(80, device="cuda"), torch.arange(80, 160, device="cuda"))
        )
        .int()
        .repeat_interleave(6, 0)
    )
    schedule = deep_gemm.get_paged_mqa_logits_metadata(
        lengths, page, deep_gemm.get_num_sms(), indices=indices
    )

    def run(fast):
        monkeypatch.setenv("DG_MQA_PAGED_GROUPS", "3" if fast else "0")
        monkeypatch.setenv("DG_MQA_PAGED_ROUNDS", "12" if fast else "0")
        return deep_gemm.fp8_fp4_paged_mqa_logits(
            (q, sf), cache, weights, lengths, blocks, schedule, 8192, indices=indices
        )

    def compare(expected, actual):
        valid = torch.arange(8192, device="cuda")[None, :] < lengths.reshape(-1, 1)
        assert not (
            (expected.view(torch.int16) != actual.view(torch.int16)) & valid
        ).any()

    expected, actual = run(False), run(True)
    compare(expected, actual)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run(True)
    q.random_(-128, 128)
    lengths.copy_(lengths.flip(0))
    expected = run(False)
    graph.replay()
    compare(expected, actual)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        actual = run(True)
    torch.cuda.current_stream().wait_stream(stream)
    compare(expected, actual)
