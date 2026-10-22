"""Single-GPU stable route positions, tile boundaries and graph replay.

This validates integer routing, not multi-GPU expert output accuracy.
"""

import os

import pytest
import torch
import torch.distributed as dist

if not torch.version.hip or not torch.cuda.is_available():
    pytest.skip("requires ROCm", allow_module_level=True)

import deep_gemm


@pytest.fixture(scope="module")
def routing(tmp_path_factory):
    assert not dist.is_initialized()
    path = tmp_path_factory.mktemp("stable-route") / "rendezvous"
    dist.init_process_group("nccl", init_method=f"file://{path}", rank=0, world_size=1)
    buffer = deep_gemm.get_symm_buffer_for_mega_moe(dist.group.WORLD, 8, 4096, 6, 256, 128,
                                                   mma_type="bf16xbf16")
    weights = (torch.zeros((8, 256, 256), device="cuda", dtype=torch.bfloat16),
               torch.zeros((8, 256, 128), device="cuda", dtype=torch.bfloat16))
    saved = {k: os.environ.get(k) for k in ("DG_MEGA_MOE_DETERMINISTIC_ROUTE", "DG_MEGA_MOE_DIRECT_COMBINE",
                                           "DG_MEGA_MOE_NV_PIPELINE", "DG_MEGA_MOE_STAGE1_QUEUE")}
    os.environ.update(DG_MEGA_MOE_DETERMINISTIC_ROUTE="1", DG_MEGA_MOE_DIRECT_COMBINE="0",
                      DG_MEGA_MOE_NV_PIPELINE="0", DG_MEGA_MOE_STAGE1_QUEUE="0")
    yield buffer, weights
    buffer.destroy()
    dist.destroy_process_group()
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def run_and_check(buffer, weights, routes):
    rows = routes.size(0)
    buffer.x[:rows].zero_()
    buffer.topk_idx[:rows].copy_(routes)
    buffer.topk_weights[:rows].fill_(1 / 6)
    output = torch.empty((rows, 256), device="cuda", dtype=torch.bfloat16)
    deep_gemm.bf16_mega_moe(output, *weights, buffer)
    assert not output.numel() or (output == 0).all()
    actual = buffer._view(buffer.layout["positions"], (4096 * 6,), torch.int32)[:rows * 6].cpu().tolist()
    counts = [0] * 8
    for entry, expert in enumerate(routes.flatten().cpu().tolist()):
        if expert >= 0:
            assert actual[entry] == counts[expert]
            counts[expert] += 1


@pytest.mark.parametrize("rows", [0, 1, 63, 64, 65, 255, 256, 257, 512, 4095, 4096])
@pytest.mark.parametrize("ordered_push", ["0", "1"])
def test_stable_positions(routing, rows, ordered_push):
    buffer, weights = routing
    os.environ["DG_MEGA_MOE_ORDERED_PUSH"] = ordered_push
    routes = ((torch.arange(rows, device="cuda")[:, None] + torch.arange(6, device="cuda")[None, :]) % 8).long()
    routes[::7, -1] = -1
    for _ in range(3):
        run_and_check(buffer, weights, routes)


def test_changing_routes_graph_replay(routing):
    buffer, weights = routing
    rows = 257
    routes = ((torch.arange(rows, device="cuda")[:, None] + torch.arange(6, device="cuda")[None, :]) % 8).long()
    run_and_check(buffer, weights, routes)
    output = torch.empty((rows, 256), device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        deep_gemm.bf16_mega_moe(output, *weights, buffer)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        deep_gemm.bf16_mega_moe(output, *weights, buffer)
    for shift in (1, 3):
        updated = (routes + shift) % 8
        buffer.topk_idx[:rows].copy_(updated)
        graph.replay()
        actual = buffer._view(buffer.layout["positions"], (4096 * 6,), torch.int32)[:rows * 6].clone()
        run_and_check(buffer, weights, updated)
        expected = buffer._view(buffer.layout["positions"], (4096 * 6,), torch.int32)[:rows * 6]
        assert torch.equal(actual, expected)
