"""Benchmark the CDNA4 bf16_mega_gate (6 of the routed experts with a bias and a physical map) on the
shapes of tests/test_mega_gate.py: the whole gate, its logits GEMM and its routing kernel, against
the same gate in torch ops (`vs_torch = torch_us / ours_us`) and the GEMM against hipBLASLt
(`gemm_vs_hipblaslt = hipblaslt_us / ours_us`, a BF16 output).

Twenty calls are captured in a CUDA graph and replayed. `--relative-only` prints no absolute numbers.
"""
import argparse

import torch
import torch.nn.functional as F

import deep_gemm
from deep_gemm import _hip_api

NUM_TOPK = 6
SCALE = 1.5


def graph_us(fn, calls=20, reps=10):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            fn()
    graph.replay()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1e3 / (reps * calls)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relative-only", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(0)
    cases = [(t, h, 384) for t in (1, 16, 128, 1024, 4096, 16384, 32768) for h in (4096, 7168)]
    cases += [(4096, 5120, 128), (8192, 7168, 256), (8192, 4096, 512), (32768, 7168, 512)]
    for t, h, e in cases:
        x = torch.randn((t, h), dtype=torch.bfloat16, device="cuda")
        w = torch.randn((e, h), dtype=torch.bfloat16, device="cuda") * h ** -0.5
        bias = torch.randn((e, ), dtype=torch.float32, device="cuda")
        to_physical_map = torch.arange(e, dtype=torch.int32, device="cuda")[:, None].contiguous()
        logical_count = torch.ones((e, ), dtype=torch.int32, device="cuda")
        topk_idx = torch.empty((t, NUM_TOPK), dtype=torch.int64, device="cuda")
        topk_weights = torch.empty((t, NUM_TOPK), dtype=torch.float32, device="cuda")
        logits = torch.empty((t, e), dtype=torch.float32, device="cuda")
        logits_bf16 = torch.empty((t, e), dtype=torch.bfloat16, device="cuda")

        total = graph_us(lambda: deep_gemm.bf16_mega_gate(
            x, w, NUM_TOPK, False, 1, SCALE, 0, bias=bias, to_physical_map=to_physical_map,
            logical_count=logical_count, out=(topk_idx, topk_weights)))
        gemm = graph_us(lambda: deep_gemm.bf16_gemm_nt(x, w, logits))
        routing = graph_us(lambda: _hip_api._backend().cdna4_mega_gate_routing(
            logits, bias, None, None, None, None, None, to_physical_map, logical_count, topk_idx, topk_weights,
            None, NUM_TOPK, 0, 0, SCALE))
        hipblaslt = graph_us(lambda: torch.matmul(x, w.T, out=logits_bf16))

        def torch_gate():
            scores = F.softplus(torch.matmul(x, w.T).float()).sqrt()
            idx = (scores + bias).topk(NUM_TOPK, dim=1).indices
            selected = scores.gather(1, idx)
            return idx, selected / (selected.sum(1, keepdim=True) + 1e-20) * SCALE

        reference = graph_us(torch_gate)
        shape = f"tokens={t:6} hidden={h} experts={e}"
        if args.relative_only:
            print(f"{shape}: vs_torch {reference / total:.2f}x, gemm share {gemm / (gemm + routing):.2f}, "
                  f"gemm_vs_hipblaslt {hipblaslt / gemm:.2f}x", flush=True)
        else:
            print(f"{shape}: {total:7.1f} us (gemm {gemm:7.1f} + routing {routing:6.1f}), "
                  f"{2 * t * e * h / total / 1e6:4.0f} TFLOPS | torch {reference:7.1f} us, vs_torch "
                  f"{reference / total:.2f}x | hipBLASLt gemm {hipblaslt:7.1f} us, gemm_vs_hipblaslt "
                  f"{hipblaslt / gemm:.2f}x", flush=True)


if __name__ == "__main__":
    main()
