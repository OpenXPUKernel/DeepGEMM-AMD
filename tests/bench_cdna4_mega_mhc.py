"""Benchmark the CDNA4 mega_mhc on the shapes of tests/test_mega_mhc.py (FP8 output in the Mega MoE
scale layout plus BF16): the logical bytes per second, the bytes per second with the intermediate
I/O (the norm input of shifted mHC, the re-read new_residual of normal mHC), and the speedup over
the same computation in torch ops (`vs_torch = torch_us / ours_us`).

Our calls are captured in a CUDA graph and replayed; the torch ops run eagerly (the reference
checks its FP8 scales on the host). `--relative-only` prints no absolute numbers.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
import test_cdna4_mega_mhc as ref


def graph_us(fn, calls, reps=5):
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


def eager_us(fn, reps=5):
    fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1e3 / reps


def num_bytes(*tensors):
    return sum(t.numel() * t.element_size() for t in tensors if isinstance(t, torch.Tensor))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relative-only", action="store_true")
    args = parser.parse_args()
    for is_shifted in (True, False):
        print(("shifted" if is_shifted else "normal") + " mHC:")
        for hidden in (4096, 7168):
            for num_tokens in (1, 64, 200, 1025, 4096, 32768):
                inputs, _ = ref.make_inputs(num_tokens, hidden, 1, is_shifted)
                outputs = ref.make_outputs(inputs, "extra")
                calls = 20 if num_tokens <= 4096 else 3
                ours = graph_us(lambda: ref.run_mega_mhc(inputs, outputs), calls)
                # the reference checks its scales on the host, so it runs eagerly
                torch_us = eager_us(lambda: ref.mhc_reference(**inputs, sf_layout="extra", mix_dtype=torch.float))
                logical = num_bytes(*inputs.values(), *ref.logical_outputs(outputs).values())
                intermediate = 2 * num_bytes(outputs["y_bf16"]) if is_shifted else num_bytes(outputs["new_residual"])
                if args.relative_only:
                    cells = f"vs_torch {torch_us / ours:5.2f}x"
                else:
                    cells = (f"{ours:8.1f} us, {logical / ours / 1e6:5.2f} TB/s ({(logical + intermediate) / ours / 1e6:5.2f} "
                             f"with intermediate I/O) | torch {torch_us:8.1f} us, vs_torch {torch_us / ours:5.2f}x")
                print(f"  T={num_tokens:5} H={hidden}: {cells}", flush=True)
                del inputs, outputs
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
