"""Exercise the actual HIP metadata function on CPU, without loading HIP libs."""

import ast
from pathlib import Path

import pytest
torch = pytest.importorskip("torch")


def metadata_function():
    source = Path(__file__).parents[1] / "deep_gemm/_hip_api.py"
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == "get_paged_sparse_mqa_logits_metadata")
    scope = {"torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
    return scope[node.name]


@pytest.mark.parametrize("page_kv,sparse_kv", [(64, 8), (64, 16), (128, 8)])
@pytest.mark.parametrize("length", [0, 1, 65535, 65536, 65537, 1048576])
def test_paged_sparse_metadata_ignores_negative_tail(page_kv, sparse_kv, length):
    capacity = 8
    count = min(capacity, (length + sparse_kv - 1) // sparse_kv)
    candidates = torch.full((2, capacity), -1, dtype=torch.int32)
    candidates[0, :count] = torch.arange(count, dtype=torch.int32)
    lenses = torch.tensor([length, 0], dtype=torch.int32)
    pages = torch.tensor([[31, 7, 19], [47, 13, 5]], dtype=torch.int32)
    got = metadata_function()(lenses, pages, None, page_kv, candidates, torch.uint8, sparse_kv)
    expected = torch.zeros_like(candidates)
    for slot in range(count):
        pos = slot * sparse_kv
        expected[0, slot] = pages[0, min(pos // page_kv, pages.size(1) - 1)] * page_kv + pos % page_kv
    assert torch.equal(got[:2], lenses)
    assert torch.equal(got[2:4], lenses)
    assert torch.equal(got[4:].view_as(expected), expected)
