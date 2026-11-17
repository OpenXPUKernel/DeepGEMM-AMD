"""Explicit plan lookup, precedence and graph replay on AMD GPU."""
import pytest
import torch

if not torch.version.hip or not torch.cuda.is_available():
    pytest.skip("requires ROCm", allow_module_level=True)

import deep_gemm


@pytest.fixture
def matrices(monkeypatch):
    monkeypatch.delenv("DG_CDNA4_GEMM_TUNE", raising=False)
    monkeypatch.delenv("DG_CDNA4_GEMM_TUNE_MAP", raising=False)
    torch.manual_seed(17)
    a = torch.randint(-3, 4, (320, 512), device="cuda").bfloat16()
    b = torch.randint(-3, 4, (128, 512), device="cuda").bfloat16()
    return a, b, torch.empty((320, 128), device="cuda", dtype=torch.bfloat16)


def test_exact_key_and_graph_replay(monkeypatch, matrices):
    a, b, output = matrices
    monkeypatch.setenv("DG_CDNA4_GEMM_TUNE_MAP",
                      "1320,128,1024,2,2,0=invalid;320,128,1024,2,2,0=2,2,2,4")
    deep_gemm.bf16_gemm_nt(a, b, output)
    assert torch.equal(output, (a.float() @ b.float().T).bfloat16())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        deep_gemm.bf16_gemm_nt(a, b, output)
    a.neg_()
    graph.replay()
    assert torch.equal(output, (a.float() @ b.float().T).bfloat16())


def test_nonmatching_key_keeps_default(monkeypatch, matrices):
    a, b, output = matrices
    deep_gemm.bf16_gemm_nt(a, b, output)
    reference = output.clone()
    monkeypatch.setenv("DG_CDNA4_GEMM_TUNE_MAP", "1320,128,1024,2,2,0=invalid")
    deep_gemm.bf16_gemm_nt(a, b, output)
    assert torch.equal(output, reference)


def test_global_off_overrides_table(monkeypatch, matrices):
    a, b, output = matrices
    monkeypatch.setenv("DG_CDNA4_GEMM_TUNE_MAP", "320,128,1024,2,2,0=invalid")
    monkeypatch.setenv("DG_CDNA4_GEMM_TUNE", "off")
    deep_gemm.bf16_gemm_nt(a, b, output)
    assert torch.equal(output, (a.float() @ b.float().T).bfloat16())


def test_invalid_matched_plan_is_rejected(monkeypatch, matrices):
    a, b, output = matrices
    monkeypatch.setenv("DG_CDNA4_GEMM_TUNE_MAP", "320,128,1024,2,2,0=0,2,2,4")
    with pytest.raises(ValueError, match="invalid DG_CDNA4_GEMM_TUNE"):
        deep_gemm.bf16_gemm_nt(a, b, output)
