"""Import dispatch regressions without loading an accelerator runtime."""

import builtins
import importlib.util
from pathlib import Path
import sys
import types
from unittest.mock import Mock

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("hip", [None, "7.2"])
def test_backend_imports_are_isolated(monkeypatch, hip):
    namespace = "_deep_gemm_dispatch_test"
    torch = types.ModuleType("torch")
    torch.version = types.SimpleNamespace(hip=hip)
    monkeypatch.setitem(sys.modules, "torch", torch)
    cuda_api = types.ModuleType(f"{namespace}._cuda_api")
    hip_api = types.ModuleType(f"{namespace}._hip_api")
    cuda_api.bf16_gemm_nt = Mock(name="cuda_gemm")
    hip_api.bf16_gemm_nt = Mock(name="hip_gemm")
    modules = {"_cuda_api": cuda_api, "_hip_api": hip_api,
               "_hip_mega_moe": types.ModuleType(f"{namespace}._hip_mega_moe"),
               "utils": types.ModuleType(f"{namespace}.utils"),
               "_C": types.ModuleType(f"{namespace}._C")}
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, f"{namespace}.{name}", module)
    uni_symm = types.ModuleType("uni_symm")
    uni_symm.bind = Mock()
    monkeypatch.setitem(sys.modules, "uni_symm", uni_symm)
    real_import = builtins.__import__
    imports = []

    def record_import(name, *args, **kwargs):
        imports.append(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", record_import)
    spec = importlib.util.spec_from_file_location(
        namespace, ROOT / "deep_gemm/__init__.py",
        submodule_search_locations=[str(ROOT / "deep_gemm")],
    )
    package = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, namespace, package)
    spec.loader.exec_module(package)
    assert package.__version__ == "2.8.1"
    assert package.bf16_gemm_nt is (hip_api if hip else cuda_api).bf16_gemm_nt
    if hip:
        uni_symm.bind.assert_called_once_with(modules["_C"])
        assert "_cuda_api" not in imports
    else:
        uni_symm.bind.assert_not_called()
        assert "_hip_api" not in imports
        assert "_hip_mega_moe" not in imports
        assert "uni_symm" not in imports
