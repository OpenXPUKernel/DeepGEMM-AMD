"""Build selection and installer regressions; run without PyTorch or a GPU.

    python tests/test_build_install.py
"""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest
from unittest.mock import Mock, patch

from setuptools import Distribution, Extension


ROOT = Path(__file__).resolve().parents[1]


def load_setup(*, rocm, toolkit="/opt/test-rocm"):
    """Load the real setup script with only its PyTorch dependency replaced."""
    torch = types.ModuleType("torch")
    torch.version = types.SimpleNamespace(hip="7.0" if rocm else None,
                                          cuda=None if rocm else "12.8")
    torch.__version__ = "2.8.0"
    torch.compiled_with_cxx11_abi = lambda: True
    torch._C = types.SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=True)
    torch.utils = types.ModuleType("torch.utils")
    extensions = types.ModuleType("torch.utils.cpp_extension")
    extensions.CppExtension = Mock(side_effect=Extension)
    extensions.CUDAExtension = Mock(side_effect=Extension)
    extensions.CUDA_HOME = None if rocm else "/opt/test-cuda"
    extensions.ROCM_HOME = toolkit if rocm else None
    torch.utils.cpp_extension = extensions

    spec = importlib.util.spec_from_file_location("deep_gemm_build_test", ROOT / "setup.py")
    module = importlib.util.module_from_spec(spec)
    modules = {"torch": torch, "torch.utils": torch.utils,
               "torch.utils.cpp_extension": extensions}
    env = {"DG_SKIP_CUDA_BUILD": "0", "DG_FORCE_BUILD": "0", "DG_USE_LOCAL_VERSION": "0"}
    with patch.dict(sys.modules, modules), patch.dict(os.environ, env), \
            patch.object(sys, "path", [str(ROOT), *sys.path]):
        spec.loader.exec_module(module)
    return module, extensions


class BuildSelectionTests(unittest.TestCase):
    def test_rocm_builds_one_native_hip_extension(self):
        setup, constructors = load_setup(rocm=True)
        extensions = setup.get_ext_modules()
        self.assertEqual(len(extensions), 1)
        extension = extensions[0]
        self.assertEqual(extension.name, "deep_gemm._C")
        self.assertEqual(extension.sources, ["csrc/hip_python_api.cpp"])
        constructors.CppExtension.assert_called_once()
        constructors.CUDAExtension.assert_not_called()
        self.assertIn(("__HIP_PLATFORM_AMD__", "1"), extension.define_macros)
        self.assertTrue({"amdhip64", "hiprtc", "c10_hip", "torch_hip"}
                        .issubset(extension.libraries))
        self.assertFalse({"cuda", "cudart", "c10_cuda", "torch_cuda"}
                         .intersection(extension.libraries))
        self.assertIn("/opt/test-rocm/include", extension.include_dirs)
        self.assertTrue(any(path.endswith("third-party/deep_jit/include") for path in extension.include_dirs))
        self.assertIn("/opt/test-rocm/lib", extension.library_dirs)
        self.assertFalse(any("cutlass" in path or "test-cuda" in path
                             for path in extension.include_dirs))

    def test_rocm_does_not_require_cutlass_when_packaging_headers(self):
        setup, _ = load_setup(rocm=True)
        with tempfile.TemporaryDirectory() as directory:
            command = setup.CustomBuildPy(Distribution())
            command.build_lib = str(Path(directory) / "build")
            # HIP packages the vendored uni_symm device headers, but needs no CUTLASS.
            headers = Path(directory) / "third-party/uni_symm/include/uni_symm"
            headers.mkdir(parents=True)
            (headers / "sentinel.hpp").write_text("// packaged device dependency\n")
            with patch.object(setup, "current_dir", directory):
                command.prepare_includes()
            self.assertTrue((Path(command.build_lib) / "deep_gemm/include/uni_symm/sentinel.hpp").is_file())

    def test_rocm_builds_from_source_without_cuda_wheel_lookup(self):
        setup, _ = load_setup(rocm=True)
        self.assertIsNone(setup.torch.version.cuda)
        command = setup.CachedWheelsCommand(Distribution())
        with patch.object(setup._bdist_wheel, "run", autospec=True) as build, \
                patch.object(setup, "get_wheel_url") as lookup:
            command.run()
        build.assert_called_once_with(command)
        lookup.assert_not_called()

    def test_cuda_extension_keeps_existing_sources_and_dependencies(self):
        setup, constructors = load_setup(rocm=False)
        extensions = setup.get_ext_modules()
        self.assertEqual(len(extensions), 1)
        extension = extensions[0]
        self.assertEqual(extension.name, "deep_gemm._C")
        self.assertEqual(extension.sources, ["csrc/python_api.cpp"])
        constructors.CUDAExtension.assert_called_once()
        constructors.CppExtension.assert_not_called()
        self.assertIn("cudart", extension.libraries)
        self.assertNotIn("hiprtc", extension.libraries)
        self.assertIn("/opt/test-cuda/include", extension.include_dirs)
        self.assertIn(str(ROOT / "third-party/cutlass/include"), extension.include_dirs)
        self.assertNotIn(("__HIP_PLATFORM_AMD__", "1"), extension.define_macros)

    def test_cuda_cached_wheel_lookup_still_falls_back_to_source(self):
        setup, _ = load_setup(rocm=False)
        url, filename = setup.get_wheel_url()
        self.assertIn("+cu12-torch2.8-", filename)
        command = setup.CachedWheelsCommand(Distribution())
        with patch.object(setup._bdist_wheel, "run", autospec=True) as build, \
                patch.object(setup.urllib.request, "urlopen",
                             side_effect=setup.urllib.error.URLError("offline")) as download:
            command.run()
        download.assert_called_once_with(url, timeout=1)
        build.assert_called_once_with(command)

    def test_missing_rocm_toolkit_reports_actionable_error(self):
        setup, constructors = load_setup(rocm=True, toolkit=None)
        with self.assertRaisesRegex(RuntimeError, "ROCm development toolkit.*ROCM_HOME"):
            setup.get_ext_modules()
        constructors.CppExtension.assert_not_called()
        constructors.CUDAExtension.assert_not_called()


class InstallerTests(unittest.TestCase):
    def run_installer(self, *, build_status=0, install_status=0):
        with tempfile.TemporaryDirectory(prefix="deep-gemm-install-") as directory:
            root = Path(directory)
            project = root / "project with spaces"
            project.mkdir()
            shutil.copy2(ROOT / "install.sh", project / "install.sh")
            binaries = root / "bin"
            binaries.mkdir()
            interpreter = binaries / "python"
            interpreter.write_text(f"#!{sys.executable}\n" + textwrap.dedent("""\
                import json
                import os
                from pathlib import Path
                import sys

                args = sys.argv[1:]
                with open(os.environ["DG_TEST_CALL_LOG"], "a") as log:
                    log.write(json.dumps({"args": args, "cwd": os.getcwd()}) + "\\n")
                if args == ["setup.py", "bdist_wheel"]:
                    status = int(os.environ["DG_TEST_BUILD_STATUS"])
                    if status == 0:
                        Path("dist").mkdir()
                        Path("dist/deep_gemm-test.whl").touch()
                    sys.exit(status)
                if args[:2] == ["-m", "pip"]:
                    sys.exit(int(os.environ["DG_TEST_INSTALL_STATUS"]))
                sys.exit(99)
                """))
            interpreter.chmod(0o755)
            # A standalone pip must never be used instead of this interpreter.
            pip = binaries / "pip"
            pip.write_text("#!/bin/sh\nexit 98\n")
            pip.chmod(0o755)
            log = root / "calls.jsonl"
            env = dict(os.environ, PATH=f"{binaries}{os.pathsep}{os.environ.get('PATH', '')}",
                       DG_TEST_CALL_LOG=str(log), DG_TEST_BUILD_STATUS=str(build_status),
                       DG_TEST_INSTALL_STATUS=str(install_status))
            result = subprocess.run(["bash", str(project / "install.sh")], cwd=root,
                                    env=env, capture_output=True, text=True, timeout=15)
            calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
            for call in calls:
                self.assertEqual(call["cwd"], str(project))
            return result, [call["args"] for call in calls]

    def test_build_failure_stops_before_install_and_preserves_exit_status(self):
        result, calls = self.run_installer(build_status=31)
        self.assertEqual(result.returncode, 31, result.stderr)
        self.assertEqual(calls, [["setup.py", "bdist_wheel"]])

    def test_success_installs_wheel_with_the_same_python(self):
        result, calls = self.run_installer()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls, [
            ["setup.py", "bdist_wheel"],
            ["-m", "pip", "install", "dist/deep_gemm-test.whl", "--force-reinstall"],
        ])

    def test_install_failure_is_not_hidden_by_final_directory_change(self):
        result, calls = self.run_installer(install_status=27)
        self.assertEqual(result.returncode, 27, result.stderr)
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
