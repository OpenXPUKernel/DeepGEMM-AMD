#!/usr/bin/env bash
set -euo pipefail

# Change current directory into project root
original_dir=$(pwd)
script_dir=$(realpath "$(dirname "$0")")
cd "$script_dir"

# CUDA uses CUTLASS headers. The native HIP host extension does not.
backend=$(python -c 'import torch; print("hip" if torch.version.hip else "cuda")')
if [ "$backend" = cuda ]; then
    ln -sfn "$script_dir/third-party/cutlass/include/cutlass" deep_gemm/include/cutlass
    ln -sfn "$script_dir/third-party/cutlass/include/cute" deep_gemm/include/cute
fi

# Remove old dist file, build files, and build
rm -rf build dist
rm -rf *.egg-info
python setup.py build

# Link only the pybind module, not a dependency library.
so_file=$(find build -name "_C*.so" -type f -print -quit)
if [ -n "$so_file" ]; then
    ln -sf "../$so_file" deep_gemm/
else
    echo "Error: No SO file found in build directory" >&2
    exit 1
fi

# Open users' original directory
cd "$original_dir"
