#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
set -euo pipefail

# Install the selected Python source using ABI-identical published binaries.
# The caller first installs that binary wheel and resolves its runtime deps.
source_commit=${1:?source commit required}
binary_commit=${2:?binary wheel commit required}
binary_version=${3:?binary wheel version required}
python_executable=${4:-$(python3 -c 'import sys; print(sys.executable)')}
for commit in "$source_commit" "$binary_commit"; do
    [[ "$commit" =~ ^[0-9a-f]{40}$ ]] || { echo "Expected a full commit SHA" >&2; exit 1; }
done
source_dir=$(mktemp -d)
trap 'rm -rf "$source_dir"' EXIT

git -C "$source_dir" init --quiet
git -C "$source_dir" remote add origin https://github.com/vllm-project/vllm.git
git -C "$source_dir" fetch --depth 64 origin "$source_commit" "$binary_commit" \
    "refs/tags/v${binary_version}:refs/tags/v${binary_version}"
git -C "$source_dir" checkout --detach "$source_commit"
# Refuse binary reuse if native code, dependencies or build configuration changed.
git -C "$source_dir" diff --exit-code "$binary_commit" "$source_commit" -- \
    csrc rust requirements pyproject.toml setup.py CMakeLists.txt cmake
uv pip install --python "$python_executable" -r "$source_dir/requirements/build/cuda.txt"
VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_LOCATION="https://wheels.vllm.ai/${binary_commit}/vllm-${binary_version}-cp38-abi3-manylinux_2_28_$(uname -m).whl" \
    uv pip install --python "$python_executable" --no-build-isolation --no-deps "$source_dir"
# This must remain importable after the temporary checkout has been removed.
rm -rf "$source_dir"
"$python_executable" -c "from vllm._version import __commit_id__; c = (__commit_id__ or '').removeprefix('g'); assert len(c) >= 8 and '$source_commit'.startswith(c), __commit_id__"
echo "Installed vLLM source=${source_commit} native-binaries=${binary_commit}"
