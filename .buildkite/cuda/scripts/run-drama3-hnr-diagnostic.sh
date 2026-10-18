#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
set -euo pipefail

root="${PWD}/drama3-diagnostic"
mkdir -p "$root"
python3 .buildkite/cuda/scripts/diagnose_drama3_hnr.py --expected-version 0.32.0 --output-dir "$root/v032" 2>&1 | tee "$root/v032.log"

git fetch --no-tags origin 1c0bde507dfbffd25b32f6092dbe4526c945b7dd
mkdir -p "$root/baseline-source"
git archive 1c0bde507dfbffd25b32f6092dbe4526c945b7dd vllm_omni tests/helpers tests/model_tests tests/__init__.py tests/assets/qwen3_tts/clone_2.wav | tar -xf - -C "$root/baseline-source"
python3 -m venv --without-pip "$root/baseline-env"
uv pip install --python "$root/baseline-env/bin/python" --no-deps 'vllm==0.31.0' 'transformers==5.14.0' 'tokenizers==0.22.2'
python3 - "$root/baseline-env" <<'PY'
import pathlib
import sys
import sysconfig

shared = pathlib.Path(sysconfig.get_paths()['purelib'])
version = f'python{sys.version_info.major}.{sys.version_info.minor}'
site = pathlib.Path(sys.argv[1]) / 'lib' / version / 'site-packages'
(site / 'shared_dependencies.pth').write_text(str(shared) + '\n')
# A path-only parent .pth does not process the parent's CUTLASS bootstrap .pth.
dsl = shared / 'nvidia_cutlass_dsl' / 'dsl_packages'
assert dsl.is_dir(), dsl
(site / 'cutlass_dsl_path.pth').write_text(str(dsl) + '\n')
PY
"$root/baseline-env/bin/python" .buildkite/cuda/scripts/diagnose_drama3_hnr.py --source-root "$root/baseline-source" --expected-version 0.31.0 --output-dir "$root/v031" 2>&1 | tee "$root/v031.log"
