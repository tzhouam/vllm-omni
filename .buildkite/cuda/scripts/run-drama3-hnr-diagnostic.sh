#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
set -euo pipefail

source_root="${PWD}"
probe="${source_root}/.buildkite/cuda/scripts/diagnose_drama3_hnr.py"
probe_dir="$(dirname "$probe")"
root="${source_root}/drama3-diagnostic"
baseline_commit="a965251bf1a3db3ab6871307ec39c938c0634479"
test_path="tests/e2e/online_serving/test_drama_3.py"
mkdir -p "$root"
# Preserve the original canonical model identity and its current immutable weights.
model_snapshot="$(hf download fishaudio/drama-3 --revision main --quiet)"
test "$(basename "$model_snapshot")" = b2b5ee998034b2975720906c815e7fb0e9f007bd
test -f "${model_snapshot}/config.json"
export DRAMA3_MODEL_SNAPSHOT="$model_snapshot"
export DRAMA3_ORIGINAL_TEST_HASH
DRAMA3_ORIGINAL_TEST_HASH="$(sha256sum "$test_path" | cut -d' ' -f1)"
git fetch --no-tags origin "$baseline_commit"
test "$(git show "${baseline_commit}:${test_path}" | sha256sum | cut -d' ' -f1)" = "$DRAMA3_ORIGINAL_TEST_HASH"
mkdir -p "$root/baseline-source"
git archive "$baseline_commit" pyproject.toml vllm_omni tests/conftest.py tests/helpers tests/model_tests tests/model_executor tests/e2e tests/__init__.py tests/assets/qwen3_tts/clone_2.wav | tar -xf - -C "$root/baseline-source"

run_phase() {
    local version="$1" interpreter="$2" source="$3" commit="$4" status
    export DRAMA3_PROBE_OUTPUT="${root}/${version}"
    export DRAMA3_PROBE_SOURCE="$source"
    export DRAMA3_PROBE_COMMIT="$commit"
    if [ "$version" = v032 ]; then
        export DRAMA3_PROBE_VERSION=0.32.0
    else
        export DRAMA3_PROBE_VERSION=0.31.0
    fi
    "$interpreter" "$probe" --prepare "${source}/${test_path}"
    set +e
    (
        cd "$source"
        PYTHONPATH="${probe_dir}${PYTHONPATH:+:${PYTHONPATH}}" "$interpreter" -m pytest -p diagnose_drama3_hnr -s -v "$test_path" -m 'advanced_model and cuda' --run-level advanced_model
    ) 2>&1 | tee "${root}/${version}.log"
    status=${PIPESTATUS[0]}
    set -e
    test "$status" -le 1
    "$interpreter" "$probe" --check "${DRAMA3_PROBE_OUTPUT}/results.json"
    # Exit1 is a retained original assertion failure, never a claim of quality success.
}

run_phase v032 "$(command -v python3)" "$source_root" "$BUILDKITE_COMMIT"
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
dsl = shared / 'nvidia_cutlass_dsl' / 'dsl_packages'
assert dsl.is_dir(), dsl
(site / 'cutlass_dsl_path.pth').write_text(str(dsl) + '\n')
PY
test "$(hf download fishaudio/drama-3 --revision main --quiet)" = "$model_snapshot"
run_phase v031 "$root/baseline-env/bin/python" "$root/baseline-source" "$baseline_commit"
