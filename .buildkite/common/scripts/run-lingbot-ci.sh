#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
set -euo pipefail

STAGE="${1:?unit, smoke, generate, or evaluate}"
PROFILE="${2:?profile JSON}"
OUTPUT="${3:?artifact directory}"
mkdir -p "$OUTPUT"
rm -f "$OUTPUT/pytest.xml" "$OUTPUT/setup-failure.json"
exec > >(tee "$OUTPUT/run.log") 2>&1

finish() {
    local status=$?
    if [[ "$status" -ne 0 && ! -f "$OUTPUT/pytest.xml" ]]; then
        python - "$OUTPUT" "$status" <<'PY'
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
output = Path(sys.argv[1])
message = "LingBot CI setup failed; see run.log (exit " + sys.argv[2] + ")"
suite = ET.Element("testsuite", name="lingbot-setup", tests="1", failures="1")
case = ET.SubElement(suite, "testcase", name="setup")
ET.SubElement(case, "failure", message=message).text = message
ET.ElementTree(suite).write(output / "pytest.xml", encoding="utf-8", xml_declaration=True)
(output / "setup-failure.json").write_text(json.dumps({"passed": False, "error": message}) + "\n")
PY
    fi
    exit "$status"
}
trap finish EXIT

if [[ "$STAGE" == "unit" ]]; then
    export CUDA_VISIBLE_DEVICES=""
    cp "$PROFILE" "$OUTPUT/policy.json"
    python - "$OUTPUT" <<'UNIT_ENV'
import importlib.metadata
import platform
import sys
from pathlib import Path
from benchmarks.lingbot_world.regression import write_json
from tests.dfx.lingbot_world.runner import ROOT, git_revision
write_json(Path(sys.argv[1]) / "environment.json", {
    "python": sys.version,
    "machine": platform.machine(),
    "source_revision": git_revision(ROOT),
    "accelerator_visibility": "",
    "packages": {name: importlib.metadata.version(name)
                 for name in ("pytest", "numpy", "torch", "vllm", "diffusers", "transformers")},
})
UNIT_ENV
    python -m pytest -sv -p tests.dfx.lingbot_world.conftest \
        tests/benchmarks/test_lingbot_world_*.py \
        tests/diffusion/models/lingbot_world/test_offload_lingbot_world.py \
        'tests/diffusion/offloader/test_backend_plan_contract.py::test_text_encoder_only_offload_keeps_early_dit_projections_resident[cpu]' \
        --lingbot-output="$OUTPUT" -m 'core_model and cpu' --run-level core_model \
        --junitxml="$OUTPUT/pytest.xml"
elif [[ "$STAGE" == "evaluate" ]]; then
    GENERATION="${4:?generation artifact directory}"
    SOURCE_STEP="${5:?generation Buildkite step key}"
    if [[ "${BUILDKITE:-}" == "true" ]]; then
        buildkite-agent artifact download "$GENERATION/**" . --step "$SOURCE_STEP"
    fi
    EVALUATOR_LOCK="$(python -c 'from tests.dfx.lingbot_world.prepare import evaluator_cache_key; print(evaluator_cache_key())')"
    export LINGBOT_EVALUATOR_ROOT="${LINGBOT_EVALUATOR_ROOT:-${HF_HOME:-${TMPDIR:-/tmp}}/lingbot-evaluators-$EVALUATOR_LOCK}"
    python -m tests.dfx.lingbot_world.prepare evaluators --root "$LINGBOT_EVALUATOR_ROOT"
    export LINGBOT_EVALUATOR_PYTHON="$LINGBOT_EVALUATOR_ROOT/venv/bin/python"
    pytest -sv tests/dfx/lingbot_world/test_quality.py \
        --lingbot-profile="$PROFILE" --lingbot-output="$OUTPUT" --lingbot-generation="$GENERATION" \
        -m 'full_model and diffusion and cuda and H100 and cards_1' --run-level full_model \
        --junitxml="$OUTPUT/pytest.xml"
elif [[ "$STAGE" == "generate" || "$STAGE" == "smoke" ]]; then
    PLATFORM="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["platform"])' "$PROFILE")"
    if [[ "$STAGE" == "generate" && "$PLATFORM" == "cuda" ]]; then
        CUDA_VISIBLE_DEVICES="$(python -m tests.dfx.lingbot_world.prepare devices --profile "$PROFILE" --root "$OUTPUT")"
        export CUDA_VISIBLE_DEVICES
        export LINGBOT_ALLOCATION_FILE="$OUTPUT/allocation.json"
    fi
    REFERENCE="${TMPDIR:-/tmp}/lingbot-reference-${BUILDKITE_JOB_ID:-$$}"
    python -m tests.dfx.lingbot_world.prepare reference --profile "$PROFILE" \
        --root "$REFERENCE" --env-output "$OUTPUT/model.json"
    LINGBOT_MODEL_PATH="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["model_path"])' "$OUTPUT/model.json")"
    export LINGBOT_MODEL_PATH
    CARDS="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["dies"])' "$PROFILE")"
    TEST_FILE="tests/dfx/lingbot_world/test_generate.py"
    if [[ "$STAGE" == "smoke" ]]; then
        TEST_FILE="tests/dfx/lingbot_world/test_smoke.py"
    fi
    pytest -sv "$TEST_FILE" --lingbot-profile="$PROFILE" --lingbot-output="$OUTPUT" \
        --lingbot-reference-root="$REFERENCE" -m "full_model and diffusion and $PLATFORM and cards_$CARDS" \
        --run-level full_model --junitxml="$OUTPUT/pytest.xml"
else
    echo "Unknown LingBot CI stage: $STAGE" >&2
    exit 2
fi
