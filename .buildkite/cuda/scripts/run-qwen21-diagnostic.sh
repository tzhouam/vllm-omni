#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
set -euo pipefail
mode="${1:?expected async or blocking}"
case "$mode" in
  async) export CUDA_LAUNCH_BLOCKING=0 ;;
  blocking) export CUDA_LAUNCH_BLOCKING=1 ;;
  *) exit 2 ;;
esac
mkdir -p "qwen21-diagnostic/$mode"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv > "qwen21-diagnostic/$mode/hardware.txt"
pip freeze > "qwen21-diagnostic/$mode/packages.txt"
sha256sum tests/dfx/perf/tests/test_qwen_image_21_vllm_omni.json > "qwen21-diagnostic/$mode/config.sha256"
for attempt in 1 2; do
  export BENCHMARK_DIR="qwen21-diagnostic/$mode/attempt-$attempt"
  mkdir -p "$BENCHMARK_DIR"
  pytest -s -v tests/dfx/perf/scripts/run_benchmark.py --test-config-file tests/dfx/perf/tests/test_qwen_image_21_vllm_omni.json -m "H100 and B200 and cards_1" --junitxml="$BENCHMARK_DIR/results.xml" 2>&1 | tee "$BENCHMARK_DIR/pytest.log"
done
