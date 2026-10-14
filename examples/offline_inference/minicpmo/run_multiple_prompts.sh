#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Multi-prompt offline batch (py_generator mode).
set -euo pipefail
cd "$(dirname "$0")"

python end2end.py --output-wav output_audio \
                  --query-type text \
                  --txt-prompts ../qwen3_omni/text_prompts_10.txt \
                  --py-generator
