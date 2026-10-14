#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Single-prompt offline smoke test (text → text + 24 kHz speech).
set -euo pipefail
cd "$(dirname "$0")"

python end2end.py --output-wav output_audio \
                  --query-type text
