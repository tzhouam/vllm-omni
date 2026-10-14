#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

set -euo pipefail

python end2end.py \
  --model aurateam/AURA \
  --deploy-config vllm_omni/deploy/aura_omni.yaml \
  --modalities text,audio \
  "$@"
