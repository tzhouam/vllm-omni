#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Lance online serving startup script

MODEL="${MODEL:-bytedance-research/Lance}"
PORT="${PORT:-8091}"
DEPLOY_CONFIG="${DEPLOY_CONFIG:-vllm_omni/deploy/lance.yaml}"

echo "Starting Lance server..."
echo "Model:         $MODEL"
echo "Deploy config: $DEPLOY_CONFIG"
echo "Port:          $PORT"

vllm serve "$MODEL" --omni \
    --deploy-config "$DEPLOY_CONFIG" \
    --port "$PORT"
