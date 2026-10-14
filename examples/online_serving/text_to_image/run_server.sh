#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Shared text-to-image online serving startup script

MODEL="${MODEL:-Qwen/Qwen-Image}"
PORT="${PORT:-8091}"

echo "Starting text-to-image server..."
echo "Model: $MODEL"
echo "Port: $PORT"

vllm serve "$MODEL" --omni \
    --port "$PORT"
