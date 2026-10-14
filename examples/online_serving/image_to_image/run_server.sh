#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Qwen-Image-Edit online serving startup script

MODEL="${MODEL:-Qwen/Qwen-Image-Edit}"
PORT="${PORT:-8092}"

echo "Starting Qwen-Image-Edit server..."
echo "Model: $MODEL"
echo "Port: $PORT"

vllm serve "$MODEL" --omni \
    --port "$PORT"
