#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# SANA-Video-2B text-to-video serving.

MODEL="${MODEL:-Efficient-Large-Model/SANA-Video_2B_480p_diffusers}"
PORT="${PORT:-8091}"

vllm serve "$MODEL" \
    --omni \
    --model-class-name SanaVideoPipeline \
    --dtype bfloat16 \
    --port "$PORT"
