#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

set -euo pipefail

MODEL="${MODEL:-robbyant/lingbot-video-dense-1.3b}"
PORT="${PORT:-8099}"

vllm serve "${MODEL}" --omni \
    --model-class-name LingBotVideoPipeline \
    --port "${PORT}"
