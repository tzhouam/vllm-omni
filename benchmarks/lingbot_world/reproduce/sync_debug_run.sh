#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# One short served run with torch's sync-debug mode armed in every worker
# (LINGBOT_SYNC_DEBUG=1 in the overlay), to find which per-step host call
# blocks on the device. Prints vllm_omni stack frames for every synchronising
# call, capped at LINGBOT_SYNC_DEBUG_MAX per worker.
#
#   CUDA_VISIBLE_DEVICES=0,1,2,3 NAME=syncdbg REV=HEAD bash sync_debug_run.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="${RESULTS:-$HERE/results}"
NAME=${NAME:?}; REV=${REV:-}; PORT=${PORT:-8881}
echo "CARDS=${CUDA_VISIBLE_DEVICES:-unset}  $NAME started $(date '+%F %T') rev=${REV:-HEAD}"
env CODE_REV="$REV" NUM_CHUNKS=12 SESSIONS=1 LINGBOT_SYNC_DEBUG=1 LINGBOT_SYNC_DEBUG_MAX=600 \
  bash "$HERE/run_arm.sh" "$ROOT/$NAME" "$PORT" "$HERE/configs/both_prs.yaml"
echo "$NAME complete $(date '+%F %T'); grep 'sync' $ROOT/$NAME/server.log"
