#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Example measurement chain: several arms back to back on one set of four cards,
# each arm a fresh server on an exact commit, with a host gate so a loaded or
# memory-starved host does not produce a noisy arm.
#
#   CUDA_VISIBLE_DEVICES=0,1,2,3 RESULTS=/path/to/results bash chain_example.sh
#
# Edit the `measure` lines at the bottom: name, commit, port, config.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="${RESULTS:-$HERE/results}"
CFG="$HERE/configs"
MIN_AVAIL_GB=${MIN_AVAIL_GB:-230}   # host RAM the four workers need while loading weights
MAX_LOAD1=${MAX_LOAD1:-120}         # 1-minute load average above which steady std explodes
LOAD_WAIT_S=${LOAD_WAIT_S:-900}
mkdir -p "$ROOT"
echo "CARDS=${CUDA_VISIBLE_DEVICES:-unset}  chain started $(date '+%F %T')"

wait_gate() {
  local waited=0
  while :; do
    local avail load1
    avail=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)
    load1=$(awk '{printf "%d", $1}' /proc/loadavg)
    if [ "$avail" -ge "$MIN_AVAIL_GB" ] && { [ "$load1" -le "$MAX_LOAD1" ] || [ "$waited" -ge "$LOAD_WAIT_S" ]; }; then
      echo "gate: MemAvailable ${avail} GB, load1 ${load1}, waited ${waited}s; go ($(date '+%T'))"; return
    fi
    [ $((waited % 300)) -eq 0 ] && echo "gate: MemAvailable ${avail} GB, load1 ${load1}; waiting ($(date '+%T'))"
    sleep 30; waited=$((waited + 30))
  done
}

measure() {  # name rev port config
  for attempt in 1 2 3; do
    echo "attempt $attempt"
    [ -s "$ROOT/$1/bench.json" ] && return
    [ -d "$ROOT/$1" ] && mv "$ROOT/$1" "$ROOT/$1.failed.$(date '+%H%M%S')"
    wait_gate
    echo "=================== $1 ($(date '+%F %T')) ==================="
    env CODE_REV="$2" NUM_CHUNKS=40 SESSIONS=3 bash "$HERE/run_arm.sh" "$ROOT/$1" "$3" "$CFG/$4.yaml"
    echo "--- $1 done ($(date '+%T')) ---"; sleep 10
  done
}

# Stock, lossless, both PRs, and a control of the last one. Commits: see README.md.
measure stock      507cb1d83 8901 stock
measure lossless   6198a81cb 8902 kvreuse
measure both_prs   HEAD      8903 both_prs
measure both_prs_b HEAD      8904 both_prs
echo "chain complete $(date '+%F %T')"
