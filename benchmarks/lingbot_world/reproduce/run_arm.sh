#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# One instrumented served run of LingBot-World v2 against this checkout.
#
#   run_arm.sh OUT_DIR [PORT] [DEPLOY_CONFIG]
#
# Environment (all optional):
#   CODE_ROOT   vllm-omni tree to serve        (default: the repo this script lives in)
#   CODE_REV    commit to check out in CODE_ROOT first (default: leave as is)
#   PY          python with vllm + vllm-omni deps (default: python3 on PATH)
#   MODEL       model path or HF id (default: robbyant/lingbot-world-v2-14b-causal-fast-diffusers)
#   NUM_CHUNKS  AR blocks per session (default 40)
#   SESSIONS    measured sessions (default 3; plus one warm-up session)
#   TARGET_FPS  real-time basis for RTF / deadline (default 12)
#   OVERLAY_DIR sitecustomize overlay for per-rank spans (default: ./overlay next to this script)
#
# `python -m vllm...` puts the cwd at sys.path[0], ahead of PYTHONPATH, so a run
# launched from another tree silently serves THAT tree's model code. The script
# cds into CODE_ROOT and asserts what actually got imported before spending a
# GPU hour on it. Set CUDA_VISIBLE_DEVICES to the four cards before calling.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
WT="${CODE_ROOT:-$(cd "$HERE/../../.." && pwd)}"
REV="${CODE_REV:-}"
OVERLAY="${OVERLAY_DIR:-$HERE/overlay}"
PY="${PY:-python3}"
MODEL="${MODEL:-robbyant/lingbot-world-v2-14b-causal-fast-diffusers}"
OUT="$1"; PORT="${2:-8771}"; CONFIG="${3:-$HERE/configs/both_prs.yaml}"

mkdir -p "$OUT"
export LINGBOT_TRACE_DIR="$OUT/spans"
export PYTHONPATH="$OVERLAY:$WT"
cd "$WT" || { echo "cannot cd to $WT"; exit 1; }
if [ -n "$REV" ]; then git -C "$WT" checkout -q "$REV" || { echo "cannot checkout $REV"; exit 1; }; fi

RESOLVED=$(env -u LINGBOT_TRACE_DIR "$PY" -c "import vllm_omni; print(vllm_omni.__file__)" 2>/dev/null | tail -1)
{
  echo "CODE_ROOT=$WT"
  echo "vllm_omni: $RESOLVED"
  git -C "$WT" rev-parse HEAD 2>&1
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "MODEL=$MODEL"
  echo "CONFIG=$CONFIG"
} > "$OUT/provenance.txt"
cat "$OUT/provenance.txt"
case "$RESOLVED" in
  "$WT"/*) echo "provenance OK" ;;
  *) echo "ABORT: vllm_omni resolved to $RESOLVED, expected a path under $WT"; exit 2 ;;
esac

"$PY" -m vllm.entrypoints.cli.main serve "$MODEL" \
  --omni --deploy-config "$CONFIG" --port "$PORT" --served-model-name lingbot-world \
  > "$OUT/server.log" 2>&1 &
SERVER_PID=$!
echo "server pid $SERVER_PID on :$PORT"

READY=0
for i in $(seq 1 240); do
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "server died early"; break; }
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { READY=1; echo "ready after ${i}0s"; break; }
  sleep 10
done

if [ "$READY" = "1" ]; then
  "$PY" "$WT/benchmarks/lingbot_world/benchmark_lingbot_world_realtime.py" \
    --host 127.0.0.1 --port "$PORT" --model lingbot-world \
    --num-chunks "${NUM_CHUNKS:-40}" --sessions "${SESSIONS:-3}" \
    --warmup-sessions 1 --target-fps "${TARGET_FPS:-12}" --print-chunks \
    --save-video "$OUT/last_session.mp4" \
    --output-json "$OUT/bench.json" > "$OUT/bench.log" 2>&1
  tail -30 "$OUT/bench.log"
else
  echo "server never became ready:"; tail -40 "$OUT/server.log"
fi

kill -TERM "$SERVER_PID" 2>/dev/null
for _ in $(seq 1 30); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 2; done
kill -KILL "$SERVER_PID" 2>/dev/null
echo "done"
