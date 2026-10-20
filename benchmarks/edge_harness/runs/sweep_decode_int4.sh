#!/bin/bash
set -u
HERE=/data/zhoutaichang/embedding_infer/analysis/experiments/spark_edge
V=/data/zhoutaichang/embedding_infer/.venvs/omni-cpu
OUT=${OUT:-$HERE/vllm_cpu_int4_decode.jsonl}
CORES=${CORES:-0-31}
cd $HERE
run () {
  local name=$1; local model=$2; shift 2
  echo "######## $name busy_before=$(python3 cpu_busy.py $CORES)%"
  KMP_BLOCKTIME=infinite OMP_WAIT_POLICY=ACTIVE KMP_LIBRARY=turnaround \
    LD_PRELOAD="$V/lib/python3.12/site-packages/vllm/libs/libtcmalloc_minimal.so.4:$V/lib/libiomp5.so" \
    CUDA_VISIBLE_DEVICES= VLLM_TARGET_DEVICE=cpu VLLM_CPU_KVCACHE_SPACE=4 \
    VLLM_ENABLE_V1_MULTIPROCESSING=0 OMP_NUM_THREADS=32 VLLM_CPU_OMP_THREADS_BIND=$CORES \
    taskset -c $CORES timeout 2400 $V/bin/python bench_cpu_vllm.py \
      --model $model --tag "$name" --reps 3 --out $OUT "$@" 2>&1 \
  | grep -E '"(tag|pp_tps|tg_tps)"|Error|Traceback' | head -6
}
G64=/data/zhoutaichang/embedding_infer/models/spark2_5-int4-g64
G128=/data/zhoutaichang/embedding_infer/models/spark2_5-int4-g128
run "g64"              $G64
run "g64-nodetok"      $G64  --no-detokenize
run "g128"             $G128
run "g128-nodetok"     $G128 --no-detokenize
echo "#### DECODE SWEEP DONE"
