#!/bin/bash
# Decode at batch 1 is dominated by ~250 OpenMP parallel regions per step, so
# how the runtime parks its threads between them matters as much as the GEMM.
set -u
HERE=/data/zhoutaichang/embedding_infer/analysis/experiments/spark_edge
V=/data/zhoutaichang/embedding_infer/.venvs/omni-cpu
M=${MODEL:-/data/zhoutaichang/embedding_infer/models/spark2_5-int4-g64}
OUT=${OUT:-$HERE/vllm_cpu_int4_env.jsonl}
cd $HERE
run () {  # name, extra env assignments...
  local name=$1; shift
  echo "######## $name"
  env "$@" \
    LD_PRELOAD="$V/lib/python3.12/site-packages/vllm/libs/libtcmalloc_minimal.so.4:$V/lib/libiomp5.so" \
    CUDA_VISIBLE_DEVICES= VLLM_TARGET_DEVICE=cpu VLLM_CPU_KVCACHE_SPACE=4 \
    VLLM_ENABLE_V1_MULTIPROCESSING=0 OMP_NUM_THREADS=32 VLLM_CPU_OMP_THREADS_BIND=56-87 \
    taskset -c 56-87 timeout 2400 $V/bin/python bench_cpu_vllm.py \
      --model $M --tag "$name" --reps 3 --out $OUT 2>&1 \
  | grep -E '"(tag|pp_tps|tg_tps)"'
}
run "env-baseline"          DUMMY=1
run "env-blocktime-inf"     KMP_BLOCKTIME=infinite
run "env-active-wait"       KMP_BLOCKTIME=infinite OMP_WAIT_POLICY=ACTIVE KMP_AFFINITY=granularity=fine,compact,1,0
run "env-spin-hard"         KMP_BLOCKTIME=infinite OMP_WAIT_POLICY=ACTIVE KMP_LIBRARY=turnaround
echo "#### ENV SWEEP DONE"
