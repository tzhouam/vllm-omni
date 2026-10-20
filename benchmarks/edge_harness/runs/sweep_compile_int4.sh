#!/bin/bash
# vLLM compiles one dynamic-shape graph, so inductor cannot see that a decode
# step's tensors are a single row and emits a parallel loop -- and a barrier --
# for every norm, rope and GEGLU. compile_sizes pins a static graph for the
# shape decode actually runs at.
set -u
HERE=/data/zhoutaichang/embedding_infer/analysis/experiments/spark_edge
V=/data/zhoutaichang/embedding_infer/.venvs/omni-cpu
M=${MODEL:-/data/zhoutaichang/embedding_infer/models/spark2_5-int4-g64}
OUT=${OUT:-$HERE/vllm_cpu_int4_compile.jsonl}
CORES=${CORES:-0-31}
cd $HERE
run () {
  local name=$1; local cc=$2; shift 2
  echo "######## $name  cc=$cc  cores_busy_before=$(python3 cpu_busy.py $CORES)%"
  KMP_BLOCKTIME=infinite OMP_WAIT_POLICY=ACTIVE KMP_LIBRARY=turnaround \
    LD_PRELOAD="$V/lib/python3.12/site-packages/vllm/libs/libtcmalloc_minimal.so.4:$V/lib/libiomp5.so" \
    CUDA_VISIBLE_DEVICES= VLLM_TARGET_DEVICE=cpu VLLM_CPU_KVCACHE_SPACE=4 \
    VLLM_ENABLE_V1_MULTIPROCESSING=0 OMP_NUM_THREADS=${THREADS:-32} VLLM_CPU_OMP_THREADS_BIND=$CORES \
    taskset -c $CORES timeout 2400 $V/bin/python bench_cpu_vllm.py \
      --model $M --tag "$name" --reps 3 --out $OUT ${cc:+--compilation-config "$cc"} "$@" 2>&1 \
  | grep -E '"(tag|pp_tps|tg_tps)"|Error|Traceback' | head -8
}
run "cmp-baseline"    ""
run "cmp-size1"       '{"compile_sizes":[1]}'
run "cmp-eager"       "" --enforce-eager
run "cmp-baseline2"   ""
echo "#### COMPILE SWEEP DONE"
