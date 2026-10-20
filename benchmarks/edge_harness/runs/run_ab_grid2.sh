#!/bin/bash
# Round two, with cpp.min_chunk_size dropped: it measured -6% decode and -19%
# prefill, so the remaining levers get tested against `gate` alone.
set -u
HERE=/data/zhoutaichang/embedding_infer/analysis/experiments/spark_edge
V=/data/zhoutaichang/embedding_infer/.venvs/omni-cpu
M=/data/zhoutaichang/embedding_infer/models
OUT=$HERE/ab_grid2.jsonl
cd $HERE
run () {
  local tag=$1 model=$2
  VLLM_OMNI_SPARK_GATE_REDUCTION=1 KMP_BLOCKTIME=infinite OMP_WAIT_POLICY=active \
  LD_PRELOAD="$V/lib/python3.12/site-packages/vllm/libs/libtcmalloc_minimal.so.4:$V/lib/libiomp5.so" \
  CUDA_VISIBLE_DEVICES= VLLM_TARGET_DEVICE=cpu VLLM_CPU_KVCACHE_SPACE=4 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  OMP_NUM_THREADS=24 VLLM_CPU_OMP_THREADS_BIND=32-55 \
    taskset -c 32-55 timeout 2400 $V/bin/python bench_cpu_vllm.py --model "$model" \
      --tag "$tag" --reps 5 --out $OUT 2>&1 | grep -E '"(tag|pp_tps|tg_tps)"'
}
run "gate-g64"          $M/spark2_5-int4-g64
run "gate-g128"         $M/spark2_5-int4-g128
run "gate-g64-noop"     $M/spark2_5-int4-g64-nocustomop
run "gate-w4a8"         $M/spark2_5-gptq-g64
run "gate-g64-again"    $M/spark2_5-int4-g64
echo "#### AB GRID 2 DONE"
