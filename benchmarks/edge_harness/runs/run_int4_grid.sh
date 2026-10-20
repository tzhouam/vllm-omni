#!/bin/bash
# Strictly sequential: one engine at a time on the same pinned cores, so
# neither steals the other's memory bandwidth. Same rules as run_cpu_grid.sh.
set -u
HERE=/data/zhoutaichang/embedding_infer/analysis/experiments/spark_edge
V=/data/zhoutaichang/embedding_infer/.venvs/omni-cpu
G=/data/zhoutaichang/.cache/huggingface/hub/models--XHToken--Spark-X2.5-1.7B-GGUF/snapshots/23e1fcac55e7dd71e4c12a23723cc228ba0e5e85
B=/data/zhoutaichang/embedding_infer/llama.cpp/build/bin/llama-bench
INT4=/data/zhoutaichang/embedding_infer/models/spark2_5-int4-g64
OUT=${1:-$HERE/int4_grid.jsonl}
cd $HERE
for t in 4 8 16 32; do
  last=$((t-1))
  echo "######## threads=$t : llama.cpp Q4_K_M"
  taskset -c 0-$last $B -m $G/Spark-X2.5-1.7B-Q4_K_M.gguf -p 512 -n 128 -t $t -o json 2>/dev/null \
      > $HERE/llamacpp_q4km_t$t.json
  python3 -c "
import json;d=json.load(open('$HERE/llamacpp_q4km_t$t.json'))
for r in d: print('  ', r['n_prompt'] or 'tg', round(1e9*(r['n_prompt'] or r['n_gen'])/r['avg_ns'],2), 'tok/s')"
  echo "######## threads=$t : vLLM int4 g64"
  LD_PRELOAD="$V/lib/python3.12/site-packages/vllm/libs/libtcmalloc_minimal.so.4:$V/lib/libiomp5.so" \
  CUDA_VISIBLE_DEVICES= VLLM_TARGET_DEVICE=cpu VLLM_CPU_KVCACHE_SPACE=4 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  OMP_NUM_THREADS=$t VLLM_CPU_OMP_THREADS_BIND=0-$last \
    taskset -c 0-$last timeout 2400 $V/bin/python bench_cpu_vllm.py --model $INT4 \
     --tag "vllm-int4-g64-t$t" --reps 3 --out $OUT 2>&1 | grep -E '"(tag|pp_tps|tg_tps|init_s)"'
done
echo "######## threads=32 : vLLM bf16 control"
LD_PRELOAD="$V/lib/python3.12/site-packages/vllm/libs/libtcmalloc_minimal.so.4:$V/lib/libiomp5.so" \
CUDA_VISIBLE_DEVICES= VLLM_TARGET_DEVICE=cpu VLLM_CPU_KVCACHE_SPACE=4 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
OMP_NUM_THREADS=32 VLLM_CPU_OMP_THREADS_BIND=0-31 \
  taskset -c 0-31 timeout 2400 $V/bin/python bench_cpu_vllm.py \
   --tag "vllm-bf16-t32-control" --reps 3 --out $OUT 2>&1 | grep -E '"(tag|pp_tps|tg_tps|init_s)"'
echo "######## GRID DONE"
