#!/bin/bash
# Strictly sequential: both engines get the same pinned cores, one at a time,
# so neither steals the other's cache or memory bandwidth.
set -u
HERE=/data/zhoutaichang/embedding_infer/analysis/experiments/spark_edge
V=/data/zhoutaichang/embedding_infer/.venvs/omni-cpu
G=/data/zhoutaichang/.cache/huggingface/hub/models--XHToken--Spark-X2.5-1.7B-GGUF/snapshots/23e1fcac55e7dd71e4c12a23723cc228ba0e5e85
B=/data/zhoutaichang/embedding_infer/llama.cpp/build/bin/llama-bench
cd $HERE
for t in 4 8 16 32; do
  last=$((t-1))
  echo "######## threads=$t : llama.cpp"
  taskset -c 0-$last $B -m $G/Spark-X2.5-1.7B.gguf -m $G/Spark-X2.5-1.7B-Q8_0.gguf \
      -m $G/Spark-X2.5-1.7B-Q4_K_M.gguf -p 512 -n 128 -t $t -o json 2>/dev/null \
      > $HERE/llamacpp_grid_t$t.json
  echo "######## threads=$t : vllm"
  LD_PRELOAD="$V/lib/python3.12/site-packages/vllm/libs/libtcmalloc_minimal.so.4:$V/lib/libiomp5.so" \
  CUDA_VISIBLE_DEVICES= VLLM_TARGET_DEVICE=cpu VLLM_CPU_KVCACHE_SPACE=4 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  OMP_NUM_THREADS=$t VLLM_CPU_OMP_THREADS_BIND=0-$last \
    taskset -c 0-$last timeout 2400 $V/bin/python bench_cpu_vllm.py \
     --tag "vllm-bf16-t$t" --out vllm_cpu_bench.jsonl 2>&1 | grep -E '"(tag|pp_tps|tg_tps|init_s)"'
done
echo "######## GRID DONE"
