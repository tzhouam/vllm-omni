#!/bin/bash
# Both engines, strictly one at a time, on NUMA node 1's physical cores.
# Node 0 is in use by another job, and a shared-core benchmark measures the
# scheduler, not the engine.
set -u
HERE=/data/zhoutaichang/embedding_infer/analysis/experiments/spark_edge
V=/data/zhoutaichang/embedding_infer/.venvs/omni-cpu
G=/data/zhoutaichang/.cache/huggingface/hub/models--XHToken--Spark-X2.5-1.7B-GGUF/snapshots/23e1fcac55e7dd71e4c12a23723cc228ba0e5e85
B=/data/zhoutaichang/embedding_infer/llama.cpp/build/bin/llama-bench
BASE=56
cd $HERE
for t in 4 8 16 32; do
  lo=$BASE; hi=$((BASE + t - 1))
  echo "######## threads=$t cores $lo-$hi : llama.cpp"
  taskset -c $lo-$hi $B -m $G/Spark-X2.5-1.7B-Q4_K_M.gguf -p 512 -n 128 -t $t -o json \
      2>/dev/null > $HERE/llamacpp_node1_t$t.json
  echo "######## threads=$t cores $lo-$hi : vllm int4"
  LD_PRELOAD="$V/lib/python3.12/site-packages/vllm/libs/libtcmalloc_minimal.so.4:$V/lib/libiomp5.so" \
  CUDA_VISIBLE_DEVICES= VLLM_TARGET_DEVICE=cpu VLLM_CPU_KVCACHE_SPACE=4 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  OMP_NUM_THREADS=$t VLLM_CPU_OMP_THREADS_BIND=$lo-$hi \
    taskset -c $lo-$hi timeout 2400 $V/bin/python bench_cpu_vllm.py \
      --model ${MODEL:-/data/zhoutaichang/embedding_infer/models/spark2_5-int4-g64} \
      --tag "${TAG:-int4-g64}-t$t" --reps 3 --out ${OUT:-$HERE/vllm_cpu_int4_node1.jsonl} 2>&1 \
    | grep -E '"(tag|pp_tps|tg_tps|init_s)"'
done
echo "#### GRID DONE"
