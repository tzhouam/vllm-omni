#!/bin/bash
# Pin to physical cores 0..N-1 on socket 0 (cores 0-55 physical, 112+ are HT
# siblings) so the thread count is enforced, not merely requested.
V=/data/zhoutaichang/embedding_infer/.venvs/omni-cpu
export LD_PRELOAD="$V/lib/python3.12/site-packages/vllm/libs/libtcmalloc_minimal.so.4:$V/lib/libiomp5.so"
export CUDA_VISIBLE_DEVICES= VLLM_TARGET_DEVICE=cpu VLLM_CPU_KVCACHE_SPACE=4 VLLM_ENABLE_V1_MULTIPROCESSING=0
for t in 4 8 16 32; do
  last=$((t-1))
  echo "### threads=$t"
  OMP_NUM_THREADS=$t VLLM_CPU_OMP_THREADS_BIND=0-$last \
    taskset -c 0-$last timeout 2400 $V/bin/python bench_cpu_vllm.py \
     --tag "vllm-bf16-t$t" --out vllm_cpu_bench.jsonl 2>&1 | grep -E '"(tag|pp_tps|tg_tps|init_s)"'
done
