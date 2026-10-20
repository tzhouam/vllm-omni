#!/bin/bash
# Init time and memory, the two things that decide whether this can be served
# from an edge box. Sequential; each row changes exactly one thing.
set -u
HERE=/data/zhoutaichang/embedding_infer/analysis/experiments/spark_edge
V=/data/zhoutaichang/embedding_infer/.venvs/omni-cpu
M=/data/zhoutaichang/embedding_infer/models/spark2_5-gptq-g128
SCRATCH=/tmp/claude-2005/-data-zhoutaichang-embedding-infer/e695f57b-0163-4287-b1be-1d0ec9330768/scratchpad
OUT=$HERE/init_memory.jsonl
CORES=32-55; T=24
cd $HERE

run () {  # tag cache_root shim kv_gib max_len
  VLLM_CACHE_ROOT=$2 VLLM_OMNI_AOT_CACHE_SHIM=$3 VLLM_CPU_KVCACHE_SPACE=$4 \
  VLLM_OMNI_SPARK_GATE_REDUCTION=1 VLLM_OMNI_CPU_FUSED_NORMS=0 \
  KMP_BLOCKTIME=infinite OMP_WAIT_POLICY=active \
  LD_PRELOAD="$V/lib/python3.12/site-packages/vllm/libs/libtcmalloc_minimal.so.4:$V/lib/libiomp5.so" \
  CUDA_VISIBLE_DEVICES= VLLM_TARGET_DEVICE=cpu VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  OMP_NUM_THREADS=$T VLLM_CPU_OMP_THREADS_BIND=$CORES \
    taskset -c $CORES timeout 1800 $V/bin/python bench_init_memory.py --model $M \
      --tag "$1" --max-model-len $5 --out $OUT 2>&1 \
    | grep -E '"(tag|init_s|rss_mb|rss_peak_mb|tokens_generated|kvcache|aot_shim)"'
}

FRESH=$SCRATCH/vllm_cache_fresh_$$
rm -rf $FRESH
echo "#### 1. cold cache, shim on"
run "cold-shim-on"   $FRESH 1 4 2048
echo "#### 2. warm cache, shim on  (the point of the shim)"
run "warm-shim-on"   $FRESH 1 4 2048
echo "#### 3. warm cache, shim OFF (vLLM recompiles: the bug)"
run "warm-shim-off"  $FRESH 0 4 2048
echo "#### 4. warm, shim on, KV sized to the context instead of 4 GiB"
run "warm-kv-0.25"   $FRESH 1 0.25 2048
echo "#### 5. warm, shim on, KV 0.25 GiB and 1k context"
run "warm-kv-0.25-1k" $FRESH 1 0.25 1024
rm -rf $FRESH
echo "#### INIT MEMORY DONE"
