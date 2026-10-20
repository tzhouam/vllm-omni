#!/bin/bash
# Memory, sized from the model instead of from a server default.
# Spark-X2.5 is 28 layers x 2 KV heads x 256 head_dim x 2 (K,V) x 2 B
# = 56 KiB/token, so a 2048-token single-stream deployment needs 112 MiB of KV
# cache. The CPU default asks for 4 GiB.
set -u
HERE=/data/zhoutaichang/embedding_infer/analysis/experiments/spark_edge
V=/data/zhoutaichang/embedding_infer/.venvs/omni-cpu
M=/data/zhoutaichang/embedding_infer/models/spark2_5-gptq-g128
OUT=$HERE/init_memory.jsonl
cd $HERE
run () {  # tag kv_gib max_len [kv_bytes]
  local extra=""
  [ -n "${4:-}" ] && extra="--kv-cache-memory-bytes $4"
  VLLM_CPU_KVCACHE_SPACE=$2 VLLM_OMNI_SPARK_GATE_REDUCTION=1 VLLM_OMNI_CPU_FUSED_NORMS=0 \
  KMP_BLOCKTIME=infinite OMP_WAIT_POLICY=active \
  LD_PRELOAD="$V/lib/python3.12/site-packages/vllm/libs/libtcmalloc_minimal.so.4:$V/lib/libiomp5.so" \
  CUDA_VISIBLE_DEVICES= VLLM_TARGET_DEVICE=cpu VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  OMP_NUM_THREADS=24 VLLM_CPU_OMP_THREADS_BIND=32-55 \
    taskset -c 32-55 timeout 1800 $V/bin/python bench_init_memory.py --model $M \
      --tag "$1" --max-model-len $3 $extra --out $OUT 2>&1 \
    | grep -E '"(tag|init_s|rss_mb|rss_peak_mb|tokens_generated|kv_cache_memory_bytes)"'
}
echo "#### server default: 4 GiB of KV for a 112 MiB need"
run "mem-kv4gib"      4 2048
echo "#### KV sized to a 2048-token context"
run "mem-kv128mib"    1 2048 $((128*1024*1024))
echo "#### KV sized to a 1024-token context"
run "mem-kv64mib"     1 1024 $((64*1024*1024))
echo "#### MEMORY SWEEP DONE"
