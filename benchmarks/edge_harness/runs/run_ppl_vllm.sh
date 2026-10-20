#!/bin/bash
set -u
HERE=/data/zhoutaichang/embedding_infer/analysis/experiments/spark_edge
V=/data/zhoutaichang/embedding_infer/.venvs/omni-cpu
cd $HERE
run () {
  echo "######## $1"
  KMP_BLOCKTIME=infinite OMP_WAIT_POLICY=ACTIVE \
  LD_PRELOAD="$V/lib/python3.12/site-packages/vllm/libs/libtcmalloc_minimal.so.4:$V/lib/libiomp5.so" \
  CUDA_VISIBLE_DEVICES= VLLM_TARGET_DEVICE=cpu VLLM_CPU_KVCACHE_SPACE=4 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  OMP_NUM_THREADS=32 VLLM_CPU_OMP_THREADS_BIND=0-31 \
    taskset -c 0-31 timeout 2400 $V/bin/python ppl_vllm.py --model "$2" --corpus ppl_corpus.txt \
      --window 512 --tag "$1" --out $HERE/ppl_vllm.jsonl 2>&1 | grep -E '"(tag|ppl|scored_tokens|chunks)"'
}
run "vllm-bf16"     XHToken/Spark-X2.5-1.7B
run "vllm-int4-g64"  /data/zhoutaichang/embedding_infer/models/spark2_5-int4-g64
run "vllm-int4-g128" /data/zhoutaichang/embedding_infer/models/spark2_5-int4-g128
echo "#### PPL DONE"
