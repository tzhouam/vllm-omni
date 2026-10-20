#!/bin/bash
V=/data/zhoutaichang/embedding_infer/.venvs/omni-cpu
export LD_PRELOAD="$V/lib/python3.12/site-packages/vllm/libs/libtcmalloc_minimal.so.4:$V/lib/libiomp5.so"
export CUDA_VISIBLE_DEVICES= VLLM_TARGET_DEVICE=cpu VLLM_CPU_KVCACHE_SPACE=8
export OMP_NUM_THREADS=16 VLLM_CPU_OMP_THREADS_BIND=64-79
exec taskset -c 64-79 $V/bin/vllm serve XHToken/Spark-X2.5-1.7B \
  --dtype bfloat16 --max-model-len 8192 --max-num-seqs 4 \
  --tool-call-parser glm45 --reasoning-parser glm45 --enable-auto-tool-choice \
  --host 127.0.0.1 --port 8077
