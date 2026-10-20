#!/bin/bash
# Output comparison only: greedy text is load-independent, so this can run
# while the machine is busy.
set -u
HERE=/data/zhoutaichang/embedding_infer/analysis/experiments/spark_edge
V=/data/zhoutaichang/embedding_infer/.venvs/omni-cpu
G=/data/zhoutaichang/.cache/huggingface/hub/models--XHToken--Spark-X2.5-1.7B-GGUF/snapshots/23e1fcac55e7dd71e4c12a23723cc228ba0e5e85
BF16=/data/zhoutaichang/.cache/huggingface/hub/models--XHToken--Spark-X2.5-1.7B/snapshots/448e61eb392c00f2c403185c5b56d5e0665bfaab
cd $HERE
gen_vllm () {
  LD_PRELOAD="$V/lib/python3.12/site-packages/vllm/libs/libtcmalloc_minimal.so.4:$V/lib/libiomp5.so" \
  CUDA_VISIBLE_DEVICES= VLLM_TARGET_DEVICE=cpu VLLM_CPU_KVCACHE_SPACE=4 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  OMP_NUM_THREADS=24 VLLM_CPU_OMP_THREADS_BIND=88-111 \
    taskset -c 88-111 timeout 3600 $V/bin/python quality_int4.py gen --engine vllm \
      --model "$1" --out "$2" 2>&1 | tail -2
}
gen_vllm "$BF16" quality_vllm_bf16.json
gen_vllm /data/zhoutaichang/embedding_infer/models/spark2_5-int4-g64 quality_vllm_w4a16.json
gen_vllm /data/zhoutaichang/embedding_infer/models/spark2_5-gptq-g64 quality_vllm_w4a8.json
taskset -c 88-111 $V/bin/python quality_int4.py gen --engine llamacpp --threads 24 \
   --gguf $G/Spark-X2.5-1.7B.gguf --out quality_llamacpp_bf16.json 2>&1 | tail -1
taskset -c 88-111 $V/bin/python quality_int4.py gen --engine llamacpp --threads 24 \
   --gguf $G/Spark-X2.5-1.7B-Q4_K_M.gguf --out quality_llamacpp_q4km.json 2>&1 | tail -1
echo "### generations done"
for pair in "quality_vllm_bf16.json quality_vllm_w4a16.json vllm-W4A16-tinygemm" \
            "quality_vllm_bf16.json quality_vllm_w4a8.json vllm-W4A8-gptq" \
            "quality_llamacpp_bf16.json quality_llamacpp_q4km.json llamacpp-Q4_K_M"; do
  set -- $pair
  $V/bin/python quality_int4.py cmp --ref $1 --got $2 --label $3 --out quality_cmp_$3.json 2>&1 | tail -6
done
echo "### QUALITY DONE"
