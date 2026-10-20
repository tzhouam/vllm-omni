#!/bin/bash
# Per-op profile instead of throughput: the two 4-bit paths differ by ~10% and
# this machine's throughput noise is ~30%, so the GEMM self time is the only
# way to rank them.
set -u
HERE=/data/zhoutaichang/embedding_infer/analysis/experiments/spark_edge
V=/data/zhoutaichang/embedding_infer/.venvs/omni-cpu
M=/data/zhoutaichang/embedding_infer/models
cd $HERE
prof () {
  VLLM_OMNI_SPARK_GATE_REDUCTION=1 VLLM_OMNI_CPU_FUSED_NORMS=$3 \
  KMP_BLOCKTIME=infinite OMP_WAIT_POLICY=active \
  LD_PRELOAD="$V/lib/python3.12/site-packages/vllm/libs/libtcmalloc_minimal.so.4:$V/lib/libiomp5.so" \
  CUDA_VISIBLE_DEVICES= VLLM_TARGET_DEVICE=cpu VLLM_CPU_KVCACHE_SPACE=4 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  OMP_NUM_THREADS=24 VLLM_CPU_OMP_THREADS_BIND=32-55 \
    taskset -c 32-55 timeout 1800 $V/bin/python profile_cpu_step.py --model "$2" \
      --out $HERE/profile_$1.json 2>&1 | sed -n '/^{/,$p' | head -12
}
for pass in 1 2; do
  echo "==== pass $pass"
  prof "w4a16_g128_p$pass" $M/spark2_5-int4-g128 0
  prof "w4a8_g128_p$pass"  $M/spark2_5-gptq-g128 0
done
echo "#### QUANT PROFILE DONE"
