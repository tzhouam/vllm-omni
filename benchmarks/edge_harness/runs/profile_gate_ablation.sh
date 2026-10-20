#!/bin/bash
# Attribute the gate delta per-op rather than by throughput: the ablation's
# 4.1 ms/token is six times what the aten::mm it removes was measured to cost,
# so either the profile or the throughput number is misleading.
set -u
HERE=/data/zhoutaichang/embedding_infer/analysis/experiments/spark_edge
V=/data/zhoutaichang/embedding_infer/.venvs/omni-cpu
M=/data/zhoutaichang/embedding_infer/models/spark2_5-int4-g64
cd $HERE
for g in 0 1; do
  echo "#### gate_reduction=$g"
  VLLM_OMNI_SPARK_GATE_REDUCTION=$g KMP_BLOCKTIME=infinite OMP_WAIT_POLICY=active \
  LD_PRELOAD="$V/lib/python3.12/site-packages/vllm/libs/libtcmalloc_minimal.so.4:$V/lib/libiomp5.so" \
  CUDA_VISIBLE_DEVICES= VLLM_TARGET_DEVICE=cpu VLLM_CPU_KVCACHE_SPACE=4 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  OMP_NUM_THREADS=24 VLLM_CPU_OMP_THREADS_BIND=32-55 \
    taskset -c 32-55 timeout 2400 $V/bin/python profile_cpu_step.py --model $M \
      --out $HERE/profile_gate_$g.json 2>&1 | sed -n '/^{/,$p' | head -30
done
echo "#### GATE PROFILE DONE"
