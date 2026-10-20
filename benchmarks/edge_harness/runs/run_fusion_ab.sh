#!/bin/bash
# Fused CPU RMSNorm kernels on/off, interleaved, on both 4-bit paths.
set -u
HERE=/data/zhoutaichang/embedding_infer/analysis/experiments/spark_edge
V=/data/zhoutaichang/embedding_infer/.venvs/omni-cpu
M=/data/zhoutaichang/embedding_infer/models
OUT=$HERE/fusion_ab.jsonl
CORES=32-55; T=24
cd $HERE
arm () {  # tag model fused
  VLLM_OMNI_CPU_FUSED_NORMS=$3 VLLM_OMNI_SPARK_GATE_REDUCTION=1 \
  KMP_BLOCKTIME=infinite OMP_WAIT_POLICY=active \
  LD_PRELOAD="$V/lib/python3.12/site-packages/vllm/libs/libtcmalloc_minimal.so.4:$V/lib/libiomp5.so" \
  CUDA_VISIBLE_DEVICES= VLLM_TARGET_DEVICE=cpu VLLM_CPU_KVCACHE_SPACE=4 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  OMP_NUM_THREADS=$T VLLM_CPU_OMP_THREADS_BIND=$CORES \
    taskset -c $CORES timeout 2400 $V/bin/python bench_cpu_vllm.py --model "$2" \
      --tag "$1" --reps 3 --out $OUT 2>&1 | grep -E '"(tag|pp_tps|tg_tps|init_s|rss_mb)"'
}
for pass in 1 2 3; do
  echo "======== pass $pass"
  arm "w4a8-g128-nofuse" $M/spark2_5-gptq-g128 0
  arm "w4a8-g128-fused"  $M/spark2_5-gptq-g128 1
  arm "w4a16-g128-fused" $M/spark2_5-int4-g128 1
done
echo "#### FUSION AB DONE"
