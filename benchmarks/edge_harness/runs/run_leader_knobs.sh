#!/bin/bash
# Two untried overhead knobs, applied to the leading config (W4A8 g128).
#   split_kv  vLLM splits the KV range across threads; at batch 1 with two KV
#             heads that may be fan-out for its own sake
#   nodetok   skip detokenization, which is pure host work per step
# Interleaved for the same reason as run_interleaved.sh.
set -u
HERE=/data/zhoutaichang/embedding_infer/analysis/experiments/spark_edge
V=/data/zhoutaichang/embedding_infer/.venvs/omni-cpu
MODEL=/data/zhoutaichang/embedding_infer/models/spark2_5-gptq-g128
OUT=$HERE/leader_knobs.jsonl
CORES=32-55; T=24
cd $HERE
arm () {  # tag, split_kv, extra-flags...
  local tag=$1 split=$2; shift 2
  VLLM_OMNI_SPARK_GATE_REDUCTION=1 VLLM_CPU_ATTN_SPLIT_KV=$split \
  KMP_BLOCKTIME=infinite OMP_WAIT_POLICY=active \
  LD_PRELOAD="$V/lib/python3.12/site-packages/vllm/libs/libtcmalloc_minimal.so.4:$V/lib/libiomp5.so" \
  CUDA_VISIBLE_DEVICES= VLLM_TARGET_DEVICE=cpu VLLM_CPU_KVCACHE_SPACE=4 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  OMP_NUM_THREADS=$T VLLM_CPU_OMP_THREADS_BIND=$CORES \
    taskset -c $CORES timeout 2400 $V/bin/python bench_cpu_vllm.py --model $MODEL \
      --tag "$tag" --reps 3 --out $OUT "$@" 2>&1 | grep -E '"(tag|pp_tps|tg_tps)"'
}
for pass in 1 2 3; do
  echo "======== pass $pass"
  arm "leader"          1
  arm "leader-nosplitkv" 0
  arm "leader-nodetok"  1 --no-detokenize
done
echo "#### LEADER KNOBS DONE"
