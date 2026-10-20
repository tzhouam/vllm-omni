#!/bin/bash
set -u
HERE=/data/zhoutaichang/embedding_infer/analysis/experiments/spark_edge
V=/data/zhoutaichang/embedding_infer/.venvs/omni-cpu
cd $HERE
for M in ${MODELS:-g64 g128}; do
  for S in "short:ref.json:ref_logits.npy:2048" "long:ref_long.json:ref_long.npy:4096" "tools:ref_tools.json:ref_tools.npy:2048"; do
    name=${S%%:*}; r=${S#*:}; rj=${r%%:*}; r2=${r#*:}; rn=${r2%%:*}; ml=${r2##*:}
    echo "######## int4-$M $name"
    LD_PRELOAD="$V/lib/python3.12/site-packages/vllm/libs/libtcmalloc_minimal.so.4:$V/lib/libiomp5.so" \
    CUDA_VISIBLE_DEVICES= VLLM_TARGET_DEVICE=cpu VLLM_CPU_KVCACHE_SPACE=4 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    OMP_NUM_THREADS=32 VLLM_CPU_OMP_THREADS_BIND=0-31 \
      taskset -c 0-31 timeout 1800 $V/bin/python cmp_vllm.py \
        --model /data/zhoutaichang/embedding_infer/models/spark2_5-int4-$M \
        --ref $rj --ref-logits $rn --max-model-len $ml \
        --out $HERE/parity_int4_${M}_${name}.json 2>&1 \
      | grep -E '"(greedy_token_match|first_token_agree|top20_logprob_overlap|top20_logprob_max_abs_diff)"'
  done
done
echo "#### PARITY DONE"
