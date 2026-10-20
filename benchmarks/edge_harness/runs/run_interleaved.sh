#!/bin/bash
# Interleaved, not sequential.
#
# A sequential grid on this machine is not trustworthy: another tenant on the
# same socket swung the llama.cpp control from 86.7 to 60.4 tok/s across one
# pass, and the vLLM baseline alone varied 57.3 -> 68.0 between two identical
# runs. So arms are rotated and each is run three times; the figure taken for
# each arm is its best pass, which is the least-contaminated estimate of what
# that configuration does when the machine cooperates.
set -u
HERE=/data/zhoutaichang/embedding_infer/analysis/experiments/spark_edge
V=/data/zhoutaichang/embedding_infer/.venvs/omni-cpu
G=/data/zhoutaichang/.cache/huggingface/hub/models--XHToken--Spark-X2.5-1.7B-GGUF/snapshots/23e1fcac55e7dd71e4c12a23723cc228ba0e5e85
B=/data/zhoutaichang/embedding_infer/llama.cpp/build/bin/llama-bench
M=/data/zhoutaichang/embedding_infer/models
OUT=$HERE/interleaved.jsonl
CORES=32-55; T=24
cd $HERE

vllm_arm () {
  VLLM_OMNI_SPARK_GATE_REDUCTION=$3 KMP_BLOCKTIME=infinite OMP_WAIT_POLICY=active \
  LD_PRELOAD="$V/lib/python3.12/site-packages/vllm/libs/libtcmalloc_minimal.so.4:$V/lib/libiomp5.so" \
  CUDA_VISIBLE_DEVICES= VLLM_TARGET_DEVICE=cpu VLLM_CPU_KVCACHE_SPACE=4 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  OMP_NUM_THREADS=$T VLLM_CPU_OMP_THREADS_BIND=$CORES \
    taskset -c $CORES timeout 2400 $V/bin/python bench_cpu_vllm.py --model "$2" \
      --tag "$1" --reps 3 --out $OUT 2>&1 | grep -E '"(tag|pp_tps|tg_tps)"'
}

for pass in 1 2 3; do
  echo "======== pass $pass"
  taskset -c $CORES $B -m $G/Spark-X2.5-1.7B-Q4_K_M.gguf -p 512 -n 128 -t $T -o json 2>/dev/null \
      > $HERE/il_llamacpp_p$pass.json
  python3 -c "
import json;d=json.load(open('$HERE/il_llamacpp_p$pass.json'))
print('   llamacpp-q4km', {('pp512' if r['n_prompt'] else 'tg128'): round(1e9*(r['n_prompt'] or r['n_gen'])/r['avg_ns'],2) for r in d})"
  vllm_arm "w4a16-g64-nogate" $M/spark2_5-int4-g64  0
  vllm_arm "w4a16-g64"        $M/spark2_5-int4-g64  1
  vllm_arm "w4a16-g128"       $M/spark2_5-int4-g128 1
  vllm_arm "w4a8-g64"         $M/spark2_5-gptq-g64  1
  vllm_arm "w4a8-g128"        $M/spark2_5-gptq-g128 1
done
echo "#### INTERLEAVED DONE"
