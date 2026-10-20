#!/bin/bash
# Final matched comparison, strictly sequential on the same pinned cores with
# nothing else running: llama.cpp Q4_K_M against both vLLM 4-bit paths.
#   W4A16  group-wise weights, bf16 activations, tinygemm kernel (cpu_int4)
#   W4A8   group-wise weights, int8 activations, AMX kernel (gptq)
# W4A8 is the arithmetic llama.cpp's Q4_K_M actually does (Q4_K x Q8_K).
#
# The OMP team is kept hot between regions (KMP_BLOCKTIME/OMP_WAIT_POLICY),
# which is the one thread-fan-out knob that survived measurement. Forcing
# inductor's small elementwise kernels single-threaded (cpp.min_chunk_size)
# was tried and dropped: -6% decode and -19% prefill, because it also
# serialises the kernels that have real work at prefill.
set -u
HERE=/data/zhoutaichang/embedding_infer/analysis/experiments/spark_edge
V=/data/zhoutaichang/embedding_infer/.venvs/omni-cpu
G=/data/zhoutaichang/.cache/huggingface/hub/models--XHToken--Spark-X2.5-1.7B-GGUF/snapshots/23e1fcac55e7dd71e4c12a23723cc228ba0e5e85
B=/data/zhoutaichang/embedding_infer/llama.cpp/build/bin/llama-bench
M=/data/zhoutaichang/embedding_infer/models
W4A16=$M/spark2_5-int4-g64
W4A8=$M/spark2_5-gptq-g64
OUT=$HERE/final_grid.jsonl
cd $HERE

run_vllm () {  # tag model threads [--cc json] [env...]
  local tag=$1 model=$2 t=$3; shift 3
  local cc=""
  if [ "${1:-}" = "--cc" ]; then cc="$2"; shift 2; fi
  local last=$((t-1))
  env "$@" LD_PRELOAD="$V/lib/python3.12/site-packages/vllm/libs/libtcmalloc_minimal.so.4:$V/lib/libiomp5.so" \
    CUDA_VISIBLE_DEVICES= VLLM_TARGET_DEVICE=cpu VLLM_CPU_KVCACHE_SPACE=4 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    OMP_NUM_THREADS=$t VLLM_CPU_OMP_THREADS_BIND=0-$last \
    taskset -c 0-$last timeout 2400 $V/bin/python bench_cpu_vllm.py --model "$model" \
      --tag "$tag" --reps 3 --out $OUT ${cc:+--compilation-config "$cc"} 2>&1 \
    | grep -E '"(tag|pp_tps|tg_tps|rss_mb)"'
}

llamacpp () {
  local t=$1 last=$((t-1))
  taskset -c 0-$last $B -m $G/Spark-X2.5-1.7B-Q4_K_M.gguf -p 512 -n 128 -t $t -o json 2>/dev/null \
      > $HERE/final_llamacpp_t$t.json
  python3 -c "
import json;d=json.load(open('$HERE/final_llamacpp_t$t.json'))
print('   llama.cpp Q4_K_M t=$t ', {('pp512' if r['n_prompt'] else 'tg128'): round(1e9*(r['n_prompt'] or r['n_gen'])/r['avg_ns'],2) for r in d})"
}

for t in 32 16; do
  echo "######## threads=$t"
  llamacpp $t
  run_vllm "w4a16-t$t"        $W4A16 $t KMP_BLOCKTIME=infinite OMP_WAIT_POLICY=active
  run_vllm "w4a8-t$t"         $W4A8  $t KMP_BLOCKTIME=infinite OMP_WAIT_POLICY=active
  run_vllm "w4a16-g128-t$t"   $M/spark2_5-int4-g128 $t KMP_BLOCKTIME=infinite OMP_WAIT_POLICY=active
done
echo "######## bf16 control t=32"
run_vllm "bf16-t32" XHToken/Spark-X2.5-1.7B 32 KMP_BLOCKTIME=infinite OMP_WAIT_POLICY=active
echo "######## FINAL GRID DONE"
