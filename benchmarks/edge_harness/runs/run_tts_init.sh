#!/bin/bash
# Does the AOT cache-load fix generalise past Spark? The shim sits in the
# shared torch.compiler path, so any CPU model vLLM compiles should stop
# recompiling on every start -- Qwen3-TTS is the harder case because the
# pipeline has two engine cores, each with its own compiled artifact.
set -u
OMNI=/data/zhoutaichang/embedding_infer/vllm-omni
V=/data/zhoutaichang/embedding_infer/.venvs/omni-cpu
SCRATCH=/tmp/claude-2005/-data-zhoutaichang-embedding-infer/e695f57b-0163-4287-b1be-1d0ec9330768/scratchpad
FRESH=$SCRATCH/vllm_cache_tts_$$
rm -rf $FRESH
cd $OMNI
run () {  # tag shim
  echo "#### $1 (shim=$2)"
  VLLM_CACHE_ROOT=$FRESH VLLM_OMNI_AOT_CACHE_SHIM=$2 \
  VLLM_CPU_KVCACHE_SPACE=4 HF_HUB_OFFLINE=1 \
  LD_PRELOAD="$V/lib/python3.12/site-packages/vllm/libs/libtcmalloc_minimal.so.4:$V/lib/libiomp5.so" \
  CUDA_VISIBLE_DEVICES= VLLM_TARGET_DEVICE=cpu VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  OMP_NUM_THREADS=24 VLLM_CPU_OMP_THREADS_BIND=32-55 \
    taskset -c 32-55 timeout 2400 $V/bin/python benchmarks/tts/stream_latency_bench.py \
      --model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --query-type CustomVoice \
      --txt-prompts benchmarks/tts/prompts_12.txt --warmup 1 --init-only \
      --tag "$1" --init-timeout 1800 --stage-init-timeout 1200 2>&1 \
    | grep -iE "init engine|ready|took|init_s|Compiling model again|Directly load AOT|error|Traceback" | tail -8
}
run tts-cold-shim-on  1
run tts-warm-shim-on  1
run tts-warm-shim-off 0
rm -rf $FRESH
echo "#### TTS INIT DONE"
