#!/bin/bash
G=/data/zhoutaichang/.cache/huggingface/hub/models--XHToken--Spark-X2.5-1.7B-GGUF/snapshots/23e1fcac55e7dd71e4c12a23723cc228ba0e5e85
B=/data/zhoutaichang/embedding_infer/llama.cpp/build/bin/llama-bench
for t in 4 8 16 32; do
  last=$((t-1))
  taskset -c 0-$last $B -m $G/Spark-X2.5-1.7B.gguf -m $G/Spark-X2.5-1.7B-Q8_0.gguf -m $G/Spark-X2.5-1.7B-Q4_K_M.gguf \
     -p 512 -n 128 -t $t -o json 2>/dev/null
done
