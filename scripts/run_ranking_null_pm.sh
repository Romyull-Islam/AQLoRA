#!/bin/bash
# Llama-3.2-1B again, with the random control matched on parameters rather than
# on layer count: at fixed count the draws carry up to 11% more memory.
set -u
cd "$(dirname "$0")"
PY=./venv/bin/python
STAMP() { date '+%m-%d %H:%M:%S'; }
find_gpu() {
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits |
    awk -F', ' '$1 != 0 && $3-$2 > 14000 {print $1; exit}'
}
for K in 11 22 34; do
  OUT="ranking_null_llama-3.2-1b-pm_k${K}.json"
  [ -f "$OUT" ] && { echo "[$(STAMP)] $OUT done"; continue; }
  GPU=""; while [ -z "$GPU" ]; do GPU=$(find_gpu); [ -z "$GPU" ] && sleep 300; done
  echo "[$(STAMP)] llama-3.2-1b params-matched k=$K on GPU$GPU"
  env CUDA_VISIBLE_DEVICES=$GPU $PY ranking_null_distribution.py \
      --model meta-llama/Llama-3.2-1B --k "$K" --n 10 --match params \
      --out "$OUT" > "rnpm-llama-k${K}.log" 2>&1
  echo "[$(STAMP)] k=$K done"
done
echo "[$(STAMP)] PM-SWEEP-DONE"
