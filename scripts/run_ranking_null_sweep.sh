#!/bin/bash
# Is the NF4-error ranking better than random across models and budgets?
# One ranking_null run per (model, budget); each compares top-K and bottom-K
# against ten budget-matched random draws.
set -u
cd "$(dirname "$0")"
PY=./venv/bin/python
STAMP() { date '+%m-%d %H:%M:%S'; }

# model:tag:k  -- k is round(rho * linear layers) for rho = .10 .20 .30
# layer counts from Table 4: phi-1.5 144, Llama-3.2-1B 112, Phi-3-mini 128
CONFIGS=(
  "microsoft/phi-1_5:phi15:14"
  "microsoft/phi-1_5:phi15:43"
  "meta-llama/Llama-3.2-1B:llama-3.2-1b:11"
  "meta-llama/Llama-3.2-1B:llama-3.2-1b:22"
  "meta-llama/Llama-3.2-1B:llama-3.2-1b:34"
  "microsoft/Phi-3-mini-4k-instruct:phi-3-mini:13"
  "microsoft/Phi-3-mini-4k-instruct:phi-3-mini:26"
  "microsoft/Phi-3-mini-4k-instruct:phi-3-mini:38"
)

find_gpu() {   # perplexity is deterministic, so a busy neighbour only slows
  # this down; unlike the timing runs it needs free memory, not an idle card.
  # GPU0 excluded, it is rarely free for long on this node.
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits |
    awk -F', ' '$1 != 0 && $3-$2 > 14000 {print $1; exit}'
}

for spec in "${CONFIGS[@]}"; do
  M=${spec%%:*}; rest=${spec#*:}; T=${rest%%:*}; K=${rest##*:}
  OUT="ranking_null_${T}_k${K}.json"
  [ -f "$OUT" ] && { echo "[$(STAMP)] $OUT already done"; continue; }
  GPU=""
  while [ -z "$GPU" ]; do GPU=$(find_gpu); [ -z "$GPU" ] && sleep 300; done
  echo "[$(STAMP)] $T k=$K on GPU$GPU"
  env CUDA_VISIBLE_DEVICES=$GPU $PY ranking_null_distribution.py \
      --model "$M" --k "$K" --n 10 --out "$OUT" > "rn-${T}-k${K}.log" 2>&1
  echo "[$(STAMP)] $T k=$K done -> $OUT"
done
echo "[$(STAMP)] RANKING-NULL-SWEEP-DONE"
