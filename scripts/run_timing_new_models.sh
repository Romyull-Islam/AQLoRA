#!/bin/bash
# Timing sweeps for the two new models, using the corrected window rule.
set -u
cd "$(dirname "$0")"
PY=./venv/bin/python
REPS=${1:-7}
STAMP() { date '+%m-%d %H:%M:%S'; }

echo "[$(STAMP)] waiting for all accuracy work to drain"
while ps -eo args | grep -qE "[v]env/bin/python (finetune|evaluate)\.py"; do sleep 300; done
echo "[$(STAMP)] drained"

find_clean() { nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
               awk -F', ' '$2 < 120 && $3 < 5 {print $1; exit}'; }

AQQ="--mode aqslora --sensitive-frac 0.20 --adapter-frac 1.0"
AQS="--mode aqslora --sensitive-frac 0.20 --adapter-frac 0.85 --placement depth"
# rslora and loftq are compute-identical to qlora and give the noise floor
METHODS=("qlora:--mode qlora" "rslora:--mode rslora" "loftq:--mode loftq"
         "aq-q:$AQQ" "aq-s:$AQS" "lora16:--mode lora16")

sweep () {   # sweep <model> <tag>
  local MODEL=$1 TAG=$2 OUT=./timing-$2
  rm -rf "$OUT"; mkdir -p "$OUT"
  local GPU
  echo "[$(STAMP)] waiting for a clean card for $TAG"
  while :; do GPU=$(find_clean); [ -n "$GPU" ] && break; sleep 300; done
  echo "[$(STAMP)] $TAG on GPU$GPU"
  for i in $(seq 1 "$REPS"); do
    for spec in "${METHODS[@]}"; do
      local n=${spec%%:*} f=${spec#*:}
      local before; before=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)
      if [ "$before" -gt 120 ]; then
        echo "[$(STAMP)] SKIP $n rep$i (${before} MiB resident)"; continue
      fi
      env CUDA_VISIBLE_DEVICES=$GPU $PY timing.py --model "$MODEL" --gpus "$GPU" \
          --max-steps 240 --warmup-skip 40 --speed-profile $f \
          --output-dir "$OUT/$n-r$i" --tag "$n-r$i" > "$OUT/$n-r$i.log" 2>&1
      echo "[$(STAMP)]   $TAG $n rep$i (card ${before} MiB)"
    done
  done
  echo "[$(STAMP)] TIMING-$TAG-DONE"
  $PY robust_timing.py "$OUT" 2>&1 | tail -14
}

sweep meta-llama/Llama-3.2-3B llama-3.2-3b
sweep microsoft/Phi-3-mini-4k-instruct phi-3-mini
echo "[$(STAMP)] NEW-MODEL-TIMING-DONE"
