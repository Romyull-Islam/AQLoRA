#!/bin/bash
# Llama-3.2-3B timing, retry, with a stronger floor.
set -u
cd "$(dirname "$0")"
PY=./venv/bin/python
REPS=${1:-7}
STAMP() { date '+%m-%d %H:%M:%S'; }

echo "[$(STAMP)] waiting for our own work to drain"
while ps -eo args | grep -qE "[v]env/bin/python (finetune|evaluate)\.py"; do sleep 300; done

OUT=./timing-llama-3.2-3b-retry; rm -rf "$OUT"; mkdir -p "$OUT"
find_clean() { nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
               awk -F', ' '$2 < 120 && $3 < 5 {print $1; exit}'; }
echo "[$(STAMP)] waiting for a clean card"
while :; do GPU=$(find_clean); [ -n "$GPU" ] && break; sleep 300; done
echo "[$(STAMP)] using GPU$GPU"

AQQ="--mode aqslora --sensitive-frac 0.20 --adapter-frac 1.0"
AQS="--mode aqslora --sensitive-frac 0.20 --adapter-frac 0.85 --placement depth"
# qlora and qlora-b are the same flags on purpose: their gap is the noise floor
METHODS=("qlora:--mode qlora" "qlora-b:--mode qlora" "rslora:--mode rslora"
         "aq-q:$AQQ" "aq-s:$AQS" "lora16:--mode lora16")

for i in $(seq 1 "$REPS"); do
  for spec in "${METHODS[@]}"; do
    n=${spec%%:*}; f=${spec#*:}
    before=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)
    if [ "$before" -gt 120 ]; then
      echo "[$(STAMP)] SKIP $n rep$i (${before} MiB resident)"; continue
    fi
    env CUDA_VISIBLE_DEVICES=$GPU $PY timing.py --model meta-llama/Llama-3.2-3B --gpus "$GPU" \
        --max-steps 240 --warmup-skip 40 --speed-profile $f \
        --output-dir "$OUT/$n-r$i" --tag "$n-r$i" > "$OUT/$n-r$i.log" 2>&1
    echo "[$(STAMP)]   $n rep$i (card ${before} MiB)"
  done
done
echo "[$(STAMP)] 3B-RETRY-DONE"
