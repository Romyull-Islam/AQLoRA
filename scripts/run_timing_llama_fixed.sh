#!/bin/bash
# Fifth Llama timing attempt, with the protocol bug fixed.
set -u
cd "$(dirname "$0")"
PY=./venv/bin/python
REPS=${1:-7}
STAMP() { date '+%m-%d %H:%M:%S'; }
echo "[$(STAMP)] waiting for the accuracy queue to drain"
while ps -eo args | grep -qE "[v]env/bin/python (finetune|evaluate)\.py"; do sleep 300; done
echo "[$(STAMP)] drained"
OUT=./timing-llama-fixedwindow; rm -rf "$OUT"; mkdir -p "$OUT"
find_clean() { nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
               awk -F', ' '$2 < 120 && $3 < 5 {print $1; exit}'; }
while :; do GPU=$(find_clean); [ -n "$GPU" ] && break; sleep 300; done
echo "[$(STAMP)] using GPU$GPU"
AQQ="--mode aqslora --sensitive-frac 0.20 --adapter-frac 1.0"
AQS="--mode aqslora --sensitive-frac 0.20 --adapter-frac 0.85 --placement depth"
METHODS=("qlora:--mode qlora" "rslora:--mode rslora" "loftq:--mode loftq"
         "aq-q:$AQQ" "aq-s:$AQS" "lora16:--mode lora16")
for i in $(seq 1 "$REPS"); do
  for spec in "${METHODS[@]}"; do
    n=${spec%%:*}; f=${spec#*:}
    before=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)
    [ "$before" -gt 120 ] && { echo "[$(STAMP)] SKIP $n rep$i (${before} MiB)"; continue; }
    env CUDA_VISIBLE_DEVICES=$GPU $PY timing.py --model meta-llama/Llama-3.2-1B --gpus "$GPU" \
        --max-steps 240 --warmup-skip 40 --speed-profile $f \
        --output-dir "$OUT/$n-r$i" --tag "$n-r$i" > "$OUT/$n-r$i.log" 2>&1
    echo "[$(STAMP)]   $n rep$i (card ${before} MiB)"
  done
done
echo "[$(STAMP)] LLAMA-FIXEDWINDOW-DONE"
