#!/bin/bash
# Speed cost of the residual protection tier.
set -u
cd "$(dirname "$0")"
PY=./venv/bin/python
REPS=${1:-7}
STAMP() { date '+%m-%d %H:%M:%S'; }

echo "[$(STAMP)] waiting for the accuracy queue to drain"
while pgrep -f "finetune.py|evaluate.py" > /dev/null; do sleep 300; done
echo "[$(STAMP)] queue drained"

OUT=./timing-resid-phi15
rm -rf "$OUT"; mkdir -p "$OUT"     # never inherit another sweep's files

find_clean() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
    awk -F', ' '$2 < 120 && $3 < 5 {print $1; exit}'
}
echo "[$(STAMP)] waiting for a clean card"
while :; do GPU=$(find_clean); [ -n "$GPU" ] && break; sleep 300; done
echo "[$(STAMP)] using GPU$GPU"

AQQ="--mode aqslora --sensitive-frac 0.20 --adapter-frac 1.0"
# qlora is the reference; rslora and loftq are compute-identical to it and give
# the floor; aq-q is the published fp16 tier; the last two are the new tier.
METHODS=("qlora:--mode qlora"
         "rslora:--mode rslora"
         "loftq:--mode loftq"
         "aq-q:$AQQ --protect-tier fp16"
         "aq-r20:$AQQ --protect-tier residual"
         "aq-r40:--mode aqslora --sensitive-frac 0.40 --adapter-frac 1.0 --protect-tier residual")

for i in $(seq 1 "$REPS"); do
  for spec in "${METHODS[@]}"; do
    n=${spec%%:*}; f=${spec#*:}
    before=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)
    if [ "$before" -gt 120 ]; then
      echo "[$(STAMP)] SKIP $n rep$i: card not clean (${before} MiB)"; continue
    fi
    env CUDA_VISIBLE_DEVICES=$GPU $PY timing.py --model microsoft/phi-1_5 --gpus "$GPU" \
        --max-steps 80 --warmup-skip 20 $f \
        --output-dir "$OUT/$n-r$i" --tag "$n-r$i" > "$OUT/$n-r$i.log" 2>&1
    echo "[$(STAMP)]   $n rep$i (card ${before} MiB before)"
  done
done
echo "[$(STAMP)] RESID-TIMING-DONE"
