#!/bin/bash
# AQLoRA-rs at the cheap protection budgets.
set -u
cd "$(dirname "$0")"
PY=./venv/bin/python
STAMP() { date '+%m-%d %H:%M:%S'; }

echo "[$(STAMP)] waiting for the frontier timing to finish"
while ! grep -q 'FRONTIER-TIMING-DONE' frontier-timing.log 2>/dev/null; do sleep 300; done
echo "[$(STAMP)] frontier timing done; starting AQLoRA-rs low-rho runs"

free_gpu() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
             awk -F', ' '$2 < 2000 {print $1; exit}'; }
wait_gpu() { local g; while :; do g=$(free_gpu); [ -n "$g" ] && { echo "$g"; return; }; sleep 240; done; }

for rho in 0.05 0.10; do
  G=$(wait_gpu)
  tag=$(echo $rho | tr -d '.')
  out=./results-rslow-phi15-$tag
  [ -f "$out/report.json" ] && continue
  ( env CUDA_VISIBLE_DEVICES=$G $PY finetune.py --model microsoft/phi-1_5 \
      --mode aqslora --sensitive-frac $rho --adapter-frac 1.0 --rslora-scaling \
      --speed-profile --task commonsense --epochs 1 --batch-size 6 --rank 8 \
      --gpus "$G" --output-dir "$out" \
    && env CUDA_VISIBLE_DEVICES=$G $PY evaluate.py --adapter-dir "$out" --gpus "$G" --commonsense \
    && echo "RSLOW-$rho-DONE" ) > "./rslow-$tag.log" 2>&1 &
  echo "[$(STAMP)]   aqslora-rs rho=$rho -> GPU$G"
  sleep 300   # longer than model load, so the next pick sees this card occupied
done
wait
echo "[$(STAMP)] RS-LOWRHO-DONE"
