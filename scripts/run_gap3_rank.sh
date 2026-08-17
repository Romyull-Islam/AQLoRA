#!/bin/bash
# Llama-3.2-1B rank sweep, deferred behind the frontier timing.
set -u
cd "$(dirname "$0")"
PY=./venv/bin/python
STAMP() { date '+%m-%d %H:%M:%S'; }
AQQ="--mode aqslora --sensitive-frac 0.20 --adapter-frac 1.0"

echo "[$(STAMP)] waiting for the frontier timing to finish"
while ! grep -q 'FRONTIER-TIMING-DONE' frontier-timing.log 2>/dev/null; do sleep 300; done
echo "[$(STAMP)] frontier timing done; starting rank sweep"

free_gpu() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
             awk -F', ' '$2 < 2000 {print $1; exit}'; }
# 300s stagger, not 90s: model load takes longer than 90s, so a shorter gap lets
# several jobs pick the same card before any of them shows up as occupied.
wait_gpu() { local g; while :; do g=$(free_gpu); [ -n "$g" ] && { echo "$g"; return; }; sleep 240; done; }

for r in 16 32; do
  for spec in "aqslora:$AQQ" "qlora:--mode qlora" "loftq:--mode loftq" "rslora:--mode rslora"; do
    m=${spec%%:*}; f=${spec#*:}
    G=$(wait_gpu)
    out=./results-rank-llama-$m-r$r
    [ -f "$out/report.json" ] && { echo "[$(STAMP)]   $m r$r exists, skip"; continue; }
    ( env CUDA_VISIBLE_DEVICES=$G $PY finetune.py --model meta-llama/Llama-3.2-1B $f \
        --task commonsense --epochs 1 --batch-size 6 --rank $r \
        --speed-profile --gpus "$G" --output-dir "$out" \
      && env CUDA_VISIBLE_DEVICES=$G $PY evaluate.py --adapter-dir "$out" --gpus "$G" --commonsense \
      && echo "RANK-llama-$m-r$r-DONE" ) > "./rank-llama-$m-r$r.log" 2>&1 &
    echo "[$(STAMP)]   llama $m r=$r -> GPU$G"
    sleep 300
  done
done
wait
echo "[$(STAMP)] GAP3-RANK-DONE"
