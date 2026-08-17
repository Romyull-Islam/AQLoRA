#!/bin/bash
# Take the variance comparison from three seeds to six.
set -u
cd "$(dirname "$0")"
PY=./venv/bin/python
STAMP() { date '+%m-%d %H:%M:%S'; }
free_gpu() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
             awk -F', ' '$2 < 2000 {print $1; exit}'; }
wait_gpu() { local g; while :; do g=$(free_gpu); [ -n "$g" ] && { echo "$g"; return; }; sleep 240; done; }

AQQ="--mode aqslora --sensitive-frac 0.20 --adapter-frac 1.0"
AQRS="--mode aqslora --sensitive-frac 0.20 --adapter-frac 1.0 --rslora-scaling"

for seed in 45 46 47; do
  for spec in "aqslora-q-nors:$AQQ" "qlora:--mode qlora" "aqslora-rs:$AQRS"; do
    m=${spec%%:*}; f=${spec#*:}
    out=./results-cs-phi15-$m-s$seed
    [ -f "$out/report.json" ] && { echo "[$(STAMP)]   $m s$seed exists, skip"; continue; }
    G=$(wait_gpu)
    ( env CUDA_VISIBLE_DEVICES=$G $PY finetune.py --model microsoft/phi-1_5 $f \
        --task commonsense --epochs 1 --batch-size 6 --rank 8 --seed $seed \
        --speed-profile --gpus "$G" --output-dir "$out" \
      && env CUDA_VISIBLE_DEVICES=$G $PY evaluate.py --adapter-dir "$out" --gpus "$G" --commonsense \
      && echo "SEED6-$m-$seed-DONE" ) > "./seed6-$m-s$seed.log" 2>&1 &
    echo "[$(STAMP)]   $m seed$seed -> GPU$G"
    sleep 300   # longer than model load, so the next pick sees this card taken
  done
done
wait
echo "[$(STAMP)] SEEDS6-DONE"
