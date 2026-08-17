#!/bin/bash
# Bring the rsLoRA baseline to six seeds.
set -u
cd "$(dirname "$0")"
PY=./venv/bin/python
STAMP() { date '+%m-%d %H:%M:%S'; }
free_gpu() { nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits |
             awk -F', ' '{f=$3-$2; if(f>10000 && f>bf){bf=f;bi=$1}} END{print bi}'; }
wait_gpu() { local g; while :; do g=$(free_gpu); [ -n "$g" ] && { echo "$g"; return; }; sleep 240; done; }

for seed in 45 46 47; do
  out=./results-cs-phi15-rslora-s$seed
  [ -f "$out/report.json" ] && continue
  G=$(wait_gpu)
  ( env CUDA_VISIBLE_DEVICES=$G $PY finetune.py --model microsoft/phi-1_5 --mode rslora \
      --task commonsense --epochs 1 --batch-size 6 --rank 8 --seed $seed \
      --speed-profile --gpus "$G" --output-dir "$out" \
    && env CUDA_VISIBLE_DEVICES=$G $PY evaluate.py --adapter-dir "$out" --gpus "$G" --commonsense \
    && echo "RSLORA-$seed-DONE" ) > "./rslora-s$seed.log" 2>&1 &
  echo "[$(STAMP)]   rslora seed$seed -> GPU$G"
  sleep 300
done
wait
echo "[$(STAMP)] RSLORA-SEEDS-DONE"
