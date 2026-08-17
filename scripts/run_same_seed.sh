#!/bin/bash
# Same seed, twice more. Separates seed variance from run nondeterminism.
set -u
cd "$(dirname "$0")"
PY=./venv/bin/python
STAMP() { date '+%m-%d %H:%M:%S'; }
free_gpu() { nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits |
             awk -F', ' '{f=$3-$2; if(f>10000 && f>bf){bf=f;bi=$1}} END{print bi}'; }
wait_gpu() { local g; while :; do g=$(free_gpu); [ -n "$g" ] && { echo "$g"; return; }; sleep 240; done; }

for rep in r2 r3; do
  out=./results-cs-phi15-aqslora-q-nors-s42$rep
  [ -f "$out/report.json" ] && continue
  G=$(wait_gpu)
  ( env CUDA_VISIBLE_DEVICES=$G $PY finetune.py --model microsoft/phi-1_5 \
      --mode aqslora --sensitive-frac 0.20 --adapter-frac 1.0 \
      --task commonsense --epochs 1 --batch-size 6 --rank 8 --seed 42 \
      --speed-profile --gpus "$G" --output-dir "$out" \
    && env CUDA_VISIBLE_DEVICES=$G $PY evaluate.py --adapter-dir "$out" --gpus "$G" --commonsense \
    && echo "SAMESEED-$rep-DONE" ) > "./sameseed-$rep.log" 2>&1 &
  echo "[$(STAMP)]   seed42 repeat $rep -> GPU$G"
  sleep 300
done
wait
echo "[$(STAMP)] SAME-SEED-DONE"
