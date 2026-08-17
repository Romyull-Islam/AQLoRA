#!/bin/bash
# Seeds for the Llama-3.2-1B comparison, which is where the accuracy claim lives.
set -u
cd "$(dirname "$0")"
PY=./venv/bin/python
STAMP() { date '+%m-%d %H:%M:%S'; }
AQQ="--mode aqslora --sensitive-frac 0.20 --adapter-frac 1.0"
free_gpu() { nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits |
             awk -F', ' '{f=$3-$2; if(f>12000 && f>bf){bf=f;bi=$1}} END{print bi}'; }
wait_gpu() { local g; while :; do g=$(free_gpu); [ -n "$g" ] && { echo "$g"; return; }; sleep 240; done; }

for seed in 43 44 45; do
  for spec in "aqslora-q-nors:$AQQ" "qlora:--mode qlora"; do
    m=${spec%%:*}; f=${spec#*:}
    out=./results-cs-llama-3.2-1b-$m-s$seed
    [ -f "$out/report.json" ] && continue
    G=$(wait_gpu)
    ( env CUDA_VISIBLE_DEVICES=$G $PY finetune.py --model meta-llama/Llama-3.2-1B $f \
        --task commonsense --epochs 1 --batch-size 6 --rank 8 --seed $seed \
        --speed-profile --gpus "$G" --output-dir "$out" \
      && env CUDA_VISIBLE_DEVICES=$G $PY evaluate.py --adapter-dir "$out" --gpus "$G" --commonsense \
      && echo "LLAMA-$m-$seed-DONE" ) > "./llama-$m-s$seed.log" 2>&1 &
    echo "[$(STAMP)]   llama $m seed$seed -> GPU$G"
    sleep 300
  done
done
wait
echo "[$(STAMP)] LLAMA-SEEDS-DONE"
