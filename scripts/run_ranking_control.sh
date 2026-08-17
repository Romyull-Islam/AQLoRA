#!/bin/bash
# Does the NF4 error ranking do any downstream work?
set -u
cd "$(dirname "$0")"
PY=./venv/bin/python
STAMP() { date '+%m-%d %H:%M:%S'; }
free_gpu() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
             awk -F', ' '$2 < 2000 {print $1; exit}'; }
wait_gpu() { local g; while :; do g=$(free_gpu); [ -n "$g" ] && { echo "$g"; return; }; sleep 240; done; }

for sel in random bottom; do
  G=$(wait_gpu)
  out=./results-rank29-phi15-$sel
  [ -f "$out/report.json" ] && continue
  ( env CUDA_VISIBLE_DEVICES=$G $PY finetune.py --model microsoft/phi-1_5 \
      --mode aqslora --sensitive-frac 0.20 --adapter-frac 1.0 \
      --skip-modules-file protect_${sel}_29.json \
      --ppl-base --speed-profile --task commonsense --epochs 1 --batch-size 6 \
      --rank 8 --gpus "$G" --output-dir "$out" \
    && env CUDA_VISIBLE_DEVICES=$G $PY evaluate.py --adapter-dir "$out" --gpus "$G" --commonsense \
    && echo "RANKCTL-$sel-DONE" ) > "./rankctl-$sel.log" 2>&1 &
  echo "[$(STAMP)]   protect=$sel -> GPU$G"
  sleep 300
done
wait
echo "[$(STAMP)] RANKING-CONTROL-DONE"
