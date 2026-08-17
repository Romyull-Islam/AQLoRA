#!/bin/bash
# Settle whether LoftQ actually fails at 7B, or whether we simply never ran it.
set -u
cd "$(dirname "$0")"
PY=./venv/bin/python
STAMP() { date '+%m-%d %H:%M:%S'; }
echo "[$(STAMP)] waiting for a card with room for a 7B run"
while :; do
  G=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits |
      awk -F', ' '{f=$3-$2; if(f>26000 && f>bf){bf=f;bi=$1}} END{print bi}')
  [ -n "$G" ] && break; sleep 300
done
echo "[$(STAMP)] LoftQ 7B on GPU$G"
out=./results-cs-qwen2.5-7b-loftq-s42
env CUDA_VISIBLE_DEVICES=$G $PY finetune.py --model Qwen/Qwen2.5-7B --mode loftq \
    --task commonsense --epochs 1 --batch-size 6 --rank 8 --seed 42 \
    --speed-profile --gpus "$G" --output-dir "$out" \
  && env CUDA_VISIBLE_DEVICES=$G $PY evaluate.py --adapter-dir "$out" --gpus "$G" --commonsense \
  && echo "LOFTQ-7B-DONE"
echo "[$(STAMP)] finished with exit $?"
