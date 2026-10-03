#!/bin/bash
# The seed-42 placement runs of Table 15 used rsLoRA scaling; the seed-43 runs did
# not. This adds the depth-matched control to the seed-42, rsLoRA-scaled column.
set -u
cd "$(dirname "$0")"
PY=./venv/bin/python
STAMP() { date '+%m-%d %H:%M:%S'; }
G=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$1!=0 && $2<500 {print $1; exit}')
OUT=./results-e3-phi15-depthmatched05-rs
echo "[$(STAMP)] depthmatched seed 42 +rs -> GPU$G"
( env CUDA_VISIBLE_DEVICES=$G $PY finetune.py --model microsoft/phi-1_5 \
    --mode aqslora --sensitive-frac 0.20 --adapter-frac 0.5 --placement depthmatched \
    --rslora-scaling --seed 42 --speed-profile --task commonsense --epochs 1 --batch-size 6 \
    --gpus $G --output-dir $OUT \
  && env CUDA_VISIBLE_DEVICES=$G $PY evaluate.py --adapter-dir $OUT --gpus $G --commonsense \
  && echo "DMRS-DONE" ) > ./e3-depthmatched-s42-rs.log 2>&1
echo "[$(STAMP)] DMRS-ALL-DONE"
