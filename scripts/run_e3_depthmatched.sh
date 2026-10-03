#!/bin/bash
# Controlled ablation for the density rule: random adapters with the density
# rule's per-third depth profile, phi-1.5, alpha=.5, both seeds of Table 15.
set -u
cd "$(dirname "$0")"
PY=./venv/bin/python
STAMP() { date '+%m-%d %H:%M:%S'; }
run () {   # run <seed> <outdir> <gpu>
  ( env CUDA_VISIBLE_DEVICES=$3 $PY finetune.py --model microsoft/phi-1_5 \
      --mode aqslora --sensitive-frac 0.20 --adapter-frac 0.5 --placement depthmatched \
      --seed $1 --speed-profile --task commonsense --epochs 1 --batch-size 6 \
      --gpus $3 --output-dir $2 \
    && env CUDA_VISIBLE_DEVICES=$3 $PY evaluate.py --adapter-dir $2 --gpus $3 --commonsense \
    && echo "DEPTHMATCHED-s$1-DONE" ) > ./e3-depthmatched-s$1.log 2>&1 &
  echo "[$(STAMP)] depthmatched seed $1 -> GPU$3"
}
run 42 ./results-e3-phi15-depthmatched05 2
run 43 ./results-e3s43-phi15-depthmatched5 3
wait
echo "[$(STAMP)] DEPTHMATCHED-ALL-DONE"
