#!/bin/bash
# E3 placement grid, second seed. Accuracy only (contention does not affect it).
# Waits for free GPUs so we do not pile onto other users' jobs.
set -u
PY=./venv/bin/python
cd "$(dirname "$0")/.."
free_gpu() {  # echo index of a GPU under 1500 MiB, else nothing
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
    awk -F', ' '$2 < 1500 {print $1; exit}'
}
launch() { m=$1; a=$2; g=$3
  out=./results-e3s43-phi15-$m$a
  ( env CUDA_VISIBLE_DEVICES=$g $PY finetune.py --model microsoft/phi-1_5 \
      --mode aqslora --sensitive-frac 0.20 --adapter-frac 0.$a --placement $m \
      --seed 43 --speed-profile --task commonsense --epochs 1 --batch-size 6 \
      --gpus $g --output-dir $out \
    && env CUDA_VISIBLE_DEVICES=$g $PY evaluate.py --adapter-dir $out --gpus $g --commonsense \
    && echo "E3S43-$m$a-DONE" ) > ./e3s43-$m$a.log 2>&1 &
  echo "  $m 0.$a -> GPU$g"
}
for spec in sparsity:5 random:5 weightnorm:5 sparsity:7 random:7 weightnorm:7; do
  m=${spec%%:*}; a=${spec##*:}
  while :; do g=$(free_gpu); [ -n "$g" ] && break; sleep 300; done
  launch $m $a $g
  sleep 90
done
echo "E3-SEED43-ALL-LAUNCHED"
