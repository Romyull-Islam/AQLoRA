#!/bin/bash
# Protocol-parity: contested methods under their NATIVE recipes (paged 8-bit
# optimizer, dropout 0.05, no length-grouping) instead of the shared levers.
# NOTE: CUDA_VISIBLE_DEVICES is set explicitly per process. finetune.py uses
# os.environ.setdefault, so an inherited value would silently win over --gpus.
set -u
PY=./venv/bin/python
cd "$(dirname "$0")/.."
launch() { m=$1; g=$2; extra=$3
  out=./results-parity-phi15-$m
  ( env CUDA_VISIBLE_DEVICES=$g $PY finetune.py --model microsoft/phi-1_5 $extra \
      --task commonsense --rank 8 --epochs 1 --batch-size 6 --gpus $g \
      --optim paged_adamw_8bit --lora-dropout 0.05 --output-dir $out \
    && env CUDA_VISIBLE_DEVICES=$g $PY evaluate.py --adapter-dir $out --gpus $g --commonsense \
    && echo "PARITY-$m-DONE" ) > ./parity-$m.log 2>&1 &
  echo "  $m -> physical GPU $g (pid $!)"; sleep 5
}
launch qlora   1 "--mode qlora"
launch loftq   5 "--mode loftq"
launch qdora   6 "--mode dora"
launch aqslora 0 "--mode aqslora --sensitive-frac 0.20 --adapter-frac 1.0"
