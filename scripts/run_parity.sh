#!/bin/bash
set -u
PY=${PYTHON:-python}
cd "$(dirname "$0")/.."
launch() { m=$1; g=$2; extra=$3
  out=./results-parity-phi15-$m
  # native recipe: paged 8-bit optimizer, dropout 0.05, no length-grouping
  ( "$PY" finetune.py --model microsoft/phi-1_5 $extra --task commonsense \
      --rank 8 --epochs 1 --batch-size 6 --gpus $g \
      --optim paged_adamw_8bit --lora-dropout 0.05 \
      --output-dir $out \
    && "$PY" evaluate.py --adapter-dir $out --gpus $g --commonsense \
    && echo "PARITY-$m-DONE" ) > ./parity-$m.log 2>&1 &
  echo "  $m -> GPU$g (pid $!)"; sleep 5
}
launch qlora   6 "--mode qlora"
launch loftq   7 "--mode loftq"
launch qdora   3 "--mode dora"
launch aqslora 4 "--mode aqslora --sensitive-frac 0.20 --adapter-frac 1.0"
