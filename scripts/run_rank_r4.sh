#!/bin/bash
set -u
PY=./venv/bin/python
cd "$(dirname "$0")/.."
launch() { m=$1; g=$2
  case $m in
    aqslora) F="--mode aqslora --sensitive-frac 0.20 --adapter-frac 1.0 --speed-profile" ;;
    qlora)   F="--mode qlora --speed-profile" ;;
    rslora)  F="--mode rslora --speed-profile" ;;
    loftq)   F="--mode loftq --speed-profile" ;;
  esac
  out=./results-rank-phi15-$m-r4
  ( $PY finetune.py --model microsoft/phi-1_5 $F --task commonsense --rank 4 \
      --epochs 1 --batch-size 6 --gpus $g --output-dir $out \
    && $PY evaluate.py --adapter-dir $out --gpus $g --commonsense \
    && echo "RANK-$m-r4-DONE" ) > ./rank-$m-r4.log 2>&1 &
  echo "  $m r=4 -> GPU$g (pid $!)"; sleep 5
}
launch aqslora 3
launch qlora   4
launch rslora  5
launch loftq   6
