#!/bin/bash
set -u
PY=./venv/bin/python
cd "$(dirname "$0")/.."
launch() { # method rank gpu
  m=$1; r=$2; g=$3
  case $m in
    rslora) F="--mode rslora --speed-profile" ;;
    loftq)  F="--mode loftq --speed-profile" ;;
    lora16) F="--mode lora16 --speed-profile" ;;
  esac
  out=./results-rank-phi15-$m-r$r
  ( $PY finetune.py --model microsoft/phi-1_5 $F --task commonsense --rank $r \
      --epochs 1 --batch-size 6 --gpus $g --output-dir $out \
    && $PY evaluate.py --adapter-dir $out --gpus $g --commonsense \
    && echo "RANK-$m-r$r-DONE" ) > ./rank-$m-r$r.log 2>&1 &
  echo "  $m r=$r -> GPU$g (pid $!)"
  sleep 5
}
launch "$@"
